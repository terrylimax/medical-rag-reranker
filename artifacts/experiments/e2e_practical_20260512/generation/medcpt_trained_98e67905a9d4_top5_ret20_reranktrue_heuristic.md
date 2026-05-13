# Generation Evaluation

## Summary

- `avg_answer_chars`: 542.3333
- `avg_answer_relevance`: 0.7278
- `avg_citation_presence_rate`: 0.8100
- `avg_context_relevance`: 0.7376
- `avg_empty_answer`: 0.0000
- `avg_end_to_end_latency_ms`: 7845.1104
- `avg_generation_latency_ms`: 5744.1882
- `avg_groundedness`: 0.6502
- `avg_insufficient_context`: 0.0567
- `avg_num_retrieved_docs`: 5.0000
- `avg_rerank_latency_ms`: 2023.9478
- `avg_retrieval_latency_ms`: 76.8100
- `avg_supported_citation_rate`: 0.6167
- `avg_unsupported_citation_rate`: 0.1933
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
3. `medquad_ans_0003102-1` (score=0.0001) - What are the signs and symptoms of I cell disease? The Human Phenotype Ontology provides the following list of signs and symptoms for I cell disease. If the information is available, the table below includes how often...
4. `medquad_ans_0006511-1` (score=0.0001) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the informatio...
5. `medquad_ans_0000259-2__dup1` (score=0.0001) - What are the signs and symptoms of Alpha-thalassemia x-linked intellectual disability syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Alpha-thalassemia x-linked intellectua...

**Answer**:

The symptoms of X-linked lymphoproliferative syndrome 1 include cellular immunodeficiency (90%), decreased antibody level in blood (50%), hepatomegaly (50%), lymphadenopathy (50%), lymphoma (50%), splenomegaly (50%), anemia (7.5%), and other features such as fulminant hepatitis, encephalitis, IgG deficiency, immunodeficiency, increased IgM level, meningitis, pancytopenia, recurrent pharyngitis, reduced natural killer cell activity, thrombocytopenia, and X-linked recessive inheritance [medquad_ans_0006510-1].

**Supported citations**: `medquad_ans_0006510-1`
**Unsupported citations**: _none_

### Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Scores**: context_relevance=0.400, groundedness=0.113, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000123-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000618-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000692-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000802-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether early-onset primary dystonia is inherited. While the context describes autosomal recessive inheritance patterns in general, it does not mention early-onset primary dystonia or its mode of inheritance. Therefore, based on the given information, the inheritance of early-onset primary dystonia cannot be determined.

Answer: Insufficient context to determine if early-onset primary dystonia is inherited.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Scores**: context_relevance=0.714, groundedness=0.952, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000650-2` (score=0.9988) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
2. `medquad_ans_0000989-2` (score=0.9987) - This condition affects an estimated 1 in 50,000 people.
3. `medquad_ans_0000874-2` (score=0.9984) - This condition affects an estimated 1 in 8,500 females.
4. `medquad_ans_0000359-2` (score=0.9977) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
5. `medquad_ans_0000397-2__dup1` (score=0.9876) - The tapeworms that cause taeniasis (Taenia saginata, T. solium, and T. asiatica) are found worldwide. Eating raw or undercooked beef or pork is the primary risk factor for acquiring taeniasis. Persons who don't eat ra...

**Answer**:

Persons who eat raw or undercooked beef or pork are at risk for taeniasis [medquad_ans_0000397-2__dup1]. Infections with Taenia saginata occur where contaminated raw beef is eaten, particularly in Eastern Europe, Russia, eastern Africa, and Latin America [medquad_ans_0000397-2__dup1]. Taenia solium infections are more prevalent in under-developed communities with poor sanitation and where raw or undercooked pork is consumed, with higher rates seen in Latin America, Eastern Europe, sub-Saharan Africa, India, and Asia [medquad_ans_0000397-2__dup1]. Taenia solium taeniasis is also seen in the United States, typically among Latin American immigrants [medquad_ans_0000397-2__dup1]. Additionally, people with poor hygiene who

**Supported citations**: `medquad_ans_0000397-2__dup1`
**Unsupported citations**: _none_

### Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000194-2__dup1` (score=0.9994) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
2. `medquad_ans_0003442-2` (score=0.0002) - What are the signs and symptoms of L1 syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for L1 syndrome. If the information is available, the table below includes how often the s...
3. `medquad_ans_0003657-1` (score=0.0001) - What are the signs and symptoms of Loeys-Dietz syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Loeys-Dietz syndrome type 2. If the information is available, the tabl...
4. `medquad_ans_0000193-2__dup1` (score=0.0001) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 1. If the information is availab...
5. `medquad_ans_0006001-2` (score=0.0001) - What are the signs and symptoms of Takayasu arteritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Takayasu arteritis. If the information is available, the table below includes h...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy. [medquad_ans_0000194-2__dup1]

**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

### Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003421-1` (score=0.9993) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0003418-1` (score=0.0002) - What are the signs and symptoms of Kosztolanyi syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kosztolanyi syndrome. If the information is available, the table below includ...
3. `medquad_ans_0003423-1` (score=0.0001) - What are the signs and symptoms of Kozlowski Celermajer Tink syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kozlowski Celermajer Tink syndrome. If the information is avail...
4. `medquad_ans_0003849-1` (score=0.0001) - What are the signs and symptoms of McKusick Kaufman syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for McKusick Kaufman syndrome. If the information is available, the table be...
5. `medquad_ans_0001335-1` (score=0.0001) - What are the signs and symptoms of Chudley Rozdilsky syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Chudley Rozdilsky syndrome. If the information is available, the table ...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

### Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000908-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000572-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000232-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000945-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention hemophilia or specify how it is inherited. Therefore, based on the given information, it is not possible to determine if hemophilia is inherited. [No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.970, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.9994) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often t...
2. `medquad_ans_0006355-2` (score=0.0002) - What are the signs and symptoms of Vici syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Vici syndrome. If the information is available, the table below includes how often t...
3. `medquad_ans_0002458-2` (score=0.0001) - What are the signs and symptoms of Fryns syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns syndrome. If the information is available, the table below includes how often...
4. `medquad_ans_0004677-1` (score=0.0001) - What are the signs and symptoms of PAGOD syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for PAGOD syndrome. If the information is available, the table below includes how often...
5. `medquad_ans_0002945-1` (score=0.0001) - What are the signs and symptoms of Hurler syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Hurler syndrome. If the information is available, the table below includes how oft...

**Answer**:

The symptoms of IVIC syndrome include hearing impairment (90%), limitation of joint mobility (90%), short stature (90%), strabismus (90%), abnormal dermatoglyphics (50%), aplasia/hypoplasia of the thumb (50%), radioulnar synostosis (50%), scoliosis (50%), synostosis of carpal bones (50%), triphalangeal thumb (50%), abnormality of the clavicle (7.5%), arrhythmia (7.5%), leukocytosis (7.5%), preaxial hand polydactyly (7.5%), thrombocytopenia (7.5%), and urogenital fistula (7.5%) [medquad_ans_0003284-1].

**Supported citations**: `medquad_ans_0003284-1`
**Unsupported citations**: _none_

### Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Scores**: context_relevance=0.833, groundedness=0.544, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0004747-1` (score=0.9758) - Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It can occur at any age, but is usually diagnosed in young adulthood. People ...
2. `medquad_ans_0000169-3` (score=0.9660) - Episodes eventually decrease in frequency and intensity over the course of eight to 12 years.
3. `medquad_ans_0000001-7__dup1` (score=0.9465) - Topics
4. `medquad_ans_0000775-1` (score=0.6112) - Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. The disorder affects red blood cells (erythrocytes), which carry oxygen; white bloo...
5. `medquad_ans_0000874-2` (score=0.0173) - This condition affects an estimated 1 in 8,500 females.

**Answer**:

Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes), which protect the body from infection; and platelets (thrombocytes), which are involved in blood clotting. PNH can occur at any age, but is usually diagnosed in young adulthood. People with PNH have sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses on the body, such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to a deficiency of red blood cells (hemolytic anemia) and symptoms such as fatigue, weakness, abnormally pale skin (pallor), shortness of breath, and an increased heart rate. Affected individuals may pass

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.630, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000010-20` (score=0.8451) - Yes. In July 2008, the U.S. Preventive Services Task Force made its strongest ever recommendation for colorectal cancer screening: it suggested that all adults between ages 50 and 75 get screened, or tested, for the d...
2. `medquad_ans_0000010-10` (score=0.6004) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the li...
3. `medquad_ans_0000221-1__dup2` (score=0.4409) - The colon and rectum are part of the large intestine. Colorectal cancer occurs when tumors form in the lining of the large intestine. It is common in both men and women. The risk of developing colorectal cancer rises ...
4. `medquad_ans_0000010-23` (score=0.0277) - Several types of surgery are available for someone with colorectal cancer. If the cancer is found at a very early stage, the doctor may remove it without cutting through the abdominal wall. Instead, the doctor may put...
5. `medquad_ans_0000037_4-1` (score=0.0212) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...

**Answer**:

Colorectal cancer is cancer of the colon or rectum, which are parts of the large intestine. It occurs when malignant tumors form in the lining of the large intestine, also known as the large bowel [medquad_ans_0000010-10]. It is common in both men and women and the risk of developing it rises after age 50 [medquad_ans_00000221-1__dup2].

**Supported citations**: `medquad_ans_0000010-10`
**Unsupported citations**: `medquad_ans_00000221-1__dup2`

### Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Scores**: context_relevance=0.833, groundedness=0.525, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003317-1` (score=0.9712) - Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz ...
2. `medquad_ans_0000544-1` (score=0.9670) - Junctional epidermolysis bullosa (JEB) is one of the major forms of epidermolysis bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. Blisters and skin erosions form in...
3. `medquad_ans_0002121-1` (score=0.0042) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main ty...
4. `medquad_ans_0002122-1` (score=0.0017) - Epidermolysis bullosa acquisita (EBA) is a rare autoimmune disorder that causes the skin to blister in response to minor injury. Common areas of blistering include the hands, feet, knees, elbows, and buttocks. It can ...
5. `medquad_ans_0000544-2` (score=0.0003) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.

**Answer**:

Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz type and the Non-Herlitz type. The Herlitz type is very severe, with blistering present from birth or early infancy, affecting large regions of the body and mucous membranes, leading to complications such as chronic malnutrition, scarring, granulation tissue formation, and increased susceptibility to infections. Many infants with Herlitz JEB do not survive beyond the first year of life. The Non-Herlitz type includes several subtypes that cause mild to severe blistering, often limited to the hands, feet, knees, and elbows, and may improve after the newborn period. JEB is inherited in an autosomal recessive pattern and is caused by

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Scores**: context_relevance=0.700, groundedness=0.948, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0000668-3` (score=0.9991) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same ...
2. `medquad_ans_0000342-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
3. `medquad_ans_0000948-4` (score=0.0001) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000650-3` (score=0.0001) - Mutations in the MUT, MMAA, MMAB, MMADHC, and MCEE genes cause methylmalonic acidemia. The long term effects of methylmalonic acidemia depend on which gene is mutated and the severity of the mutation. About 60 percent...
5. `medquad_ans_0000956-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...

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
4. `medquad_ans_0000445-1__dup1` (score=0.0284) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...
5. `medquad_ans_0000755-4` (score=0.0067) - Ornithine transcarbamylase deficiency is an X-linked disorder. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes. A characte...

**Answer**:

Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The nervous system is especially sensitive to the effects of excess ammonia. Ornithine translocase deficiency varies widely in its severity and age of onset. An infant with ornithine translocase deficiency may be lacking in energy (lethargic) or refuse to eat, or have poorly controlled breathing or body temperature. Some babies with this disorder may experience seizures or unusual body movements, or go into a coma. Episodes of illness may coincide with the introduction of high-protein formulas or solid foods into the diet. In most affected individuals, signs and symptoms of ornithine translocase deficiency do not appear until later in life. Later-onset forms of ornithine translocase deficiency are usually less severe than the

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Scores**: context_relevance=0.714, groundedness=0.912, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000414-1__dup2` (score=0.9976) - Frequently Asked Questions (FAQs) Fact Sheets
2. `medquad_ans_0000586-1__dup2` (score=0.9970) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state ha...
3. `medquad_ans_0000362-1__dup2` (score=0.9954) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
4. `medquad_ans_0000590-1__dup2` (score=0.9948) - Summary : Part D is the name of Medicare's prescription drug coverage. It's insurance that helps people pay for prescription drugs. It is available to everyone who has Medicare. It provides protection if you pay high ...
5. `medquad_ans_0000431-1__dup1` (score=0.9945) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...

**Answer**:

Yes, there is information about financial assistance. Financial assistance may be available if you don't have health insurance or need help with costs not covered by insurance. Certain government programs and nonprofit organizations can provide such assistance. You can also discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2]. Additionally, Medicaid helps low-income individuals pay their medical bills, and each state may require partial payment for some services based on income, age, disability, pregnancy, or immigration status [medquad_ans_0000586-1__dup2].

**Supported citations**: `medquad_ans_0000362-1__dup2`, `medquad_ans_0000586-1__dup2`
**Unsupported citations**: _none_

### Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Scores**: context_relevance=0.800, groundedness=0.942, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000239-2__dup1` (score=0.9893) - TSEs tend to progress rapidly and usually culminate in death over the course of a few months to a few years.
2. `medquad_ans_0000258-2__dup3` (score=0.9836) - LCMV is most commonly recognized as causing neurological disease, as its name implies, though infection without symptoms or mild febrile illnesses are more common clinical manifestations. For infected persons who do b...
3. `medquad_ans_0004105-1` (score=0.8755) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About hal...
4. `medquad_ans_0000645-1__dup2` (score=0.0814) - The brain, spinal cord, and nerves make up the nervous system. Together they control all the workings of the body. When something goes wrong with a part of your nervous system, you can have trouble moving, speaking, s...
5. `medquad_ans_0000344-5__dup1` (score=0.0304) - If you suspect you have been infected, consult your health care provider immediately. Be sure to tell your health care provider if you have recently been exposed to raccoons or their feces. Diagnosis is difficult beca...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningitis), and other symptoms associated with meningitis [medquad_ans_0004105-1].

**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

### Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003319-3` (score=0.9871) - What causes juvenile amyotrophic lateral sclerosis? Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases the underlying gene abnormality...
2. `medquad_ans_0003331-3` (score=0.9829) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...
3. `medquad_ans_0000102-3__dup1` (score=0.0060) - The progression of the disorder is usually 10 years or longer.
4. `medquad_ans_0000650-2` (score=0.0020) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
5. `medquad_ans_0000989-2` (score=0.0013) - This condition affects an estimated 1 in 50,000 people.

**Answer**:

Juvenile amyotrophic lateral sclerosis is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases, the underlying gene abnormality cannot be determined. Juvenile ALS may be inherited in an autosomal dominant (as in ALS type 4) or autosomal recessive (as in ALS type 2) fashion [medquad_ans_0003319-3].

**Supported citations**: `medquad_ans_0003319-3`
**Unsupported citations**: _none_

### Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Scores**: context_relevance=0.889, groundedness=0.777, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0000001-7__dup1` (score=0.9936) - Topics
2. `medquad_ans_0000211-7` (score=0.9927) - Men should tell a health care professional, such as a family practice physician, a nurse, an internist, or a urologista doctor who specializes in urinary problemsthey have UI, even if they feel embarrassed. To diagnos...
3. `medquad_ans_0000219-11` (score=0.9849) - - Erectile dysfunction (ED) is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last...
4. `medquad_ans_0000219-1__dup4` (score=0.9725) - Erectile dysfunction\* is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last long ...
5. `medquad_ans_0000058-19` (score=0.9633) - Yes. Any of the symptoms caused by prostate cancer may also be due to enlargement of the prostate, which is not cancer. If you have any of the symptoms mentioned in question #10, see your doctor or a urologist to find...

**Answer**:

The provided context does not contain information about erectile dysfunction. Therefore, it is not possible to answer the question based on the given text. [medquad_ans_0000211-7] discusses urinary incontinence (UI) and related diagnostic steps, but does not address erectile dysfunction.

**Supported citations**: `medquad_ans_0000211-7`
**Unsupported citations**: _none_

### Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Scores**: context_relevance=0.889, groundedness=0.962, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000713-5` (score=0.9955) - These resources address the diagnosis or management of Nager syndrome: - Genetic Testing Registry: Nager syndrome - University of California San Francisco Medical Center These resources from MedlinePlus offer informat...
2. `medquad_ans_0000208-2__dup1` (score=0.9944) - Treatments for the myopathies depend on the disease or condition and specific causes. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders. Treatment for other disor...
3. `medquad_ans_0006267-3` (score=0.9702) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
4. `medquad_ans_0000438-5` (score=0.1473) - These resources address the diagnosis or management of Greenberg dysplasia: - Genetic Testing Registry: Greenberg dysplasia - Lurie Children's Hospital of Chicago: Fetal Skeletal Dysplasia These resources from Medline...
5. `medquad_ans_0000575-5` (score=0.0703) - These resources address the diagnosis or management of Laing distal myopathy: - Gene Review: Gene Review: Laing Distal Myopathy - Genetic Testing Registry: Myopathy, distal, 1 These resources from MedlinePlus offer in...

**Answer**:

Physical therapy, including early mobilization, regular stretching, and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation often become necessary in the first or second decade of life. Prevention of chest infections may be achieved with the use of antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that people with Ullrich congenital muscular dystrophy may benefit from cyclosporin A, although more studies are needed to confirm its benefits [medquad_ans_0006267-3].

**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

### Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Scores**: context_relevance=0.250, groundedness=0.129, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000043-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000528-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000197-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000949-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether microcephalic osteodysplastic primordial dwarfism type II is inherited. While several documents describe autosomal recessive inheritance patterns in general, no specific mention is made of microcephalic osteodysplastic primordial dwarfism type II. Therefore, based on the given information, the inheritance pattern for this specific condition cannot be determined.

Answer: Insufficient context to determine if microcephalic osteodysplastic primordial dwarfism type II is inherited.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000396-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000925-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000232-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000020-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Ochoa syndrome or its mode of inheritance. Therefore, it is not possible to determine based on the given information whether Ochoa syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.747, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000531-5` (score=0.9960) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management...
2. `medquad_ans_0001072-5` (score=0.0566) - These resources address the diagnosis or management of 16p11.2 deletion syndrome: - Gene Review: Gene Review: 16p11.2 Recurrent Microdeletion - Genetic Testing Registry: 16p11.2 deletion syndrome - Genetic Testing Reg...
3. `medquad_ans_0001071-5` (score=0.0163) - These resources address the diagnosis or management of 15q24 microdeletion: - Gene Review: Gene Review: 15q24 Microdeletion - Genetic Testing Registry: 15q24 deletion syndrome These resources from MedlinePlus offer in...
4. `medquad_ans_0001070-5` (score=0.0153) - These resources address the diagnosis or management of 15q13.3 microdeletion: - Gene Review: Gene Review: 15q13.3 Microdeletion - Genetic Testing Registry: 15q13.3 microdeletion syndrome These resources from MedlinePl...
5. `medquad_ans_0001099-5` (score=0.0142) - These resources address the diagnosis or management of 5q minus syndrome: - American Cancer Society: How are Myelodysplastic Syndromes Diagnosed? - Cancer.Net: MyelodysplasticSyndromes: Treatment Options - Genetic Tes...

**Answer**:

The provided context does not specify treatments for isodicentric chromosome 15 syndrome. While it mentions that resources from MedlinePlus cover diagnosis and management of various health conditions, including diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care, no specific treatments are outlined for isodicentric chromosome 15 syndrome [medquad_ans_0000531-5]. Therefore, based on the given information, the treatments for isodicentric chromosome 15 syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000531-5`
**Unsupported citations**: _none_

### Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.935, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006224-3` (score=0.9931) - What causes Troyer syndrome? Troyer syndrome is caused by mutations in the SPG20 gene. This gene gives the body instructions to make a protein called spartin, which is present in many body tissues, including those of ...
2. `medquad_ans_0005951-3` (score=0.0002) - What causes Swyer-James syndrome? The cause of Swyer-James syndrome is not completely understood. Most experts agree that the initial abnormality occurs in the distal bronchi (air tubes that bring air to and from the ...
3. `medquad_ans_0000272-4__dup1` (score=0.0002) - The NINDS supports research on genetic disorders such as the hereditary spastic paraplegias. A gene for Troyer syndrome has been identified and others may be identified in the future. Understanding how these genes cau...
4. `medquad_ans_0000964-2` (score=0.0001) - Swyer syndrome occurs in approximately 1 in 80,000 people.
5. `medquad_ans_0000998-2` (score=0.0001) - About 20 cases of Troyer syndrome have been reported in the Old Order Amish population of Ohio. It has not been found outside this population.

**Answer**:

Troyer syndrome is caused by mutations in the SPG20 gene. This gene provides instructions for making a protein called spartin, which is present in many body tissues, including those of the nervous system. It is thought that the syndrome is caused by a loss of function of the spartin protein, although the exact role of this protein in cellular functions is not fully understood [medquad_ans_0006224-3].

**Supported citations**: `medquad_ans_0006224-3`
**Unsupported citations**: _none_

### Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.923, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0000983-3` (score=0.9990) - Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. This gene provides instructions for making a protein that acts as a channel across cell membranes. This channel, known as CaV1....
2. `medquad_ans_0000159-3` (score=0.9948) - CPVT can result from mutations in two genes, RYR2 and CASQ2. RYR2 gene mutations cause about half of all cases, while mutations in the CASQ2 gene account for 1 percent to 2 percent of cases. In people without an ident...
3. `medquad_ans_0000662-3` (score=0.9940) - MELAS can result from mutations in one of several genes, including MT-ND1, MT-ND5, MT-TH, MT-TL1, and MT-TV. These genes are found in the DNA of cellular structures called mitochondria, which convert the energy from f...
4. `medquad_ans_0000467-3` (score=0.0219) - HMERF can be caused by a mutation in the TTN gene. This gene provides instructions for making a protein called titin. Titin plays an important role in muscles the body uses for movement (skeletal muscles) and in heart...
5. `medquad_ans_0000812-3` (score=0.0063) - Mutations in the SCN4A gene cause potassium-aggravated myotonia. The SCN4A gene provides instructions for making a protein that is critical for the normal function of skeletal muscle cells. For the body to move normal...

**Answer**:

Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. These mutations change the structure of CaV1.2 channels, causing them to stay open much longer than usual, which leads to an abnormal flow of calcium ions into cells. This results in an overload of calcium ions within cardiac muscle cells, altering the heart's rhythm and potentially causing arrhythmia [medquad_ans_0000983-3].

**Supported citations**: `medquad_ans_0000983-3`
**Unsupported citations**: _none_

### Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000061-11` (score=0.9910) - Rheumatoid arthritis can be difficult to diagnose in its early stages because the full range of symptoms develops over time, and only a few symptoms may be present in the early stages. As part of the diagnosis, your d...
2. `medquad_ans_0000049-3__dup3` (score=0.3534) - Pain and Stiffness in Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Osteoarthritis usually develops slowly and ca...
3. `medquad_ans_0004326-1` (score=0.0007) - What are the signs and symptoms of Negative rheumatoid factor polyarthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Negative rheumatoid factor polyarthritis. If the informat...
4. `medquad_ans_0005261-2` (score=0.0002) - What are the signs and symptoms of Reactive arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Reactive arthritis. If the information is available, the table below includes h...
5. `medquad_ans_0005361-1` (score=0.0002) - What are the signs and symptoms of Rheumatoid nodulosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Rheumatoid nodulosis. If the information is available, the table below includ...

**Answer**:

The symptoms of rheumatoid arthritis include swelling, warmth, pain, and limitations in joint motion throughout the body. Your doctor may also ask about the intensity of pain, how often it occurs, and what makes the pain better or worse [medquad_ans_0000061-11].

**Supported citations**: `medquad_ans_0000061-11`
**Unsupported citations**: _none_

### Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Scores**: context_relevance=0.800, groundedness=0.544, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002796-1` (score=0.9718) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is no...
2. `medquad_ans_0000149-1__dup3` (score=0.9287) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, d...
3. `medquad_ans_0000149-4__dup2` (score=0.0723) - In children, the risk of kidney damage leading to long-term problems may be as high as 15 percent, but kidney failure affects only about 1 percent of children with HSP.1 Up to 40 percent of adults with HSP will have C...
4. `medquad_ans_0000053-3__dup2` (score=0.0036) - Some diseases and conditions can disrupt the body's normal blood clotting process and lead to disseminated intravascular coagulation (DIC). These diseases and conditions include: Sepsis (an infection in the bloodstrea...
5. `medquad_ans_0000136-1__dup2` (score=0.0029) - Vasculitis (vas-kyu-LI-tis) is a condition that involves inflammation in the blood vessels. The condition occurs if your immune system attacks your blood vessels by mistake. This may happen as the result of an infecti...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). It occurs when small blood vessels in the body become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can affect the kidneys, digestive tract, and joints. It is most common in children between 2 and 6 years of age but can occur at any age. Most people recover completely, though kidney damage is the most likely long-term complication. In adults, HSP can lead to chronic kidney disease (CKD) and kidney failure, and a rare complication is intussusception of the bowel. Women with a history of HSP who become pregnant are at higher risk for high blood pressure and proteinuria during pregnancy [medquad_ans_0002

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Scores**: context_relevance=1.000, groundedness=0.967, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0003166-1` (score=0.9994) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information ...
2. `medquad_ans_0003104-1` (score=0.0001) - What are the signs and symptoms of ICF syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for ICF syndrome. If the information is available, the table below includes how often the...
3. `medquad_ans_0003163-1` (score=0.0001) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information ...
4. `medquad_ans_0006511-1` (score=0.0001) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the informatio...
5. `medquad_ans_0000579-2__dup1` (score=0.0001) - What are the signs and symptoms of Autosomal recessive hyper IgE syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal recessive hyper IgE syndrome. If the information ...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [medquad_ans_0003166-1].

**Supported citations**: `medquad_ans_0003166-1`
**Unsupported citations**: _none_

### Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.962, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000582-3` (score=0.9992) - Mutations in the FLNB gene cause Larsen syndrome. The FLNB gene provides instructions for making a protein called filamin B. This protein helps build the network of protein filaments (cytoskeleton) that gives structur...
2. `medquad_ans_0000896-3` (score=0.9991) - Senior-Lken syndrome can be caused by mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopic, finger...
3. `medquad_ans_0000606-3` (score=0.5271) - Mutations in the OCRL gene cause Lowe syndrome. The OCRL gene provides instructions for making an enzyme that helps modify fat (lipid) molecules called membrane phospholipids. By controlling the levels of specific mem...
4. `medquad_ans_0000193-3` (score=0.0535) - Coats plus syndrome results from mutations in the CTC1 gene. This gene provides instructions for making a protein that plays an important role in structures known as telomeres, which are found at the ends of chromosom...
5. `medquad_ans_0000398-1__dup1` (score=0.0235) - Genes are the building blocks of heredity. They are passed from parent to child. They hold DNA, the instructions for making proteins. Proteins do most of the work in cells. They move molecules from one place to anothe...

**Answer**:

The genetic changes related to Senior-Lken syndrome are mutations in one of at least five genes. These mutations likely lead to problems with the structure and function of cilia, which are microscopic, finger-like projections on cells involved in signaling pathways. The defective cilia disrupt important chemical signaling pathways within cells, contributing to the features of the disorder [medquad_ans_0000896-3].

**Supported citations**: `medquad_ans_0000896-3`
**Unsupported citations**: _none_

### Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Scores**: context_relevance=0.125, groundedness=0.774, answer_relevance=0.325

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
3. `medquad_ans_0000790-2` (score=0.9995) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
4. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000904-2` (score=0.9995) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.

**Answer**:

The incidence of lymphedema-distichiasis syndrome is estimated to affect 1 in 40,000 to 50,000 people [medquad_ans_0000195-2], and another estimate suggests it affects 1 in 50,000 people [medquad_ans_0000989-2]. Additionally, it is thought to affect approximately 1 in 35,000 to 50,000 newborns [medquad_ans_0000904-2]. Therefore, the condition affects an estimated range of 1 in 25,000 to 50,000 individuals, with specific estimates varying between 1 in 35,000 to 50,000 [medquad_ans_0000904-2]. [

**Supported citations**: `medquad_ans_0000195-2`, `medquad_ans_0000989-2`, `medquad_ans_0000904-2`
**Unsupported citations**: _none_

### Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Scores**: context_relevance=0.857, groundedness=0.883, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000399-1__dup1` (score=0.9977) - Summary : Genetic tests are tests on blood and other tissue to find genetic disorders. Over 2000 tests are available. Doctors use genetic tests for several reasons. These include - Finding genetic diseases in unborn b...
2. `medquad_ans_0000001-7__dup1` (score=0.9863) - Topics
3. `medquad_ans_0000397-1__dup1` (score=0.1860) - Summary : Genetic counseling provides information and support to people who have, or may be at risk for, genetic disorders. A genetic counselor meets with you to discuss genetic risks. The counseling may be for yourse...
4. `medquad_ans_0000067-17` (score=0.0118) - It is important to keep a copy of your medical records to share with any new doctors that you see. This information should contain the type of cancer you were diagnosed with, test results, and treatment details. It is...
5. `medquad_ans_0000869-1__dup1` (score=0.0101) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can brin...

**Answer**:

Yes, there is information about genetic testing. Genetic tests are conducted on blood and other tissues to identify genetic disorders. Over 2,000 tests are available, and doctors use them for various purposes, such as detecting genetic diseases in unborn babies, determining if individuals carry a gene that could be passed to their children, screening embryos for disease, testing adults before symptoms appear, making a diagnosis in someone with symptoms, and determining the best type or dose of medicine for a person [medquad_ans_0000399-1__dup1]. Genetic counseling can provide information about the benefits and risks of testing, and individuals may seek it if they have a personal or family history of genetic conditions, are planning a pregnancy after age 35, have a child with a genetic disorder, have experienced multiple pregnancy losses, or have had ultrasound or screening results suggesting a possible issue [medquad_ans_0000397-

**Supported citations**: `medquad_ans_0000399-1__dup1`
**Unsupported citations**: _none_

### Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.550, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000149-1` (score=0.8681) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxy...
2. `medquad_ans_0000030-1__dup2` (score=0.0268) - Arteriovenous malformations (AVMs) are abnormal, snarled tangles of blood vessels that cause multiple irregular connections between the arteries and veins. These malformations most often occur in the spinal cord and i...
3. `medquad_ans_0003238-1` (score=0.0137) - Intracranial arteriovenous malformations (AVMs) are abnormal connections between the arteries and veins in the brain. Most people with brain or spinal AVMs experience few, if any, major symptoms. About 12 percent of p...
4. `medquad_ans_0000168-1` (score=0.0039) - Cerebral cavernous malformations are collections of small blood vessels (capillaries) in the brain that are enlarged and irregular in structure. These capillaries have abnormally thin walls, and they lack other suppor...
5. `medquad_ans_0000641-1` (score=0.0032) - Megalencephaly-capillary malformation syndrome (MCAP) is a disorder characterized by overgrowth of several tissues in the body. Its primary features are a large brain (megalencephaly) and abnormalities of small blood ...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin and appear as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs. These spots may be visible at birth or develop during childhood. In most cases, capillary malformations do not cause health problems and may be the only sign of the disorder. However, some individuals with CM-AVM also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs), which are abnormal connections between arteries, veins, and capillaries that can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure. These complications may appear in infancy or early

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001992-1` (score=0.9994) - What are the signs and symptoms of Dystonia 16? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 16. If the information is available, the table below includes how often the s...
2. `medquad_ans_0001994-1` (score=0.0010) - What are the signs and symptoms of Dystonia 18? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 18. If the information is available, the table below includes how often the s...
3. `medquad_ans_0001995-1` (score=0.0003) - What are the signs and symptoms of Dystonia 19? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 19. If the information is available, the table below includes how often the s...
4. `medquad_ans_0006128-1` (score=0.0001) - What are the signs and symptoms of Torsion dystonia with onset in infancy? The Human Phenotype Ontology provides the following list of signs and symptoms for Torsion dystonia with onset in infancy. If the information ...
5. `medquad_ans_0002001-2` (score=0.0001) - What are the signs and symptoms of Dystonia 8? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 8. If the information is available, the table below includes how often the sym...

**Answer**:

The symptoms of Dystonia 16 include cognitive impairment (5%), abnormal pyramidal signs, bradykinesia, delayed speech and language development, dysarthria, dysphagia, gait disturbance, hyperreflexia, involuntary movements, laryngeal dystonia, limb dystonia, lower limb pain, morphological abnormality of the pyramidal tract, motor delay, parkinsonism, postural tremor, progressive features, retrocollis [medquad_ans_0001992-1].

**Supported citations**: `medquad_ans_0001992-1`
**Unsupported citations**: _none_

### Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Scores**: context_relevance=0.714, groundedness=0.935, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000749-5` (score=0.9973) - These resources address the diagnosis or management of Ohdo syndrome, SBBYS variant: - Gene Review: Gene Review: KAT6B-Related Disorders - Genetic Testing Registry: Young Simpson syndrome These resources from MedlineP...
2. `medquad_ans_0000096-2__dup1` (score=0.9966) - Treatment for SOD is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems associated with SOD are generally not treatable. Vision, physical, and occupational therapie...
3. `medquad_ans_0000899-5` (score=0.9950) - These resources address the diagnosis or management of septo-optic dysplasia: - Genetic Testing Registry: Septo-optic dysplasia sequence - MedlinePlus Encyclopedia: Growth Hormone Deficiency - MedlinePlus Encyclopedia...
4. `medquad_ans_0000150-2__dup1` (score=0.9760) - There is no standard course of treatment for holoprosencephaly. Treatment is symptomatic and supportive.
5. `medquad_ans_0005548-5` (score=0.9652) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not tr...

**Answer**:

The treatments for septo-optic dysplasia are symptomatic and supportive. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not treatable, but vision, physical, and occupational therapies may be required [medquad_ans_0000899-5]. There is no cure for septo-optic dysplasia [medquad_ans_0005548-5].

**Supported citations**: `medquad_ans_0000899-5`, `medquad_ans_0005548-5`
**Unsupported citations**: _none_

### Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Scores**: context_relevance=0.900, groundedness=0.994, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000208-7` (score=0.9931) - The complications of benign prostatic hyperplasia may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney da...
2. `medquad_ans_0000208-11` (score=0.9898) - The complications of benign prostatic hyperplasia treatment depend on the type of treatment. Medications Medications used to treat benign prostatic hyperplasia may have side effects that sometimes can be serious. Men ...
3. `medquad_ans_0000058-19` (score=0.9847) - Yes. Any of the symptoms caused by prostate cancer may also be due to enlargement of the prostate, which is not cancer. If you have any of the symptoms mentioned in question #10, see your doctor or a urologist to find...
4. `medquad_ans_0000201-4__dup2` (score=0.0124) - Hematuria can be caused by menstruation, vigorous exercise, sexual activity, viral illness, trauma, or infection, such as a urinary tract infection (UTI). More serious causes of hematuria include - cancer of the kidne...
5. `medquad_ans_0000219-3__dup3` (score=0.0058) - Erectile dysfunction often has more than one cause. Many diseases can damage nerves, arteries, and muscles. Some can lead to ED, such as - high blood pressure - diabetes, when your blood glucose, also called blood sug...

**Answer**:

The complications of benign prostatic hyperplasia (prostate enlargement) may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney damage - bladder stones. Most men with benign prostatic hyperplasia do not develop these complications. However, kidney damage in particular can be a serious health threat when it occurs [medquad_ans_0000208-7].

Complications related to treatment include side effects from medications such as hives, rash, itching, shortness of breath, rapid or irregular heartbeat, painful erection lasting hours, swelling of the face, eyes, lips, tongue, throat, arms, hands, feet, ankles, or legs, difficulty breathing or swallowing, chest pain, dizziness or fainting when standing up suddenly, sudden decrease or loss of vision, blurred vision, sudden decrease or loss of hearing, chest

**Supported citations**: `medquad_ans_0000208-7`
**Unsupported citations**: _none_

### Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Scores**: context_relevance=0.444, groundedness=0.000, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0000251-2__dup1` (score=0.9965) - Treatment generally consists of physical therapy and drugs to prevent seizures. In cases that are complicated by hydrocephalus, a surgically implanted tube, called a shunt, is often used to divert fluid to another are...
2. `medquad_ans_0000230-2__dup1` (score=0.9963) - There is no specific treatment for PVL. Treatment is symptomatic and supportive. Children with PVL should receive regular medical screenings to determine appropriate interventions.
3. `medquad_ans_0000004-2__dup1` (score=0.9962) - Treatment for LKS usually consists of medications, such as anticonvulsants and corticosteroids, and speech therapy, which should be started early. A controversial treatment option involves a surgical technique called ...
4. `medquad_ans_0000054-2__dup1` (score=0.9958) - Generally treatment for individuals with BSS focuses on the underlying cause of the disorder. Early treatment with high-dose steroids may be beneficial in many cases. Other treatment is symptomatic and supportive.
5. `medquad_ans_0000172-2__dup1` (score=0.9956) - Treatment is symptomatic and supportive, and may include drug therapy.

**Answer**:

The provided context does not contain information about the treatment for Absence of the Septum Pellucidum. Therefore, a specific answer cannot be given based on the available information. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Scores**: context_relevance=1.000, groundedness=0.510, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.9992) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (...
2. `medquad_ans_0002796-2` (score=0.0001) - What are the signs and symptoms of Henoch-Schonlein purpura? The Human Phenotype Ontology provides the following list of signs and symptoms for Henoch-Schonlein purpura. If the information is available, the table belo...
3. `medquad_ans_0005135-1` (score=0.0001) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
4. `medquad_ans_0006001-2` (score=0.0001) - What are the signs and symptoms of Takayasu arteritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Takayasu arteritis. If the information is available, the table below includes h...
5. `medquad_ans_0005360-2` (score=0.0001) - What are the signs and symptoms of Rheumatic Fever? Rheumatic fever is primarily diagnosed in children between the ages of 6 and 16 and can affect many different systems of the body, including the heart, joints, nervo...

**Answer**:

The symptoms of Febrile Ulceronecrotic Mucha-Habermann disease (FUMHD) include red scaly skin lesions (papules) that ulcerate, break down, form open sores, and then develop a red-brown crust (i.e., PLEVA). The lesions can progress to large, destructive ulcers with extensive, painful loss of skin tissue. Skin lesions may become infected, leading to pus and a putrid odor. FUMHD is often associated with high fever (up to 104°F), which may be persistent or intermittent. Other symptoms may include feeling ill, sore throat, congestion, muscle soreness or pain, joint pain, diarrhea, central nervous system symptoms, abdominal pain, enlarged spleen, arthritis, megaloblastic anemia, interstitial pneumonitis, lymphocytic (viral) myocarditis, and sepsis. FUMHD can become life

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000078-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000977-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000460-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000124-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether hereditary leiomyomatosis and renal cell cancer is inherited. While the context describes autosomal recessive inheritance patterns in general, it does not mention hereditary leiomyomatosis and renal cell cancer specifically. Therefore, based on the given information, the inheritance pattern of this condition cannot be determined.

Answer: Insufficient context to determine if hereditary leiomyomatosis and renal cell cancer is inherited. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002445-1` (score=0.9994) - What are the signs and symptoms of Frontometaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontometaphyseal dysplasia. If the information is available, the tabl...
2. `medquad_ans_0005837-1` (score=0.0001) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia x-linked with mental deterioration? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplas...
3. `medquad_ans_0005210-2` (score=0.0001) - What are the signs and symptoms of Pycnodysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pycnodysostosis. If the information is available, the table below includes how oft...
4. `medquad_ans_0005857-1` (score=0.0001) - What are the signs and symptoms of Spondylometaphyseal dysplasia X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia X-linked. If the information ...
5. `medquad_ans_0005835-1` (score=0.0001) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia with multiple dislocations? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia with ...

**Answer**:

The symptoms of Frontometaphyseal dysplasia include abnormality of dental morphology, abnormality of frontal sinus, abnormality of the metaphyses, bowing of the long bones, camptodactyly of finger, craniofacial hyperostosis, hypertelorism, limitation of joint mobility, prominent supraorbital ridges, abnormal form of the vertebral bodies, abnormality of the palate, accelerated skeletal maturation, aplasia/hypoplasia of the thumb, arachnodactyly, conductive hearing impairment, elbow dislocation, scoliosis, sensorineural hearing impairment, skeletal muscle atrophy, synostosis of carpal bones, ulnar deviation of finger, abnormality of the larynx, abnormality of the urethra, complete atrioventricular canal defect, craniosynostosis, tracheal stenosis, ureteral stenosis, ankle

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Scores**: context_relevance=0.818, groundedness=0.857, answer_relevance=0.527

**Top docs**:

1. `medquad_ans_0000172-2__dup1` (score=0.9941) - Treatment is symptomatic and supportive, and may include drug therapy.
2. `medquad_ans_0000033_1-5` (score=0.9870) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - Whether the calcium level in the blood can be controlled...
3. `medquad_ans_0000128-7` (score=0.9832) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding,...
4. `medquad_ans_0000128-9` (score=0.9815) - A doctor may prescribe antibiotics and recommend following a liquid diet. Most people get better with this treatment. Some people may need surgery and other treatments. - Surgery. Serious problems from diverticulitis ...
5. `medquad_ans_0000097-8` (score=0.9577) - A health care provider may treat the symptoms of diverticulosis with a high-fiber diet or fiber supplements, medications, and possibly probiotics. Treatment for diverticular disease varies, depending on whether a pers...

**Answer**:

Treatment for diverticular disease depends on the specific condition (diverticulosis or diverticulitis) and the severity of symptoms. For diverticulosis, treatment may include a high-fiber diet or fiber supplements to prevent complications, along with medications and possibly probiotics to manage symptoms [medquad_ans_0000097-8]. A high-fiber diet can help prevent further development of diverticular disease in individuals already diagnosed with diverticulosis, and a slow increase in dietary fiber may minimize gas and abdominal discomfort [medquad_ans_0000097-8].

For diverticulitis, treatment typically includes antibiotics and a liquid diet, with most people recovering with this approach [medquad_ans_0000128-9]. If complications arise, such as bleeding or recurrent infections, surgery may be necessary. Surgery options include cleaning the abdomen, removing bleeding pouches or fist

**Supported citations**: `medquad_ans_0000097-8`, `medquad_ans_0000128-9`
**Unsupported citations**: _none_

### Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000619-1__dup1` (score=0.9995) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is avail...
2. `medquad_ans_0004414-2` (score=0.0002) - What are the signs and symptoms of Nicolaides-Baraitser syndrome? Nicolaides-Baraitser syndrome (NCBRS) is typically characterized by intellectual disability, seizures, short stature, sparse hair, distinctive facial f...
3. `medquad_ans_0000621-1__dup1` (score=0.0001) - What are the signs and symptoms of Baraitser-Winter syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser-Winter syndrome. If the information is available, the table be...
4. `medquad_ans_0004688-2` (score=0.0001) - What are the signs and symptoms of Pallister-Killian mosaic syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pallister-Killian mosaic syndrome. If the information is availab...
5. `medquad_ans_0003437-1` (score=0.0001) - What are the signs and symptoms of Kuster Majewski Hammerstein syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kuster Majewski Hammerstein syndrome. If the information is a...

**Answer**:

The symptoms of Baraitser Brett Piesowicz syndrome include cerebral calcification, hyperreflexia, hypertonia, microcephaly, seizures, abnormality of movement, cerebral cortical atrophy, cataract, opacification of the corneal stroma, renal insufficiency, anteverted nares, cerebellar hypoplasia, decreased liver function, elevated hepatic transaminases, failure to thrive, hepatomegaly, high palate, increased CSF protein, intellectual disability (profound), jaundice, lissencephaly, long philtrum, low-set ears, muscular hypotonia of the trunk, nystagmus, pachygyria, petechiae, phenotypic variability, polymicrogyria, sloping forehead, spasticity, splenomegaly, thrombocytopenia, ventriculomegaly [medquad*ans*

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Scores**: context_relevance=0.667, groundedness=0.874, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000052-2__dup1` (score=0.9958) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
2. `medquad_ans_0000172-2__dup1` (score=0.9954) - Treatment is symptomatic and supportive, and may include drug therapy.
3. `medquad_ans_0000060-2__dup1` (score=0.9927) - The primary treatment option for a CCM is surgical removal. Radiation therapy has not been shown to be effective. The decision to operate is made based upon the risk of approaching the lesion. For example, symptomatic...
4. `medquad_ans_0000234-2__dup1` (score=0.9922) - Generally, treatment depends on the type of tumor, the size of the tumor, whether the tumor has invaded or pressed on surrounding structures, such as the brain and visual pathways, and the individuals age and overall ...
5. `medquad_ans_0000159-3__dup1` (score=0.9914) - IBM is generally resistant to all therapies and its rate of progression appears to be unaffected by currently available treatments.

**Answer**:

The three most commonly used treatments for meningioma are surgery, radiation, and chemotherapy. Doctors may also prescribe steroids to reduce tumor-related swelling inside the central nervous system (CNS) [medquad_ans_0000052-2__dup1]. Surgical removal is the primary treatment option, especially for symptomatic lesions located in non-eloquent brain areas, while radiation therapy has not been shown to be effective [medquad_ans_0000060-2__dup1]. Treatment is also symptomatic and supportive, and may include drug therapy to manage symptoms such as headache, back pain, and seizures [medquad_ans_0000172-2__dup1]. Medications may also be used to block hormone overproduction, and in some cases, hormone replacement therapy may be necessary after tumor removal [medquad_ans_0000234-2__dup1].

**Supported citations**: `medquad_ans_0000052-2__dup1`, `medquad_ans_0000060-2__dup1`, `medquad_ans_0000172-2__dup1`, `medquad_ans_0000234-2__dup1`
**Unsupported citations**: _none_

### Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Scores**: context_relevance=1.000, groundedness=0.938, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000052-2__dup1` (score=0.9968) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
2. `medquad_ans_0000006_4-7` (score=0.9766) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000006_7-8` (score=0.9738) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a te...
4. `medquad_ans_0000015_1-8` (score=0.9636) - Key Points - There are different types of treatment for children with extracranial germ cell tumors. - Children with extracranial germ cell tumors should have their treatment planned by a team of health care providers...
5. `medquad_ans_0000052-4__dup1` (score=0.9215) - Scientists continue to investigate ways to better understand, diagnose, and treat CNS tumors. Experimental treatment options may include new drugs, gene therapy, surgery , radiation, biologic modulators that enhance t...

**Answer**:

The treatments for Childhood Central Nervous System (CNS) Germ Cell Tumors include surgery, radiation, and chemotherapy. Doctors may also prescribe steroids to reduce tumor-related swelling inside the CNS. Additionally, targeted therapy, which uses drugs or other substances to attack specific cancer cells and causes less harm to normal cells than chemotherapy or radiation, is being studied in clinical trials for recurrent childhood CNS atypical teratoid/rhabdoid tumor. Patients may consider participating in clinical trials, which are part of the cancer research process to determine if new treatments are safe and effective or better than standard treatments. These trials may occur before, during, or after starting treatment and can test new ways to stop cancer from recurring or reduce treatment side effects [medquad_ans_0000052-2\_\_dup1][medquad_ans_0000006_4-7][medquad_ans_0000006

**Supported citations**: `medquad_ans_0000052-2__dup1`, `medquad_ans_0000006_4-7`
**Unsupported citations**: _none_

### Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Scores**: context_relevance=0.889, groundedness=0.984, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0004401-1` (score=0.9127) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (my...
2. `medquad_ans_0000729-1` (score=0.8874) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the ...
3. `medquad_ans_0000065-1__dup1` (score=0.0113) - Farbers disease, also known as Farber's lipogranulomatosis, describes a group of inherited metabolic disorders called lipid storage diseases, in which excess amounts of lipids (oils, fatty acids, and related compounds...
4. `medquad_ans_0000934-2` (score=0.0032) - This condition affects fewer than 1 in 150,000 males and is very rare in females.
5. `medquad_ans_0000091-2` (score=0.0021) - This condition is rare, affecting fewer than 1 per million people.

**Answer**:

Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (myopathy). This condition is caused by mutations in the PNPLA2 gene and is inherited in an autosomal recessive pattern. Other features may include a fatty liver, a weakened and enlarged heart (cardiomyopathy), inflammation of the pancreas (pancreatitis), reduced thyroid activity (hypothyroidism), and type 2 diabetes mellitus. Signs and symptoms vary greatly among affected individuals [medquad_ans_0004401-1][medquad_ans_0000729-1].

**Supported citations**: `medquad_ans_0004401-1`, `medquad_ans_0000729-1`
**Unsupported citations**: _none_

### Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000001-7__dup1` (score=0.9207) - Topics
2. `medquad_ans_0000981-1__dup1` (score=0.4161) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been repo...
3. `medquad_ans_0006546-1` (score=0.0613) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. ...
4. `medquad_ans_0000512-1__dup1` (score=0.0080) - Infectious diseases kill more people worldwide than any other single cause. Infectious diseases are caused by germs. Germs are tiny living things that are found everywhere - in air, soil and water. You can get infecte...
5. `medquad_ans_0000199-19` (score=0.0017) - Vaccines and Preventable Diseases On this Page Vaccine Shortages & Delays Potential New Vaccines Vaccines: The Basics FAQ about Vaccines & Diseases they Prevent VACCINE-PREVENTABLE DISEASES OR, find it by Vaccine Anth...

**Answer**:

Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been reports that the virus has spread through blood transfusions. Most people who get the virus do not get sick. One in five people do get symptoms, which can include a fever, rash, joint pain, and conjunctivitis (pinkeye). Symptoms are usually mild, and start 2 to 7 days after being bitten by an infected mosquito. A blood test can tell whether you have the infection. There are no vaccines or medicines to treat it. Drinking lots of fluids, resting, and taking acetaminophen might help. Zika can cause microcephaly (a serious birth defect of the brain) and other problems in babies whose mothers were infected while pregnant. The Centers for Disease Control and Prevention recommends that pregnant women

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.977, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000033-9` (score=0.9942) - A heart attack occurs when the supply of blood and oxygen to an area of the heart muscle is blocked, usually by a blood clot in a coronary artery. If the blockage is not treated within a few hours, the heart muscle wi...
2. `medquad_ans_0000019-3__dup2` (score=0.9850) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack...
3. `medquad_ans_0000033-3__dup2` (score=0.9790) - Most heart attacks are caused by a blood clot that blocks one of the coronary arteries, the blood vessels that bring blood and oxygen to the heart muscle. When blood cannot reach part of your heart, that area starves ...
4. `medquad_ans_0000169-3` (score=0.8721) - Episodes eventually decrease in frequency and intensity over the course of eight to 12 years.
5. `medquad_ans_0000008-4__dup2` (score=0.7459) - An arrhythmia can occur if the electrical signals that control the heartbeat are delayed or blocked. This can happen if the special nerve cells that produce electrical signals don't work properly. It also can happen i...

**Answer**:

Broken heart syndrome is believed to be caused by extreme emotional or physical stress, which triggers a surge of fight or flight hormones such as adrenaline and noradrenaline. This hormone surge overwhelms and "stuns" the heart muscle, leading to temporary changes in heart muscle cells or coronary blood vessels. Potential causes include a hormone surge, coronary artery spasm, or microvascular dysfunction [medquad_ans_0000019-3__dup2].

**Supported citations**: `medquad_ans_0000019-3__dup2`
**Unsupported citations**: _none_

### Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Scores**: context_relevance=0.800, groundedness=0.981, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000498-1__dup2` (score=0.5367) - Your thyroid is a butterfly-shaped gland in your neck, just above your collarbone. It is one of your endocrine glands, which make hormones. Thyroid hormones control the rate of many activities in your body. These incl...
2. `medquad_ans_0000005-1__dup5` (score=0.4499) - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. An autoimmune disease is a disorder in which the bodys immune system attacks the bodys own cells and...
3. `medquad_ans_0000448-1` (score=0.2328) - Hashimoto thyroiditis is a condition that affects the function of the thyroid, which is a butterfly-shaped gland in the lower neck. The thyroid makes hormones that help regulate a wide variety of critical body functio...
4. `medquad_ans_0000005-8` (score=0.0084) - Iodine is an essential mineral for the thyroid. However, people with Hashimotos disease may be sensitive to harmful side effects from iodine. Taking iodine drops or eating foods containing large amounts of iodinesuch ...
5. `medquad_ans_0002735-1` (score=0.0051) - Hashimoto's encephalitis (HE) is a condition characterized by onset of confusion with altered level of consciousness; seizures; and jerking of muscles (myoclonus). Psychosis, including visual hallucinations and parano...

**Answer**:

Hashimoto thyroiditis is an autoimmune disease in which the body's immune system attacks the thyroid gland, causing inflammation and interfering with its ability to produce thyroid hormones. This condition is also known as chronic lymphocytic thyroiditis or autoimmune thyroiditis. Large numbers of white blood cells called lymphocytes accumulate in the thyroid and produce antibodies that initiate the autoimmune process. Hashimoto thyroiditis often leads to reduced thyroid function, or hypothyroidism [medquad_ans_0000005-1__dup5]. It is the most common cause of hypothyroidism in the United States [medquad_ans_0000005-1__dup5].

**Supported citations**: `medquad_ans_0000005-1__dup5`
**Unsupported citations**: _none_

### Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Scores**: context_relevance=0.800, groundedness=0.533, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001043-1` (score=0.7181) - Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). In affected individuals, harmful amounts of lipids accumulate in the spleen, liver, ...
2. `medquad_ans_0006463-1` (score=0.6708) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include...
3. `medquad_ans_0000002-1__dup1` (score=0.0062) - Acid lipase disease or deficiency occurs when the enzyme needed to break down certain fats that are normally digested by the body is lacking or missing, resulting in the toxic buildup of these fats in the bodys cells ...
4. `medquad_ans_0000874-2` (score=0.0038) - This condition affects an estimated 1 in 8,500 females.
5. `medquad_ans_0000650-2` (score=0.0033) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). It is a type of lysosomal storage disorder caused by mutations in the LIPA gene and inherited in an autosomal recessive manner. The disease results in a buildup of lipids (fats) such as cholesteryl esters and triglycerides in body organs, including the spleen, liver, bone marrow, small intestine, adrenal glands, and lymph nodes. In addition, calcium deposits are seen in the adrenal glands. Infants are typically healthy at birth but soon develop symptoms such as enlarged liver and spleen (hepatosplenomegaly), poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor absorption of nutrients. The condition is severe and life-threatening, with most children not surviving past early childhood. Bone marrow transplantation and enzyme replacement therapy

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002753-3` (score=0.9955) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma a...
2. `medquad_ans_0000087-1__dup2` (score=0.0014) - Tumors are abnormal growths in your body. They are made up of extra cells. Normally, cells grow and divide to form new cells as your body needs them. When cells grow old, they die, and new cells take their place. Some...
3. `medquad_ans_0000621-2` (score=0.0003) - This condition is very rare; fewer than 30 cases have been reported.
4. `medquad_ans_0000886-2` (score=0.0003) - This condition is uncommon; it occurs in an estimated 1 in 100,000 to 125,000 newborns.
5. `medquad_ans_0000094-2` (score=0.0002) - This condition appears to be uncommon, although its prevalence is unknown.

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a possible relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

### Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Scores**: context_relevance=0.750, groundedness=0.870, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000656-3` (score=0.9993) - Mutations in the HCCS gene or a deletion of genetic material that includes the HCCS gene cause microphthalmia with linear skin defects syndrome. The HCCS gene carries instructions for producing an enzyme called holocy...
2. `medquad_ans_0000655-3` (score=0.9980) - Microphthalmia may be caused by changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or...
3. `medquad_ans_0000311-3` (score=0.0157) - Approximately 10 percent to 33 percent of people with juvenile open-angle glaucoma have mutations in the MYOC gene. MYOC gene mutations have also been detected in some people with primary congenital glaucoma. The MYOC...
4. `medquad_ans_0000924-3` (score=0.0030) - Mutations in the SOX2 gene cause SOX2 anophthalmia syndrome. This gene provides instructions for making a protein that plays a critical role in the formation of many different tissues and organs during embryonic devel...
5. `medquad_ans_0000593-3` (score=0.0024) - Mutations in at least two genes on the X chromosome are thought to be responsible for Lenz microphthalmia syndrome. Only one of these genes, BCOR, has been identified. The BCOR gene provides instructions for making a ...

**Answer**:

Genetic changes related to microphthalmia include mutations in genes involved in the early development of the eye, such as the HCCS gene, which causes microphthalmia with linear skin defects syndrome when mutated or deleted [medquad_ans_0000656-3]. Mutations in the HCCS gene result in a nonfunctional holocytochrome c-type synthase enzyme, impairing energy production and cell apoptosis, leading to developmental abnormalities [medquad_ans_0000656-3]. Additionally, microphthalmia may result from chromosomal abnormalities affecting one or more genes, and most genetic changes associated with isolated microphthalmia have been identified in only a small number of affected individuals [medquad_ans_0000655-3]. Environmental factors are also known to contribute to microphthalmia, but the specific genetic changes are primarily linked to mutations in developmental genes [medquad_ans_0

**Supported citations**: `medquad_ans_0000656-3`, `medquad_ans_0000655-3`
**Unsupported citations**: _none_

### Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Scores**: context_relevance=0.833, groundedness=0.698, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000193-2__dup1` (score=0.0001) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 1. If the information is availab...
2. `medquad_ans_0006001-2` (score=0.0001) - What are the signs and symptoms of Takayasu arteritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Takayasu arteritis. If the information is available, the table below includes h...
3. `medquad_ans_0000195-2__dup1` (score=0.0001) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 3. If the information is availab...
4. `medquad_ans_0000872-2__dup1` (score=0.0001) - What are the signs and symptoms of Buerger disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Buerger disease. If the information is available, the table below includes how oft...
5. `medquad_ans_0005261-2` (score=0.0001) - What are the signs and symptoms of Reactive arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Reactive arthritis. If the information is available, the table below includes h...

**Answer**:

The provided context does not contain information about the symptoms of gout. [medquad_ans_0000193-2__dup1] and [medquad_ans_0006001-2] discuss symptoms of Aicardi-Goutieres syndrome type 1 and Takayasu arteritis, respectively, but neither addresses gout. Therefore, no symptoms of gout can be identified from the given context.

**Supported citations**: `medquad_ans_0000193-2__dup1`, `medquad_ans_0006001-2`
**Unsupported citations**: _none_

### Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Scores**: context_relevance=0.333, groundedness=0.000, answer_relevance=0.333

**Top docs**:

1. `medquad_ans_0000102-3__dup1` (score=0.0026) - The progression of the disorder is usually 10 years or longer.
2. `medquad_ans_0000874-2` (score=0.0009) - This condition affects an estimated 1 in 8,500 females.
3. `medquad_ans_0000112-3__dup1` (score=0.0006) - The initial symptoms can be very mild and may be noticeable only after prolonged exertion, stress, or fatigue. Dystonias often progress through various stages. Initially, dystonic movements are intermittent and appear...
4. `medquad_ans_0000904-2` (score=0.0004) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.
5. `medquad_ans_0000971-2` (score=0.0001) - Task-specific focal dystonia affects an estimated 7 to 69 per million people in the general population. Musician's dystonia that is severe enough to impact performance occurs in about 1 percent of musicians.

**Answer**:

The provided context does not contain information about the cause of Dystonia 8. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Scores**: context_relevance=0.111, groundedness=0.780, answer_relevance=0.378

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0001101-2` (score=0.9995) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
3. `medquad_ans_0001092-2` (score=0.9995) - This condition is detected in an estimated 1 in 36,000 newborns worldwide.
4. `medquad_ans_0000790-2` (score=0.9995) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
5. `medquad_ans_0000904-2` (score=0.9995) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.

**Answer**:

The exact incidence of lysinuric protein intolerance is unknown, but it is estimated to affect 1 in 87,000 people [medquad_ans_0000484-2]. Additionally, the condition is detected in an estimated 1 in 36,000 newborns worldwide [medquad_ans_0001092-2], and it is thought to affect approximately 1 in 35,000 to 50,000 newborns [medquad_ans_0000904-2]. The prevalence ranges from 1 in 7,500 to 20,000 people [medquad_ans_0001101-2], and estimates vary widely, with some sources indicating a range of 1 in 25,000 to 300,000

**Supported citations**: `medquad_ans_0000484-2`, `medquad_ans_0001092-2`, `medquad_ans_0000904-2`, `medquad_ans_0001101-2`
**Unsupported citations**: _none_

### Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Scores**: context_relevance=0.750, groundedness=0.700, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0005243-1` (score=0.0002) - What are the signs and symptoms of Radio renal syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Radio renal syndrome. If the information is available, the table below includ...
2. `medquad_ans_0004173-1` (score=0.0001) - What are the signs and symptoms of Multicentric osteolysis nephropathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Multicentric osteolysis nephropathy. If the information is ava...
3. `medquad_ans_0001079-1__dup1` (score=0.0001) - What are the signs and symptoms of Cervical ribs, Sprengel anomaly, anal atresia, and urethral obstruction? The Human Phenotype Ontology provides the following list of signs and symptoms for Cervical ribs, Sprengel an...
4. `medquad_ans_0004494-1` (score=0.0001) - What are the signs and symptoms of Oculo skeletal renal syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Oculo skeletal renal syndrome. If the information is available, the ...
5. `medquad_ans_0004307-2` (score=0.0001) - What are the signs and symptoms of Nail-patella syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Nail-patella syndrome. If the information is available, the table below incl...

**Answer**:

The provided context does not contain information about the symptoms of Renal Artery Stenosis. [medquad_ans_0005243-1] and [medquad_ans_0004173-1] discuss symptoms of Radio renal syndrome and Multicentric osteolysis nephropathy, respectively, but neither addresses Renal Artery Stenosis. Therefore, no symptoms can be identified from the given context.

**Supported citations**: `medquad_ans_0005243-1`, `medquad_ans_0004173-1`
**Unsupported citations**: _none_

### Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Scores**: context_relevance=0.700, groundedness=0.706, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0000718-5` (score=0.9972) - These resources address the diagnosis or management of NOMID: - Genetic Testing Registry: Chronic infantile neurological, cutaneous and articular syndrome These resources from MedlinePlus offer information about the d...
2. `medquad_ans_0000172-2__dup1` (score=0.9963) - Treatment is symptomatic and supportive, and may include drug therapy.
3. `medquad_ans_0000712-5` (score=0.9908) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis -...
4. `medquad_ans_0000618-5` (score=0.9550) - These resources address the diagnosis or management of mal de Meleda: - Foundation for Ichthyosis and Related Skin Types: Palmoplantar Keratodermas - Genetic Testing Registry: Acroerythrokeratoderma These resources fr...
5. `medquad_ans_0000071-2__dup1` (score=0.0408) - There is no standard course of treatment for Sotos syndrome. Treatment is symptomatic.

**Answer**:

The provided context does not specify treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis. While it mentions that resources for these conditions include information on drug therapy, genetic counseling, and other management strategies, no specific treatments are detailed [medquad_ans_0000712-5]. Therefore, based on the available information, the exact treatments are not explicitly stated.

**Supported citations**: `medquad_ans_0000712-5`
**Unsupported citations**: _none_

### Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.9994) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0004851-1` (score=0.0001) - What are the signs and symptoms of Pheochromocytoma-islet cell tumor syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pheochromocytoma-islet cell tumor syndrome. If the info...
3. `medquad_ans_0004718-1` (score=0.0001) - What are the signs and symptoms of Paragangliomas 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Paragangliomas 1. If the information is available, the table below includes how o...
4. `medquad_ans_0005142-2` (score=0.0001) - What are the signs and symptoms of Pseudohypoparathyroidism type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudohypoparathyroidism type 1C. If the information is available...
5. `medquad_ans_0005138-2` (score=0.0001) - What are the signs and symptoms of Pseudohypoaldosteronism type 2? The most common symptom of pseudohypoaldosteronism type 2 is high blood pressure in adolescents or young adults. In its most severe form, it is associ...

**Answer**:

The symptoms of Familial hyperaldosteronism type III include hypercalciuria (5%), metabolic acidosis (5%), polydipsia (5%), polyuria (5%), adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

**Supported citations**: `medquad_ans_0002259-1`
**Unsupported citations**: _none_

### Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000642-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000111-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000460-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000851-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Spinocerebellar ataxia 11 or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Spinocerebellar ataxia 11 is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Scores**: context_relevance=1.000, groundedness=0.847, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0003040-1` (score=0.9993) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload....
2. `medquad_ans_0000259-2__dup1` (score=0.0001) - What are the signs and symptoms of Alpha-thalassemia x-linked intellectual disability syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Alpha-thalassemia x-linked intellectua...
3. `medquad_ans_0005045-2` (score=0.0001) - What are the signs and symptoms of Primary Familial Brain Calcification? The Human Phenotype Ontology provides the following list of signs and symptoms for Primary Familial Brain Calcification. If the information is a...
4. `medquad_ans_0000535-2` (score=0.0001) - What are the signs and symptoms of Autoimmune hemolytic anemia? The Human Phenotype Ontology provides the following list of signs and symptoms for Autoimmune hemolytic anemia. If the information is available, the tabl...
5. `medquad_ans_0001504-1` (score=0.0001) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital dyserythropoietic anemia type 3. If the info...

**Answer**:

The signs and symptoms of Hypochromic microcytic anemia with iron overload include abnormality of metabolism/homeostasis, abnormality of the liver, anemia, and autosomal recessive inheritance. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which compiles information from sources such as Orphanet, with frequencies typically estimated as percentages or fractions of patients exhibiting the symptom [medquad_ans_0003040-1]. Specific symptom frequencies are not detailed in the provided context.

**Supported citations**: `medquad_ans_0003040-1`
**Unsupported citations**: _none_

### Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000441-4` (score=0.9988) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has a parent and other family ...
3. `medquad_ans_0000403-4` (score=0.9988) - This condition has an autosomal dominant inheritance pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. All reported cases have resulted from new mutations in the gene ...
4. `medquad_ans_0000429-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000045-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention benign chronic pemphigus or its inheritance pattern. Therefore, it is not possible to determine based on the given information whether benign chronic pemphigus is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000926-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000949-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000721-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000231-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000856-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Camurati-Engelmann disease or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Camurati-Engelmann disease is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Scores**: context_relevance=0.900, groundedness=0.978, answer_relevance=0.740

**Top docs**:

1. `medquad_ans_0000088-3` (score=0.9993) - Mutations in the FAS gene cause ALPS in approximately 75 percent of affected individuals. The FAS gene provides instructions for making a protein involved in cell signaling that results in the self-destruction of cell...
2. `medquad_ans_0000805-3` (score=0.9993) - Mutations in the TREM2 gene or the TYROBP gene (also called DAP12) can cause PLOSL. The proteins produced from these two genes work together to activate certain kinds of cells. These proteins appear to be particularly...
3. `medquad_ans_0000483-3` (score=0.9990) - Histiocytosis-lymphadenopathy plus syndrome is caused by mutations in the SLC29A3 gene, which provides instructions for making a protein called equilibrative nucleoside transporter 3 (ENT3). ENT3 belongs to a family o...
4. `medquad_ans_0000013-9__dup1` (score=0.9385) - A heritable disease is caused by a mutation, or harmful change, in a gene inherited from a parent. Genes are small structures in your body's cells that determine how you look and tell your body how to work. Examples o...
5. `medquad_ans_0000398-1__dup1` (score=0.0217) - Genes are the building blocks of heredity. They are passed from parent to child. They hold DNA, the instructions for making proteins. Proteins do most of the work in cells. They move molecules from one place to anothe...

**Answer**:

The genetic change related to histiocytosis-lymphadenopathy plus syndrome is mutations in the SLC29A3 gene, which provides instructions for making a protein called equilibrative nucleoside transporter 3 (ENT3) [medquad_ans_0000483-3].

**Supported citations**: `medquad_ans_0000483-3`
**Unsupported citations**: _none_

### Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000647-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000423-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000070-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000931-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Noonan syndrome or its mode of inheritance. Therefore, it is not possible to determine from the given information whether Noonan syndrome is inherited. [no_source]

**Supported citations**: _none_
**Unsupported citations**: `no_source`

### Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Scores**: context_relevance=0.500, groundedness=0.650, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001093-5` (score=0.9952) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Aci...
2. `medquad_ans_0001092-5` (score=0.9949) - These resources address the diagnosis or management of 3-MCC deficiency: - Baby's First Test - Genetic Testing Registry: 3 Methylcrotonyl-CoA carboxylase 1 deficiency - Genetic Testing Registry: 3-methylcrotonyl CoA c...
3. `medquad_ans_0001089-5` (score=0.2636) - These resources address the diagnosis or management of HMG-CoA lyase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of hydroxymethylglutaryl-CoA lyase These resources from MedlinePlus offer inf...
4. `medquad_ans_0000120-5` (score=0.0064) - These resources address the diagnosis or management of beta-ketothiolase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of acetyl-CoA acetyltransferase These resources from MedlinePlus offer in...
5. `medquad_ans_0000444-5` (score=0.0056) - These resources address the diagnosis or management of gyrate atrophy: - Baby's First Test - Genetic Testing Registry: Ornithine aminotransferase deficiency These resources from MedlinePlus offer information about the...

**Answer**:

The provided context does not specify the treatments for 3-methylglutaconyl-CoA hydratase deficiency. While it mentions resources related to diagnosis and management, such as "Baby's First Test" and references to the Genetic Testing Registry, no specific treatments are listed or described [medquad_ans_0001093-5]. Therefore, based on the given information, the treatments cannot be determined.

**Supported citations**: `medquad_ans_0001093-5`
**Unsupported citations**: _none_

### Example 61 (`0003000-1`)

**Question**: What is (are) Hyperkalemic periodic paralysis ?

**Scores**: context_relevance=0.833, groundedness=0.994, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003000-1` (score=0.9120) - Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move...
2. `medquad_ans_0000171-3__dup1` (score=0.8521) - KTS is often a progressive disorder, and complications may be life-threatening. However, many individuals can live well while managing their symptoms.
3. `medquad_ans_0000505-1` (score=0.7986) - Hypokalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness typically beginning in childhood or adolescence. Most often, these episodes involve a temporary inability to move muscles i...
4. `medquad_ans_0000169-3` (score=0.7807) - Episodes eventually decrease in frequency and intensity over the course of eight to 12 years.
5. `medquad_ans_0000494-1` (score=0.7255) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to...

**Answer**:

Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms and legs. Episodes tend to increase in frequency until about age 25, after which they may occur less frequently. Factors that can trigger attacks include rest after exercise, potassium-rich foods, stress, fatigue, and long periods without food. Muscle strength improves between attacks, although many affected people continue to experience mild stiffness, particularly in muscles of the face and hands. This condition is caused by mutations in the SCN4A gene and is inherited in an autosomal dominant fashion [medquad_ans_0003000-1].

**Supported citations**: `medquad_ans_0003000-1`
**Unsupported citations**: _none_

### Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Scores**: context_relevance=0.600, groundedness=0.064, answer_relevance=0.760

**Top docs**:

1. `medquad_ans_0000218-5` (score=0.9971) - These resources address the diagnosis or management of CCFDN: - Gene Review: Gene Review: Congenital Cataracts, Facial Dysmorphism, and Neuropathy - Genetic Testing Registry: Congenital Cataracts, Facial Dysmorphism, ...
2. `medquad_ans_0000130-5` (score=0.9968) - These resources address the diagnosis or management of BPES: - Gene Review: Gene Review: Blepharophimosis, Ptosis, and Epicanthus Inversus - Genetic Testing Registry: Blepharophimosis, ptosis, and epicanthus inversus ...
3. `medquad_ans_0000054-2__dup1` (score=0.9968) - Generally treatment for individuals with BSS focuses on the underlying cause of the disorder. Early treatment with high-dose steroids may be beneficial in many cases. Other treatment is symptomatic and supportive.
4. `medquad_ans_0000043-2__dup2` (score=0.9966) - In most cases of BEB the treatment of choice is botulinum toxin injections which relax the muscles and stop the spasms. Other treatment options include medications (drug therapy) or surgery--either local surgery of th...
5. `medquad_ans_0000199-2__dup1` (score=0.9963) - No cures or specific therapies for ML currently exists. Therapies are generally geared toward treating symptoms and providing supportive care to the child. For individuals with corneal clouding, surgery to remove the ...

**Answer**:

The provided context does not contain information about the treatments for congenital fibrosis of the extraocular muscles. Therefore, a specific answer cannot be generated from the given resources. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

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

**Scores**: context_relevance=0.111, groundedness=0.000, answer_relevance=0.289

**Top docs**:

1. `medquad_ans_0001101-2` (score=0.9996) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
2. `medquad_ans_0000484-2` (score=0.9996) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
3. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
4. `medquad_ans_0000904-2` (score=0.9995) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.
5. `medquad_ans_0000874-2` (score=0.9995) - This condition affects an estimated 1 in 8,500 females.

**Answer**:

The provided context does not contain any information about glutamate formiminotransferase deficiency. Therefore, the prevalence or incidence of this specific condition cannot be determined from the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Scores**: context_relevance=0.125, groundedness=0.740, answer_relevance=0.425

**Top docs**:

1. `medquad_ans_0001101-2` (score=0.9995) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
2. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
3. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0000790-2` (score=0.9995) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
5. `medquad_ans_0000770-2` (score=0.9995) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.

**Answer**:

The prevalence of Noonan syndrome is estimated to range from 1 in 7,500 to 20,000 people [medquad_ans_0001101-2]. Other estimates suggest it affects 1 in 87,000 people [medquad_ans_0000484-2], 1 in 40,000 to 50,000 people [medquad_ans_0000195-2], or 1 in 25,000 to 300,000 individuals [medquad_ans_0000790-2]. Additionally, it is estimated to affect 1 to 3 per million people worldwide [medquad_ans_0000770-2]. Due to the variation in estimates, the exact number of people affected is not consistently

**Supported citations**: `medquad_ans_0001101-2`, `medquad_ans_0000484-2`, `medquad_ans_0000195-2`, `medquad_ans_0000790-2`, `medquad_ans_0000770-2`
**Unsupported citations**: _none_

### Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Scores**: context_relevance=0.714, groundedness=0.989, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000053-5__dup2` (score=0.9966) - Body lice infestation is diagnosed by finding eggs and crawling lice in the seams of clothing. Sometimes a body louse can be seen crawling or feeding on the skin. Although body lice and nits can be large enough to be ...
2. `medquad_ans_0000339-5__dup1` (score=0.9943) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on t...
3. `medquad_ans_0000265-8` (score=0.9512) - In people who have been bitten by the flies that carry Loa loa in areas where Loa loa is known to exist, the diagnosis can be made in the following ways: - Identification of the adult worm by a microbiologist or patho...
4. `medquad_ans_0000214-5__dup1` (score=0.0870) - Misdiagnosis of head lice infestation is common. The diagnosis of head lice infestation is best made by finding a live nymph or adult louse on the scalp or hair of a person. Because adult and nymph lice are very small...
5. `medquad_ans_0000440-5__dup1` (score=0.0291) - Cutaneous larva migrans (CLM) is a clinical diagnosis based on the presence of the characteristic signs and symptoms, and exposure history to zoonotic hookworm. For example, the diagnosis can be made based on finding ...

**Answer**:

Pubic "crab" lice infestation is diagnosed by finding a "crab" louse or eggs on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp) [medquad_ans_0000339-5__dup1]. Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

### Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.873, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000749-5` (score=0.9969) - These resources address the diagnosis or management of Ohdo syndrome, SBBYS variant: - Gene Review: Gene Review: KAT6B-Related Disorders - Genetic Testing Registry: Young Simpson syndrome These resources from MedlineP...
2. `medquad_ans_0000172-2__dup1` (score=0.9966) - Treatment is symptomatic and supportive, and may include drug therapy.
3. `medquad_ans_0005571-3` (score=0.9688) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
4. `medquad_ans_0000860-2__dup1` (score=0.9474) - How might Brown-Sequard syndrome be treated?
5. `medquad_ans_0000124-2__dup1` (score=0.1886) - There is no cure for Fahr's Syndrome, nor is there a standard course of treatment. Treatment addresses symptoms on an individual basis.

**Answer**:

Treatment for Shapiro syndrome has been limited due to the small number of diagnosed cases and the periodic nature of the disease. Attempted treatments include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, or sympathectomy, with varying responses observed. However, treatment options should be discussed with a healthcare provider, as only they can determine the appropriate course of treatment for an individual patient [medquad_ans_0005571-3].

**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

### Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.970, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002359-1` (score=0.6838) - FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. People wi...
2. `medquad_ans_0000373-1` (score=0.6256) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects ...
3. `medquad_ans_0002458-1` (score=0.0445) - Fryns syndrome is a condition that affects the development of many parts of the body. Signs and symptoms vary widely among affected individuals. Many affected individuals have a defect in the diaphragm muscle such as ...
4. `medquad_ans_0000392-1` (score=0.0020) - Fryns syndrome is a condition that affects the development of many parts of the body. The features of this disorder vary widely among affected individuals and overlap with the signs and symptoms of several other disor...
5. `medquad_ans_0000087-3__dup1` (score=0.0014) - The prognosis is dependent upon the specific syndrome, however, some of the syndromes are fatal if left untreated.

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. It is characterized by intellectual disability ranging from mild to severe, hypotonia, constipation and/or anal anomalies, a distinctive facial appearance, broad thumbs and great toes, relative macrocephaly, and abnormalities of the corpus callosum. Medical problems such as heart defects, seizures, undescended testicles, and inguinal hernia have also been reported. The condition is inherited in an X-linked recessive pattern, and mutations in genes such as MED12 (FGS1), FLNA (FGS2), CASK (FGS4), UPF3B (FGS6), and BRWD3 (FGS7) are associated with the disorder [medquad_ans_0002359-1]. Affected individuals often

**Supported citations**: `medquad_ans_0002359-1`
**Unsupported citations**: _none_

### Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006242-1` (score=0.9994) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0000886-2__dup1` (score=0.0002) - What are the signs and symptoms of C syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for C syndrome. If the information is available, the table below includes how often the sym...
3. `medquad_ans_0006175-1` (score=0.0001) - What are the signs and symptoms of Tricho-dento-osseous syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Tricho-dento-osseous syndrome 1. If the information is available, ...
4. `medquad_ans_0006572-2` (score=0.0001) - What are the signs and symptoms of 22q13.3 deletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 22q13.3 deletion syndrome. If the information is available, the table be...
5. `medquad_ans_0006183-2` (score=0.0001) - What are the signs and symptoms of Trichorhinophalangeal syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Trichorhinophalangeal syndrome type 1. If the information is...

**Answer**:

The signs and symptoms of Tukel syndrome include carpal bone aplasia, carpal synostosis, compensatory chin elevation, congenital fibrosis of extraocular muscles, nonprogressive restrictive external ophthalmoplegia, postaxial oligodactyly, ptosis, restrictive external ophthalmoplegia, syndactyly [medquad_ans_0006242-1].

**Supported citations**: `medquad_ans_0006242-1`
**Unsupported citations**: _none_

### Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Scores**: context_relevance=0.167, groundedness=0.817, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000179-5` (score=0.9971) - These resources address the diagnosis or management of cholesteryl ester storage disease: - Genetic Testing Registry: Lysosomal acid lipase deficiency - MedlinePlus Encyclopedia: Atherosclerosis - MedlinePlus Encyclop...
2. `medquad_ans_0000217-2` (score=0.9970) - Treatment involves eliminating or reducing exposure to the toxic substance, followed by symptomatic and supportive therapy.
3. `medquad_ans_0000172-2__dup1` (score=0.9966) - Treatment is symptomatic and supportive, and may include drug therapy.
4. `medquad_ans_0000674-5` (score=0.9859) - These resources address the diagnosis or management of mucolipidosis II alpha/beta: - Gene Review: Gene Review: Mucolipidosis II - Genetic Testing Registry: I cell disease - MedlinePlus Encyclopedia: Clubfoot - Medlin...
5. `medquad_ans_0001043-5` (score=0.9789) - These resources address the diagnosis or management of Wolman disease: - Genetic Testing Registry: Lysosomal acid lipase deficiency These resources from MedlinePlus offer information about the diagnosis and management...

**Answer**:

The treatments for Mucolipidoses are symptomatic and supportive, and may include drug therapy [medquad_ans_0000172-2__dup1]. Additionally, management may involve diagnostic tests, surgery and rehabilitation, genetic counseling, and palliative care as offered by MedlinePlus resources [medquad_ans_0000674-5].

**Supported citations**: `medquad_ans_0000172-2__dup1`, `medquad_ans_0000674-5`
**Unsupported citations**: _none_

### Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0005084-2` (score=0.9994) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis...
2. `medquad_ans_0001503-2` (score=0.0001) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to abs...
3. `medquad_ans_0003240-2` (score=0.0001) - What are the signs and symptoms of Intrahepatic cholestasis of pregnancy? The Human Phenotype Ontology provides the following list of signs and symptoms for Intrahepatic cholestasis of pregnancy. If the information is...
4. `medquad_ans_0000998-1__dup1` (score=0.0001) - What are the signs and symptoms of Cataract, autosomal recessive congenital 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Cataract, autosomal recessive congenital 2. If the info...
5. `medquad_ans_0003292-2` (score=0.0001) - What are the signs and symptoms of Jejunal atresia? Signs and symptoms of jejunal atresia vary but may include: Feeding difficulties Failure to thrive Vomiting bile (a bitter-tasting yellowish-green fluid) Abdominal s...

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 include cirrhosis, conjugated hyperbilirubinemia, death in childhood, diarrhea, elevated alkaline phosphatase, failure to thrive, fat malabsorption, hepatocellular carcinoma, hepatomegaly, infantile onset, intermittent jaundice, intrahepatic cholestasis, pruritus, short stature, and splenomegaly [medquad_ans_0005084-2].

**Supported citations**: `medquad_ans_0005084-2`
**Unsupported citations**: _none_

### Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Scores**: context_relevance=0.200, groundedness=0.000, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
3. `medquad_ans_0001101-2` (score=0.9995) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
4. `medquad_ans_0000790-2` (score=0.9994) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
5. `medquad_ans_0000770-2` (score=0.9994) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.

**Answer**:

The provided context does not contain information about the number of people affected by Urinary Incontinence in Men. Therefore, the question cannot be answered with the given information. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0005415-1` (score=0.3207) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
2. `medquad_ans_0000882-1` (score=0.2402) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
3. `medquad_ans_0000392-1` (score=0.0009) - Fryns syndrome is a condition that affects the development of many parts of the body. The features of this disorder vary widely among affected individuals and overlap with the signs and symptoms of several other disor...
4. `medquad_ans_0005892-2` (score=0.0002) - What are the signs and symptoms of Stickler syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Stickler syndrome type 1. If the information is available, the table belo...
5. `medquad_ans_0005864-1` (score=0.0002) - What are the signs and symptoms of Spranger Schinzel Myers syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Spranger Schinzel Myers syndrome. If the information is available...

**Answer**:

Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Two major types have been identified: autosomal recessive and autosomal dominant Robinow syndrome. These types are distinguished by the severity of their signs and symptoms and by their pattern of inheritance. Autosomal recessive Robinow syndrome is characterized by skeletal abnormalities such as shortening of the long bones in the arms and legs, abnormally short fingers and toes (brachydactyly), wedge-shaped spinal bones (hemivertebrae) leading to kyphoscoliosis, fused or missing ribs, and short stature. Affected individuals also have distinctive facial features, including a broad forehead, prominent and widely spaced eyes, a short nose with an upturned tip, a wide nasal bridge, and a broad, triangle-shaped mouth, which are sometimes described as "fetal facies." Other features may include underdeveloped genitalia, dental

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 74 (`0000551-1`)

**Question**: What is (are) juvenile primary lateral sclerosis ?

**Scores**: context_relevance=0.857, groundedness=0.979, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000551-1` (score=0.9132) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. The features of this disorder are caused by damage to moto...
2. `medquad_ans_0003331-1` (score=0.8675) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
3. `medquad_ans_0000238-1__dup1` (score=0.1857) - Primary lateral sclerosis (PLS) is a rare neuromuscular disease with slowly progressive weakness in voluntary muscle movement. PLS belongs to a group of disorders known as motor neuron diseases. PLS affects the upper ...
4. `medquad_ans_0000102-3__dup1` (score=0.1733) - The progression of the disorder is usually 10 years or longer.
5. `medquad_ans_0003319-1` (score=0.0260) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. U...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. It is caused by damage to motor neurons, which are specialized nerve cells in the brain and spinal cord that control muscle movement. Symptoms begin in early childhood and progress slowly over many years. Early symptoms include clumsiness, muscle weakness, and spasticity in the legs, and difficulty with balance. As the disorder progresses, spasticity spreads to the arms and hands, and individuals may develop slurred speech, drooling, difficulty swallowing, and an inability to walk. Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene and is inherited in an autosomal recessive pattern [medquad_ans_0003331-1]. The progression of the disorder typically occurs over a period of 15 to 20 years [medquad_ans_000

**Supported citations**: `medquad_ans_0003331-1`
**Unsupported citations**: _none_

### Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Scores**: context_relevance=0.750, groundedness=0.977, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000040-11` (score=0.9829) - There are four common types of leukemia. They are chronic lymphocytic leukemia, chronic myeloid leukemia, acute myeloid leukemia, and acute lymphocytic leukemia. Chronic lymphocytic leukemia, chronic myeloid leukemia,...
2. `medquad_ans_0000040-9` (score=0.9663) - Leukemia is a cancer of the blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white ...
3. `medquad_ans_0000200-1__dup3` (score=0.8027) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells c...
4. `medquad_ans_0000554-1__dup2` (score=0.7928) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells c...
5. `medquad_ans_0000010-1__dup4` (score=0.7665) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. Thes...

**Answer**:

Leukemia is a cancer of the white blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white blood cells builds up in the blood, they crowd out healthy blood cells, making it difficult for the blood to carry out its normal functions [medquad_ans_0000040-9]. There are four common types of leukemia: chronic lymphocytic leukemia, chronic myeloid leukemia, acute myeloid leukemia, and acute lymphocytic leukemia [medquad_ans_0000040-11].

**Supported citations**: `medquad_ans_0000040-9`, `medquad_ans_0000040-11`
**Unsupported citations**: _none_

### Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Scores**: context_relevance=0.750, groundedness=0.967, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0002485-1` (score=0.7954) - Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. Gangliocytomas may occur in all age groups but most often occur in people between the ages of 10 and 30. The most common si...
2. `medquad_ans_0000127-1__dup3` (score=0.0233) - A brain tumor is a growth of abnormal cells in the tissues of the brain. Brain tumors can be benign, with no cancer cells, or malignant, with cancer cells that grow quickly. Some are primary brain tumors, which start ...
3. `medquad_ans_0000006_3-5` (score=0.0074) - Tests that examine the brain and spinal cord are used to detect (find) childhood astrocytomas. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general signs o...
4. `medquad_ans_0000144-1__dup2` (score=0.0050) - Cancer begins in the cells, which are the building blocks of your body. Normally, new cells form as you need them, replacing old cells that die. Sometimes, this process goes wrong. New cells form when you don't need t...
5. `medquad_ans_0000052-3__dup1` (score=0.0045) - Symptoms of brain and spinal cord tumors generally develop slowly and worsen over time unless they are treated. The tumor may be classified as benign or malignant and given a numbered score that reflects its rate of m...

**Answer**:

Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. It may occur in all age groups but most commonly occurs in people between the ages of 10 and 30. The most common site is the temporal lobe of the brain, although it can arise anywhere in the CNS, including the cerebellum, brainstem, floor of the third ventricle, and spinal cord. Gangliocytomas are among the most frequent tumors associated with epilepsy and are generally slow-growing, usually not becoming malignant. Treatment involves surgical removal of the tumor [medquad_ans_0002485-1].

**Supported citations**: `medquad_ans_0002485-1`
**Unsupported citations**: _none_

### Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.989, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000130-1__dup1` (score=0.7352) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of ...
2. `medquad_ans_0000658-1` (score=0.0291) - Miller syndrome is a rare condition that mainly affects the development of the face and limbs. The severity of this disorder varies among affected individuals. Children with Miller syndrome are born with underdevelope...
3. `medquad_ans_0000229-3__dup1` (score=0.0052) - In acute neuropathies, such as Guillain-Barr syndrome, symptoms appear suddenly, progress rapidly, and resolve slowly as damaged nerves heal. In chronic forms, symptoms begin subtly and progress slowly. Some people ma...
4. `medquad_ans_0000267-4__dup1` (score=0.0018) - The National Institute of Neurological Disorders and Stroke (NINDS) and other institutes at the National Institutes of Health support research in TOS through grants to major medical research institutions across the co...
5. `medquad_ans_0000222-3__dup1` (score=0.0018) - The prognosis for those with paresthesia depends on the severity of the sensations and the associated disorders.

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of tendon reflexes. Symptoms may be preceded by a viral illness and include generalized muscle weakness and respiratory failure. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder [medquad_ans_0000130-1__dup1].

**Supported citations**: `medquad_ans_0000130-1__dup1`
**Unsupported citations**: _none_

### Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006570-3` (score=0.9947) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location desig...
2. `medquad_ans_0006579-3` (score=0.0223) - What causes 2q37 deletion syndrome? 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at ...
3. `medquad_ans_0001085-3` (score=0.0010) - People with 22q11.2 duplication have an extra copy of some genetic material at position q11.2 on chromosome 22. In most cases, this extra genetic material consists of a sequence of about 3 million DNA building blocks ...
4. `medquad_ans_0001087-3` (score=0.0003) - 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at a location designated 2q37. The size...
5. `medquad_ans_0006554-3` (score=0.0002) - What causes a 15q11.2 microdeletion? A 15q11.2 microdeletion may occur randomly for the first time in an affected person (a de novo mutation), or it may be inherited from a parent with the microdeletion. A blood test ...

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
4. `medquad_ans_0000062-1__dup5` (score=0.0012) - Shingles is a painful rash that develops on one side of the face or body. The rash forms blisters that typically scab over in 7 to 10 days and clear up within 2 to 4 weeks. Most commonly, the rash occurs in a single s...
5. `medquad_ans_0000149-4__dup1` (score=0.0006) - The NINDS supports research on shingles and shingles-related conditions. Current studies focus on the relationship between the persistence of neurotropic viruses and development of neurological diseases including herp...

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
3. `medquad_ans_0002359-2` (score=0.0068) - What are the signs and symptoms of FG syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for FG syndrome. If the information is available, the table below includes how often the s...
4. `medquad_ans_0004688-2` (score=0.0001) - What are the signs and symptoms of Pallister-Killian mosaic syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pallister-Killian mosaic syndrome. If the information is availab...
5. `medquad_ans_0003260-1` (score=0.0001) - What are the signs and symptoms of Irons Bhan syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Irons Bhan syndrome. If the information is available, the table below includes...

**Answer**:

The symptoms of Fryns Hofkens Fabry syndrome include ulnar deviation of the finger (90%), distal ulnar hypoplasia, dysplastic radii, hypoplasia of the radius, mesomelic arm shortening, radial bowing, and ulnar deviation of the hand [medquad_ans_0002456-1].

**Supported citations**: `medquad_ans_0002456-1`
**Unsupported citations**: _none_

### Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Scores**: context_relevance=0.900, groundedness=0.024, answer_relevance=0.660

**Top docs**:

1. `medquad_ans_0000169-4` (score=0.9973) - NINDS supports a broad range of clinical and basic research on diseases causing sleep disorders in an effort to clarify the mechanisms of these conditions and to develop better treatments for them.
2. `medquad_ans_0000086-4__dup1` (score=0.9969) - The NINDS supports and conducts research studies on degenerative disorders such as corticobasal degeneration. The goals of these studies are to increase scientific understanding of these disorders and to find ways to ...
3. `medquad_ans_0000090-4__dup1` (score=0.9968) - Much of the on-going research on RMDs is aimed at prevention and rehabilitation. The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) funds research on RMDs.
4. `medquad_ans_0000207-1__dup2` (score=0.9963) - Summary : Clinical trials are research studies that test how well new medical approaches work in people. Each study answers scientific questions and tries to find better ways to prevent, screen for, diagnose, or treat...
5. `medquad_ans_0000927-1__dup2` (score=0.9963) - Summary : It seems to happen almost every day - you hear about the results of a new medical research study. Sometimes the results of one study seem to disagree with the results of another study. It's important to be c...

**Answer**:

The provided context does not mention any research or clinical trials specifically related to osteoporosis. Therefore, based on the given information, it is not possible to answer the question. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Scores**: context_relevance=0.857, groundedness=0.891, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000414-1__dup2` (score=0.9978) - Frequently Asked Questions (FAQs) Fact Sheets
2. `medquad_ans_0000269-1__dup4` (score=0.9973) - Frequently Asked Questions (FAQs) Vector Information
3. `medquad_ans_0000338-1__dup1` (score=0.9973) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the s...
4. `medquad_ans_0000001-7__dup1` (score=0.9967) - Topics
5. `medquad_ans_0000430-1__dup1` (score=0.9966) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. ...

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. It often includes red flags such as "miracle cure," "quick fix," "ancient remedy," "secret ingredient," or "scientific breakthrough." These scams can be dangerous, waste money, and may prevent individuals from receiving necessary treatment. The Food and Drug Administration (FDA) advises that if something sounds too good to be true, it probably is a scam. Before taking an unproven treatment, it is recommended to consult a doctor or healthcare professional, especially when considering prescription drugs [medquad_ans_0000430-1__dup1].

**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

### Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Scores**: context_relevance=0.750, groundedness=0.735, answer_relevance=0.617

**Top docs**:

1. `medquad_ans_0000042-3__dup3` (score=0.9969) - There are many signs that can signal vision loss. For example, even with your regular glasses, do you have difficulty - recognizing faces of friends and relatives? - doing things that require you to see well up close,...
2. `medquad_ans_0000016-3__dup2` (score=0.9952) - Diabetic retinopathy often has no early warning signs. Don't wait for symptoms. Be sure to have a comprehensive dilated eye exam at least once a year to detect the disease before it causes damage to your vision. Visio...
3. `medquad_ans_0000055-3__dup2` (score=0.9946) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision -...
4. `medquad_ans_0000015-15` (score=0.9867) - Many people with diabetes experience one or more symptoms, including extreme thirst or hunger, a frequent need to urinate and/or fatigue. Some lose weight without trying. Additional signs include sores that heal slowl...
5. `medquad_ans_0000043-2__dup4` (score=0.9731) - Many people have no visible signs or symptoms of diabetes. Symptoms can also be so mild that you might not notice them. More than 5 million people in the United States have type 2 diabetes and do not know it. - increa...

**Answer**:

The provided context does not contain information about symptoms of preventing diabetes problems or specifically about keeping eyes healthy in the context of diabetes prevention. While the context discusses vision changes and diabetic retinopathy, including early warning signs and the importance of regular eye exams, it does not outline symptoms related to preventing diabetes problems or eye health maintenance as a preventive measure. Therefore, based on the given information, the answer cannot be determined.

[medquad_ans_0000042-3__dup3] [medquad_ans_0000016-3__dup2]

**Supported citations**: `medquad_ans_0000042-3__dup3`, `medquad_ans_0000016-3__dup2`
**Unsupported citations**: _none_

### Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000424-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000477-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000572-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000264-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Aarskog-Scott syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Aarskog-Scott syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Scores**: context_relevance=0.800, groundedness=0.956, answer_relevance=0.560

**Top docs**:

1. `medquad_ans_0001062-3` (score=0.9992) - Mutations in the ALAS2 gene cause X-linked sideroblastic anemia. The ALAS2 gene provides instructions for making an enzyme called erythroid ALA-synthase, which plays a critical role in the production of heme (a compon...
2. `medquad_ans_0001063-3` (score=0.9982) - Mutations in the ABCB7 gene cause X-linked sideroblastic anemia and ataxia. The ABCB7 gene provides instructions for making a protein that is critical for heme production. Heme is a component of the hemoglobin protein...
3. `medquad_ans_0000823-4` (score=0.9903) - This condition is generally not inherited but arises from gene mutations that occur in early blood-forming cells after conception. These alterations are called somatic mutations.
4. `medquad_ans_0000013-9__dup1` (score=0.0549) - A heritable disease is caused by a mutation, or harmful change, in a gene inherited from a parent. Genes are small structures in your body's cells that determine how you look and tell your body how to work. Examples o...
5. `medquad_ans_0000398-1__dup1` (score=0.0183) - Genes are the building blocks of heredity. They are passed from parent to child. They hold DNA, the instructions for making proteins. Proteins do most of the work in cells. They move molecules from one place to anothe...

**Answer**:

Mutations in the ALAS2 gene cause X-linked sideroblastic anemia. The ALAS2 gene provides instructions for making an enzyme called erythroid ALA-synthase, which plays a critical role in the production of heme in bone marrow. ALAS2 mutations impair the activity of erythroid ALA-synthase, disrupting normal heme production and preventing erythroblasts from making enough hemoglobin. This leads to a buildup of excess iron in erythroblasts and organ damage due to iron overload [medquad_ans_0001062-3].

Additionally, mutations in the ABCB7 gene cause X-linked sideroblastic anemia and ataxia. The ABCB7 gene provides instructions for making a protein critical for heme production and iron homeostasis in developing red blood cells. ABCB7 mutations slightly alter the protein's structure, disrupting its role in heme production and iron balance

**Supported citations**: `medquad_ans_0001062-3`
**Unsupported citations**: _none_

### Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Scores**: context_relevance=0.833, groundedness=0.944, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000602-1` (score=0.3290) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifi...
2. `medquad_ans_0003601-1` (score=0.1475) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
3. `medquad_ans_0003610-1` (score=0.0108) - Limb-girdle muscular dystrophy type 2C (LGMD2C) is a condition that affects the muscles and is caused by mutations in the gamma-sarcoglycan gene. This condition belongs to a group of muscle disorders called limb-girdl...
4. `medquad_ans_0003604-1` (score=0.0029) - Limb-girdle muscular dystrophy type 2A (LGMD2A) is an autosomal recessive limb-girdle muscular dystrophy characterized by progressive, symmetrical weakness of the proximal limb and girdle muscles (mainly those around ...
5. `medquad_ans_0003608-1` (score=0.0018) - Limb-girdle muscular dystrophy type 2I (LGMD2I) is a form of limb-girdle muscular dystrophy, which refers to a group of conditions that cause weakness and wasting of the muscles in the arms and legs. The proximal musc...

**Answer**:

Limb-girdle muscular dystrophy is a group of diseases that cause weakness and wasting of the muscles in the arms and legs, particularly affecting the proximal muscles closest to the body, such as those in the shoulders, upper arms, pelvic area, and thighs [medquad_ans_0000602-1]. It is a progressive condition leading to a loss of muscle strength and bulk over time, with onset possibly occurring in childhood, adolescence, young adulthood, or later in life [medquad_ans_0003601-1]. The severity, age of onset, and features vary among the subtypes and may differ even within the same family [medquad_ans_0000602-1]. Most forms are inherited in an autosomal recessive manner, while several rare forms are inherited in an autosomal dominant pattern [medquad_ans_0003601-1].

**Supported citations**: `medquad_ans_0000602-1`, `medquad_ans_0003601-1`
**Unsupported citations**: _none_

### Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Scores**: context_relevance=0.833, groundedness=0.719, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000771-1__dup1` (score=0.9778) - The cornea and lens of your eye helps you focus. Refractive errors are vision problems that happen when the shape of the eye keeps you from focusing well. The cause could be the length of the eyeball (longer or shorte...
2. `medquad_ans_0000042-7` (score=0.9672) - Low vision can result from a variety of diseases, disorders, and injuries that affect the eye. Many people with low vision have age-related macular degeneration, cataracts, glaucoma, or diabetic retinopathy. Age-relat...
3. `medquad_ans_0000950-1__dup1` (score=0.9655) - If you have low vision, eyeglasses, contact lenses, medicine, or surgery may not help. Activities like reading, shopping, cooking, writing, and watching TV may be hard to do. The leading causes of low vision and blind...
4. `medquad_ans_0000042-2__dup3` (score=0.8272) - Eye Diseases and Health Conditions Most people develop low vision because of eye diseases and health conditions like macular degeneration, cataracts, glaucoma, and diabetes. Your eye care professional can tell the dif...
5. `medquad_ans_0000275-1__dup1` (score=0.4466) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...

**Answer**:

The provided context does not contain specific information about diabetic eye problems. While it mentions that diabetic retinopathy is one of the eye disorders that can cause low vision [medquad_ans_0000042-7], no detailed information about diabetic eye problems is available in the given text. Therefore, based on the context, diabetic eye problems cannot be fully described.

Answer: Insufficient context to describe diabetic eye problems. [medquad_ans_0000042-7]

**Supported citations**: `medquad_ans_0000042-7`
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

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001503-2` (score=0.9990) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to abs...
2. `medquad_ans_0000259-2__dup1` (score=0.0001) - What are the signs and symptoms of Alpha-thalassemia x-linked intellectual disability syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Alpha-thalassemia x-linked intellectua...
3. `medquad_ans_0001972-1` (score=0.0001) - What are the signs and symptoms of Dyskeratosis congenita autosomal recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Dyskeratosis congenita autosomal recessive. If the info...
4. `medquad_ans_0005115-2` (score=0.0001) - What are the signs and symptoms of Proteus syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Proteus syndrome. If the information is available, the table below includes how o...
5. `medquad_ans_0001971-1` (score=0.0001) - What are the signs and symptoms of Dyskeratosis congenita autosomal dominant? The Human Phenotype Ontology provides the following list of signs and symptoms for Dyskeratosis congenita autosomal dominant. If the inform...

**Answer**:

The signs and symptoms of Congenital dyserythropoietic anemia type 2 include jaundice, gallstones, an enlarged liver and spleen (splenomegaly), anemia of inadequate production, endopolyploidy on chromosome studies of bone marrow, reduced activity of N-acetylglucosaminyltransferase II, reticulocytosis, and iron overload. Iron overload can lead to complications such as an abnormal heart rhythm (arrhythmia), congestive heart failure, diabetes, and chronic liver disease (cirrhosis). Rarely, mediastinal tumors may occur [medquad_ans_0001503-2].

**Supported citations**: `medquad_ans_0001503-2`
**Unsupported citations**: _none_

### Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Scores**: context_relevance=0.800, groundedness=0.812, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000199-1__dup2` (score=0.9861) - You have two kidneys, each about the size of your fist. Their main job is to filter wastes and excess water out of your blood to make urine. They also keep the body's chemical balance, help control blood pressure, and...
2. `medquad_ans_0000058-11__dup1` (score=0.9837) - Kidney failure, also called end-stage kidney disease or ESRD, means your kidneys no longer work well enough to do their job. You will need treatment to replace the work your kidneys have stopped doing.
3. `medquad_ans_0000155-3__dup3` (score=0.9495) - Chronic kidney disease means you have damaged kidneys that cannot filter blood normally. Wastes then build up in your blood, harming your body. Kidney disease usually does not get better and may lead to kidney failure...
4. `medquad_ans_0000538-1__dup1` (score=0.9443) - Your kidneys are two bean-shaped organs, each about the size of your fists. They are located near the middle of your back, just below the rib cage. Inside each kidney about a million tiny structures called nephrons fi...
5. `medquad_ans_0000140-2__dup2` (score=0.9286) - Chronic kidney disease is kidney damage that occurs slowly over many years, often due to diabetes or high blood pressure. Once damaged, the kidneys cant filter blood as they should. This damage can cause wastes to bui...

**Answer**:

Kidney disease refers to conditions that damage the kidneys, impairing their ability to filter blood effectively. This damage can lead to the buildup of wastes and excess fluids in the body, harming overall health. Chronic kidney disease (CKD) specifically involves damaged kidneys that cannot filter blood normally, often resulting from conditions like diabetes and high blood pressure [medquad_ans_0000199-1__dup2]. Most kidney diseases affect the nephrons, the tiny structures within the kidneys responsible for filtering blood [medquad_ans_0000538-1__dup1]. Kidney disease may progress slowly over years and can eventually lead to kidney failure, also known as end-stage kidney disease (ESRD) [medquad_ans_0000155-3__dup3]. In severe cases, when kidneys fail completely, treatment options such as dialysis or kidney transplantation are required [medquad_ans_0

**Supported citations**: `medquad_ans_0000199-1__dup2`, `medquad_ans_0000538-1__dup1`, `medquad_ans_0000155-3__dup3`
**Unsupported citations**: _none_

### Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Scores**: context_relevance=0.857, groundedness=0.832, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000945-5` (score=0.9964) - These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliative Care
2. `medquad_ans_0000836-5` (score=0.9860) - These resources address the diagnosis or management of prostate cancer: - American College of Radiology: Prostate Cancer Radiation Treatment - Genetic Testing Registry: Familial prostate cancer - Genetic Testing Regis...
3. `medquad_ans_0000058-24` (score=0.9795) - Surgery, radiation therapy, and hormonal therapy all have the potential to disrupt sexual desire or performance for a short while or permanently. Discuss your concerns with your health care provider. Several options a...
4. `medquad_ans_0000058-28` (score=0.9740) - Hormonal therapy deprives cancer cells of the male hormones they need to grow and survive. This treatment is often used for prostate cancer that has spread to other parts of the body. Sometimes doctors use hormonal th...
5. `medquad_ans_0000058-4__dup3` (score=0.9707) - If tests show that you have cancer, you should talk with your doctor in order to make treatment decisions. Working With a Team of Specialists A team of specialists often treats people with cancer. The team will keep t...

**Answer**:

The treatments for prostate cancer include surgery (such as radical prostatectomy), radiation therapy, hormonal therapy, and participation in clinical trials. Surgical options include radical prostatectomy, while radiation therapy includes prostate brachytherapy and external beam radiation. Hormonal therapy is used to deprive cancer cells of male hormones and is often employed when the cancer has spread or to prevent recurrence after surgery or radiation. Additionally, patients may consider clinical trials for new treatments [medquad_ans_0000836-5].

**Supported citations**: `medquad_ans_0000836-5`
**Unsupported citations**: _none_

### Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Scores**: context_relevance=0.714, groundedness=0.883, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000586-1__dup2` (score=0.9973) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state ha...
2. `medquad_ans_0000431-1__dup1` (score=0.9972) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...
3. `medquad_ans_0000362-1__dup2` (score=0.9971) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
4. `medquad_ans_0000414-1__dup2` (score=0.9969) - Frequently Asked Questions (FAQs) Fact Sheets
5. `medquad_ans_0000589-1__dup2` (score=0.9958) - Summary : Medicare is the U.S. government's health insurance program for people age 65 or older. Some people under age 65 can qualify for Medicare, too. They include those with disabilities, permanent kidney failure, ...

**Answer**:

Yes, there is information about health insurance. Health insurance is a contract between you and your insurance company, where you buy a plan and the company agrees to pay part of your medical expenses when you need care [medquad_ans_0000431-1__dup1]. Many people obtain health insurance through their employers, and employer-sponsored plans are often managed care plans that contract with healthcare providers to offer care at reduced costs [medquad_ans_0000431-1__dup1]. Individuals can also purchase health insurance on their own or qualify for government health insurance programs such as Medicare and Medicaid [medquad_ans_0000431-1__dup1]. Medicaid is a government health insurance program that assists low-income individuals, with state-specific rules regarding income, age, disability, pregnancy, and immigration status [medquad_ans_0000586-1__dup2].

**Supported citations**: `medquad_ans_0000431-1__dup1`, `medquad_ans_0000586-1__dup2`
**Unsupported citations**: _none_

### Example 93 (`0003860-1`)

**Question**: What is (are) MECP2 duplication syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.530, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003860-1` (score=0.4641) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...
2. `medquad_ans_0000635-1` (score=0.4579) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infan...
3. `medquad_ans_0000635-4` (score=0.0009) - MECP2 duplication syndrome is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome...
4. `medquad_ans_0000814-4` (score=0.0008) - More than 99 percent of PPM-X syndrome cases occur in people with no history of the disorder in their family. Many of these cases result from new mutations in the MECP2 gene. A few families with more than one affected...
5. `medquad_ans_0000635-3` (score=0.0006) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...

**Answer**:

MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia, delayed motor milestones (e.g., sitting up, crawling), recurrent infections, poor or absent speech, seizures, and/or spasticity. The condition results from an extra copy (duplication) of the MECP2 gene in each cell, which is generally caused by a duplication of genetic material on the long (q) arm of the X chromosome. It is inherited in an X-linked manner. In males, a duplication of the only copy of the MECP2 gene in each cell is sufficient to cause the condition, whereas in females, a duplication of one of the two copies typically does not cause the disorder due to X-inactivation, which inactivates one X chromosome in each somatic cell, usually randomly [medquad_ans_0003860

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Scores**: context_relevance=0.333, groundedness=0.000, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000034_1-5` (score=0.9976) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The location and size of the ...
2. `medquad_ans_0000043_1-4` (score=0.9975) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The patient's age and general...
3. `medquad_ans_0000014_2-5` (score=0.9975) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The type and size of the tumo...
4. `medquad_ans_0000028_3-5` (score=0.9974) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer (the size of the tumor and wheth...
5. `medquad_ans_0000033_1-5` (score=0.9974) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - Whether the calcium level in the blood can be controlled...

**Answer**:

The provided context does not contain specific information about the outlook for melanoma. Therefore, based on the given documents, it is not possible to determine the outlook for melanoma. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Scores**: context_relevance=0.500, groundedness=0.095, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000298-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000643-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000101-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000179-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention "boomerang dysplasia" or specify its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether boomerang dysplasia is inherited.

Answer: Insufficient context to determine if boomerang dysplasia is inherited.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Scores**: context_relevance=0.500, groundedness=0.872, answer_relevance=0.660

**Top docs**:

1. `medquad_ans_0000466-3` (score=0.9991) - Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and the EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. The two exosto...
2. `medquad_ans_0001013-4` (score=0.0005) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000342-4` (score=0.0003) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000948-4` (score=0.0003) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000441-4` (score=0.0003) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has a parent and other family ...

**Answer**:

Mutations in the EXT1 and EXT2 genes are related to hereditary multiple osteochondromas. These mutations cause defects in the production of exostosin-1 and exostosin-2 proteins, which form a complex in the Golgi apparatus responsible for modifying heparan sulfate. When there is a mutation in exostosin-1 or exostosin-2, heparan sulfate cannot be processed correctly and becomes nonfunctional [medquad_ans_0000466-3]. A mutation in the EXT1 gene results in hereditary multiple osteochondromas type 1, while a mutation in the EXT2 gene causes hereditary multiple osteochondromas type 2. Mutations in the EXT1 gene are estimated to account for 55 to 75 percent of all cases, and symptoms tend to be more severe in type 1. Approximately 15 percent of individuals with

**Supported citations**: `medquad_ans_0000466-3`
**Unsupported citations**: _none_

### Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Scores**: context_relevance=0.125, groundedness=0.000, answer_relevance=0.625

**Top docs**:

1. `medquad_ans_0001101-2` (score=0.9996) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
2. `medquad_ans_0000484-2` (score=0.9996) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
3. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0000790-2` (score=0.9995) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
5. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.

**Answer**:

The provided context does not mention "oculofaciocardiodental syndrome" or provide specific data regarding its prevalence or incidence. Therefore, based on the given information, it is not possible to determine how many people are affected by oculofaciocardiodental syndrome.

Answer: Insufficient context [medquad_ans_0001101-2, medquad_ans_0000484-2, medquad_ans_0000195-2, medquad_ans_0000790-2, medquad_ans_0000989-2].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0001101-2, medquad_ans_0000484-2, medquad_ans_0000195-2, medquad_ans_0000790-2, medquad_ans_0000989-2`

### Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Scores**: context_relevance=0.750, groundedness=0.550, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000189-1` (score=0.0631) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused b...
2. `medquad_ans_0000445-1__dup1` (score=0.0302) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...
3. `medquad_ans_0000065-1` (score=0.0260) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down ...
4. `medquad_ans_0001350-1` (score=0.0046) - Citrullinemia type I is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. This condition, also known as classic citrullinemia, belongs to a class of genetic diseases call...
5. `medquad_ans_0004334-1` (score=0.0025) - Neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) is a liver condition is also known as neonatal-onset type II citrullinemia. NICCD blocks the flow of bile (a digestive fluid produced by the liver)...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused by mutations in different genes. Type I citrullinemia (also known as classic citrullinemia) usually becomes evident in the first few days of life. Affected infants typically appear normal at birth, but as ammonia builds up in the body they experience a progressive lack of energy (lethargy), poor feeding, vomiting, seizures, and loss of consciousness. These medical problems are life-threatening in many cases. Less commonly, a milder form of type I citrullinemia can develop later in childhood or adulthood. This later-onset form is associated with intense headaches, partial loss of vision, problems with balance and muscle coordination (ataxia), and lethargy. Some people with gene mutations that cause type

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

**Scores**: context_relevance=0.429, groundedness=0.620, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000943-5` (score=0.9942) - These resources address the diagnosis or management of spondylocarpotarsal synostosis syndrome: - Gene Review: Gene Review: FLNB-Related Disorders - Genetic Testing Registry: Spondylocarpotarsal synostosis syndrome Th...
2. `medquad_ans_0000949-5` (score=0.9918) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from Medli...
3. `medquad_ans_0000944-5` (score=0.9918) - These resources address the diagnosis or management of spondylocostal dysostosis: - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive - Genetic Testing Registry: Jarcho-Levin syndrome - Genetic ...
4. `medquad_ans_0000187-5` (score=0.9854) - These resources address the diagnosis or management of CHST3-related skeletal dysplasia: - Gene Review: Gene Review: CHST3-Related Skeletal Dysplasia - Genetic Testing Registry: Spondyloepiphyseal dysplasia with conge...
5. `medquad_ans_0000946-5` (score=0.1287) - These resources address the diagnosis or management of spondyloepimetaphyseal dysplasia, Strudwick type: - Genetic Testing Registry: Spondyloepimetaphyseal dysplasia Strudwick type - MedlinePlus Encyclopedia: Clubfoot...

**Answer**:

The provided context does not specify treatments for spondylocostal dysostosis. While it lists resources such as Gene Reviews, Genetic Testing Registry entries, and MedlinePlus topics like X-ray exams and scoliosis, it does not detail specific treatment approaches. Therefore, based on the given information, the treatments for spondylocostal dysostosis cannot be determined. [medquad_ans_0000944-5]

**Supported citations**: `medquad_ans_0000944-5`
**Unsupported citations**: _none_

### Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.946, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004825-1` (score=0.9994) - What are the signs and symptoms of Perry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Perry syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0003303-1` (score=0.0001) - What are the signs and symptoms of Johnson neuroectodermal syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Johnson neuroectodermal syndrome. If the information is available...
3. `medquad_ans_0001725-2` (score=0.0001) - What are the signs and symptoms of De Barsy syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for De Barsy syndrome. If the information is available, the table below includes how...
4. `medquad_ans_0001689-1` (score=0.0001) - What are the signs and symptoms of Curry Jones syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Curry Jones syndrome. If the information is available, the table below includ...
5. `medquad_ans_0005008-2` (score=0.0001) - What are the signs and symptoms of Potocki-Shaffer syndrome? The signs and symptoms can vary depending on the area and amount deleted. Some individuals with the syndrome have few issues and lead a normal life while ot...

**Answer**:

The symptoms of Perry syndrome include abnormality of extrapyramidal motor function, respiratory insufficiency, sleep disturbance, tremor, weight loss, developmental regression, hallucinations, hypotension, apathy, autosomal dominant inheritance, bradykinesia, central hypoventilation, dysarthria, hypoventilation, inappropriate behavior, insomnia, mask-like facies, parkinsonism, rapidly progressive course, rigidity, short stepped shuffling gait, vertical supranuclear gaze palsy, and weak voice. These symptoms occur in approximately 90% of patients, except for developmental regression, hallucinations, hypotension, apathy, and others which occur in about 7.5% of patients [medquad_ans_0004825-1].

**Supported citations**: `medquad_ans_0004825-1`
**Unsupported citations**: _none_

### Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Scores**: context_relevance=1.000, groundedness=0.904, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0006398-2` (score=0.9992) - What are the signs and symptoms of Warfarin sensitivity? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin sensitivity. If the information is available, the table below includ...
2. `medquad_ans_0006399-1` (score=0.9876) - What are the signs and symptoms of Warfarin syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin syndrome. If the information is available, the table below includes how...
3. `medquad_ans_0000108-5__dup1` (score=0.0013) - Major Signs and Symptoms Signs and symptoms of pulmonary embolism (PE) include unexplained shortness of breath, problems breathing, chest pain, coughing, or coughing up blood. An arrhythmia (irregular heartbeat) also ...
4. `medquad_ans_0001028-2` (score=0.0001) - The prevalence of warfarin sensitivity is unknown. However, it appears to be more common in people who are older, those with lower body weights, and individuals of Asian ancestry. Of the approximately 2 million people...
5. `medquad_ans_0003545-2` (score=0.0001) - What are the signs and symptoms of LEOPARD syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for LEOPARD syndrome. If the information is available, the table below includes how o...

**Answer**:

The signs and symptoms of Warfarin sensitivity include abnormality of blood and blood-forming tissues, abnormality of metabolism/homeostasis, and autosomal dominant inheritance. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which collects information from sources such as Orphanet. However, specific symptom frequencies are not detailed in the provided context [medquad_ans_0006398-2].

**Supported citations**: `medquad_ans_0006398-2`
**Unsupported citations**: _none_

### Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.947, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0004262-1` (score=0.7196) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, a...
2. `medquad_ans_0000701-1` (score=0.6933) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most...
3. `medquad_ans_0000701-2` (score=0.0005) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.
4. `medquad_ans_0004688-2` (score=0.0002) - What are the signs and symptoms of Pallister-Killian mosaic syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pallister-Killian mosaic syndrome. If the information is availab...
5. `medquad_ans_0001725-2` (score=0.0001) - What are the signs and symptoms of De Barsy syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for De Barsy syndrome. If the information is available, the table below includes how...

**Answer**:

Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement of the muscles (muscle hypertrophy), and/or joint stiffness. It is caused by mutations in the SMAD4 gene and inherited in an autosomal dominant pattern, with most cases resulting from a new mutation [medquad_ans_0004262-1]. The condition affects multiple systems and includes delayed development of language and motor skills, hearing loss (sensorineural, conductive, or mixed), growth restriction from before birth through adolescence, skin stiffness, and muscular appearance. Skeletal abnormalities include thickened skull bones, flattened spine bones (platyspondyly), broad ribs, underdeveloped pelvic wings (hypoplastic iliac wings), and short fingers and toes (brachydactyly). Facial features commonly include short

**Supported citations**: `medquad_ans_0004262-1`
**Unsupported citations**: _none_

### Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006057-2` (score=0.9994) - What are the signs and symptoms of Thanatophoric dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Thanatophoric dysplasia. If the information is available, the table below ...
2. `medquad_ans_0006045-2` (score=0.0001) - What are the signs and symptoms of Tetralogy of Fallot? The Human Phenotype Ontology provides the following list of signs and symptoms for Tetralogy of Fallot. If the information is available, the table below includes...
3. `medquad_ans_0005369-1` (score=0.0001) - What are the signs and symptoms of Rhizomelic syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Rhizomelic syndrome. If the information is available, the table below includes...
4. `medquad_ans_0004622-1` (score=0.0001) - What are the signs and symptoms of Osteoglophonic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteoglophonic dysplasia. If the information is available, the table belo...
5. `medquad_ans_0003780-1` (score=0.0001) - What are the signs and symptoms of Mandibuloacral dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Mandibuloacral dysplasia. If the information is available, the table belo...

**Answer**:

The symptoms of Thanatophoric dysplasia include abnormality of the femur, abnormality of the metaphyses, abnormality of the sacroiliac joint, aplasia/hypoplasia of the lungs, bowing of the long bones, brachydactyly syndrome, cognitive impairment, cutis laxa, depressed nasal bridge, increased nuchal translucency, macrocephaly, malar flattening, micromelia, muscular hypotonia, narrow chest, platyspondyly, respiratory insufficiency, short stature, short thorax, skeletal dysplasia, small face, split hand, abnormality of neuronal migration, frontal bossing, hearing impairment, intrauterine growth retardation, kyphosis, polyhydramnios, proptosis, ventriculomegaly, abnormality of the kidney, acanthosis nigricans, atria septal defect,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002417-2` (score=0.9990) - What are the signs and symptoms of Fragile X syndrome? Fragile X syndrome is characterized by developmental problems including intellectual disability and delayed speech and language development. Males are usually mor...
2. `medquad_ans_0002458-2` (score=0.0007) - What are the signs and symptoms of Fryns syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns syndrome. If the information is available, the table below includes how often...
3. `medquad_ans_0000259-2__dup1` (score=0.0001) - What are the signs and symptoms of Alpha-thalassemia x-linked intellectual disability syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Alpha-thalassemia x-linked intellectua...
4. `medquad_ans_0004036-1` (score=0.0001) - What are the signs and symptoms of Microduplication Xp11.22-p11.23 syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Microduplication Xp11.22-p11.23 syndrome. If the informat...
5. `medquad_ans_0006051-2` (score=0.0001) - What are the signs and symptoms of Tetrasomy X? The Human Phenotype Ontology provides the following list of signs and symptoms for Tetrasomy X. If the information is available, the table below includes how often the s...

**Answer**:

The symptoms of Fragile X syndrome include cognitive impairment, joint hypermobility, macroorchidism, neurological speech impairment, otitis media, pes planus, abnormality of the pinna, attention deficit hyperactivity disorder, frontal bossing, intellectual disability, moderate, long face, macrocephaly, mandibular prognathia, muscular hypotonia, narrow face, sinusitis, abnormality of the mitral valve, autism, cerebral cortical atrophy, dilatation of the ascending aorta, seizures, self-injurious behavior, and strabismus [medquad_ans_0002417-2]. Additionally, males may have enlarged testicles (macroorchidism) after puberty, and physical features such as a long and narrow face, large ears, prominent jaw and forehead, unusually flexible fingers, and flat feet may become more apparent with age [medquad_ans_000241

**Supported citations**: `medquad_ans_0002417-2`
**Unsupported citations**: _none_

### Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Scores**: context_relevance=0.750, groundedness=0.960, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000056-1__dup1` (score=0.2599) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease that occurs when the thickening of blood vessel walls blocks the fl...
2. `medquad_ans_0000889-1__dup1` (score=0.2255) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited disease of the blood vessels that occurs when the thickening of blood vessel walls blocks the flow ...
3. `medquad_ans_0000166-1` (score=0.1349) - Cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy, usually called CADASIL, is an inherited condition that causes stroke and other impairments. This condition affects blood flow...
4. `medquad_ans_0001051-1__dup1` (score=0.0563) - Cerebral autosomal recessive arteriopathy with subcortical infarcts and leukoencephalopathy, commonly known as CARASIL, is an inherited condition that causes stroke and other impairments. This progressive condition is...
5. `medquad_ans_0000167-1` (score=0.0264) - Cerebral autosomal recessive arteriopathy with subcortical infarcts and leukoencephalopathy, commonly known as CARASIL, is an inherited condition that causes stroke and other impairments. Abnormalities affecting the b...

**Answer**:

CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease caused by a mutation in the NOTCH3 gene, which leads to thickening of the blood vessel walls and blocks blood flow to the brain, particularly affecting small blood vessels in the white matter of the brain [medquad_ans_0000056-1__dup1]. It is characterized by migraine headaches, multiple strokes, cognitive deterioration, seizures, vision problems, psychiatric symptoms such as severe depression and behavioral changes, and an increased risk of heart attack [medquad_ans_0000056-1__dup1]. The disease progresses to dementia and typically presents with symptoms in the mid-30s, though onset can vary widely and some individuals may not show signs until later in life. CADASIL is inherited

**Supported citations**: `medquad_ans_0000056-1__dup1`
**Unsupported citations**: _none_

### Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Scores**: context_relevance=0.833, groundedness=0.613, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0004185-1` (score=0.9647) - Multiple endocrine neoplasia, type 1 (MEN1) is an inherited condition that causes tumors of the endocrine system (the body's network of hormone-producing glands). People affected by MEN1 typically develop tumors of th...
2. `medquad_ans_0004187-1` (score=0.9426) - Multiple endocrine neoplasia type 2A (MEN 2A) is is an inherited disorder caused by mutations in the RET gene. Individuals with MEN 2A are at high risk of developing medullary carcinoma of the thyroid. About 50% will ...
3. `medquad_ans_0000010-1__dup6` (score=0.9398) - MEN1 is an inherited disorder that causes tumors in the endocrine glands and the duodenum, the first part of the small intestine. MEN1 is sometimes called multiple endocrine adenomatosis or Wermer's syndrome, after on...
4. `medquad_ans_0000688-1` (score=0.6657) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate...
5. `medquad_ans_0000014-5__dup1` (score=0.0088) - In about 80 percent of people with primary hyperparathyroidism, a benign, or noncancerous, tumor called an adenoma has formed in one of the parathyroid glands.2 The tumor causes the gland to become overactive. In most...

**Answer**:

Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the blood, controlling and instructing the functions of various organs. In these disorders, multiple endocrine glands form tumors and become hormonally overactive, often at the same time. The overactive glands may include the parathyroids, pancreas, or pituitary [medquad_ans_0000688-1]. Specific types, such as MEN1 and MEN2A, are caused by mutations in genes like MEN1 and RET, respectively, and are inherited in an autosomal dominant manner [medquad_ans_0004185-1, medquad_ans_0004187-1]. MEN1 typically causes tumors in the parathyroid, pituitary, and pancreas, while MEN

**Supported citations**: `medquad_ans_0000688-1`
**Unsupported citations**: `medquad_ans_0004185-1, medquad_ans_0004187-1`

### Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004155-3` (score=0.9959) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycos...
2. `medquad_ans_0000679-2` (score=0.0011) - MPS II occurs in approximately 1 in 100,000 to 1 in 170,000 males.
3. `medquad_ans_0000199-1__dup1` (score=0.0001) - The mucolipidoses (ML) are a group of inherited metabolic diseases that affect the bodys ability to carry out the normal turnover of various materials within cells. In ML, abnormal amounts of carbohydrates and fatty m...
4. `medquad_ans_0000200-3__dup1` (score=0.0001) - The mucopolysaccharidoses syndromes share many clinical features but have varying degrees of severity. Most individuals with a mucopolysaccharidosis syndrome generally experience a period of normal development followe...
5. `medquad_ans_0004155-1` (score=0.0001) - Mucopolysaccharidosis I (MPS I) is a condition that affects many parts of the body. It is a progressively debilitating disorder; however, the rate of progression varies among affected individuals. MPS I is caused by m...

**Answer**:

Mutations in the IDUA gene cause Mucopolysaccharidosis type I (MPS I). The IDUA gene provides instructions for producing an enzyme involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs). Mutations in the IDUA gene reduce or completely eliminate the function of the IDUA enzyme, leading to the accumulation of GAGs within cells, specifically inside the lysosomes. This accumulation results in lysosomal storage disorder characteristics [medquad_ans_0004155-3].

**Supported citations**: `medquad_ans_0004155-3`
**Unsupported citations**: _none_

### Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Scores**: context_relevance=0.900, groundedness=0.965, answer_relevance=0.580

**Top docs**:

1. `medquad_ans_0000825-3` (score=0.9992) - Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition. The FLCN gene provides instructions for making a protein called follicu...
2. `medquad_ans_0000126-3` (score=0.9992) - Mutations in the FLCN gene cause Birt-Hogg-Dub syndrome. This gene provides instructions for making a protein called folliculin. The normal function of this protein is unknown, but researchers believe that it may act ...
3. `medquad_ans_0000963-3` (score=0.9989) - Surfactant dysfunction is caused by mutations in one of several genes, including SFTPB, SFTPC, and ABCA3. Each of these genes is involved in the production of surfactant. The production and release of surfactant is a ...
4. `medquad_ans_0000823-4` (score=0.4817) - This condition is generally not inherited but arises from gene mutations that occur in early blood-forming cells after conception. These alterations are called somatic mutations.
5. `medquad_ans_0000608-3` (score=0.0047) - Cancers occur when genetic mutations build up in critical genes, specifically those that control cell growth and division or the repair of damaged DNA. These changes allow cells to grow and divide uncontrollably to fo...

**Answer**:

Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition [medquad_ans_0000825-3]. The FLCN gene provides instructions for making a protein called folliculin, which is found in lung connective tissue cells and in cells that line the alveoli. Researchers believe folliculin may help control cell growth and division and may play a role in repairing and re-forming lung tissue after damage. One theory is that altered folliculin protein may trigger inflammation in lung tissue, leading to tissue damage and the formation of blebs, which increase the risk of pneumothorax [medquad_ans_0000825-3]. However, the exact mechanism by which FLCN mutations lead to pneumothorax is not fully understood. Primary spontaneous pneumothorax most often occurs in individuals without an identified

**Supported citations**: `medquad_ans_0000825-3`
**Unsupported citations**: _none_

### Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Scores**: context_relevance=0.750, groundedness=0.906, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000032-13` (score=0.9905) - At the first sign of a stroke, you should get medical care right away. If blood vessels to your brain are blocked by blood clots, the doctor can give you a "clot-busting" drug. The drug must be given soon after a stro...
2. `medquad_ans_0000129-4__dup1` (score=0.9755) - The National Institute of Neurological Disorders and Stroke (NINDS), a component of the National Institutes of Health (NIH) within the U.S. Department of Health and Human Services, is the nations primary funding sourc...
3. `medquad_ans_0000527-1__dup2` (score=0.9744) - A stroke is a medical emergency. There are two types - ischemic and hemorrhagic. Ischemic stroke is the most common type. It is usually caused by a blood clot that blocks or plugs a blood vessel in the brain. This kee...
4. `medquad_ans_0000857-1__dup2` (score=0.9664) - A stroke is a medical emergency. Strokes happen when blood flow to your brain stops. Within minutes, brain cells begin to die. There are two kinds of stroke. The more common kind, called ischemic stroke, is caused by ...
5. `medquad_ans_0000066-13` (score=0.9599) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blo...

**Answer**:

A stroke is a medical emergency that occurs when blood flow to the brain stops, leading to the death of brain cells within minutes. There are two main types: ischemic stroke, the most common type, which is caused by a blood clot that blocks or plugs a blood vessel in the brain, and hemorrhagic stroke, which involves bleeding in the brain. Ischemic stroke can also result from stenosis (narrowing of the artery) due to atherosclerosis, where plaque builds up in the arteries. A transient ischemic attack (TIA) is a brief interruption of blood supply to the brain and can indicate a risk for a more serious stroke [medquad_ans_0000527-1__dup2]. Symptoms include sudden numbness or weakness (especially on one side of the body), confusion, trouble speaking or understanding speech, sudden trouble seeing, difficulty walking, dizziness, loss of balance, or a sudden

**Supported citations**: `medquad_ans_0000527-1__dup2`
**Unsupported citations**: _none_

### Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.992, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006276-3` (score=0.9953) - What causes uncombable hair syndrome? The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been s...
2. `medquad_ans_0000048-3__dup1` (score=0.0099) - Although the skin abnormalities usually regress, and sometimes disappear completely, there may be residual neurological difficulties.
3. `medquad_ans_0000742-1__dup1` (score=0.0001) - What are the signs and symptoms of Bjornstad syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Bjornstad syndrome. If the information is available, the table below includes h...
4. `medquad_ans_0006276-6` (score=0.0001) - How might uncombable hair syndrome be treated? There is no definitive treatment for uncombable hair syndrome, but the condition usually improves or resolves on its own with the onset of puberty. Gentle hair care is ge...
5. `medquad_ans_0006276-1` (score=0.0001) - Uncombable hair syndrome (UHS) is a rare disorder of the hair shaft of the scalp. It usually is characterized by silvery-blond or straw-colored hair that is disorderly; stands out from the scalp; and cannot be combed ...

**Answer**:

The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been suggested that the condition may result from premature keratinization (development of keratin) of the inner root sheath, which forms the channel for the growing hair. The inner root sheath conforms in configuration to the abnormal outline of the hair shaft. It thus forms an irregular, rigid tube that then alters the shape of the emerging hair. While it is assumed that the condition is autosomal dominant and thus due to changes (mutations) in a gene, no responsible gene has been identified [medquad_ans_0006276-3].

**Supported citations**: `medquad_ans_0006276-3`
**Unsupported citations**: _none_

### Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000441-4` (score=0.9987) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has a parent and other family ...
3. `medquad_ans_0000742-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000193-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000674-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain information about the inheritance pattern of Russell-Silver syndrome. Therefore, it is not possible to determine based on the given text whether Russell-Silver syndrome is inherited [doc_id].

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Scores**: context_relevance=0.875, groundedness=0.951, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000414-1__dup2` (score=0.9975) - Frequently Asked Questions (FAQs) Fact Sheets
2. `medquad_ans_0000969-1__dup2` (score=0.9971) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
3. `medquad_ans_0000001-7__dup1` (score=0.9958) - Topics
4. `medquad_ans_0000968-1__dup2` (score=0.0360) - Summary : Keeping a healthy weight is crucial. If you are underweight, overweight, or obese, you may have a higher risk of certain health problems. About two thirds of adults in the U.S. are overweight or obese. Achie...
5. `medquad_ans_0000111-1__dup2` (score=0.0073) - Summary : Do you know if your current weight is healthy? "Underweight", "normal", "overweight", and "obese" are all labels for ranges of weight. Obese and overweight mean that your weight is greater than it should be ...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are different types of weight loss surgery, which often limit the amount of food you can take in. Some types also affect how you digest food and absorb nutrients. All types have risks and complications, such as infections, hernias, and blood clots. Many people who have the surgery lose weight quickly, but some regain weight later. Following diet and exercise recommendations can help keep most of the weight off. Lifelong medical follow-up is required. [medquad_ans_0000969-1__dup2]

**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

### Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.871, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000172-2__dup1` (score=0.9970) - Treatment is symptomatic and supportive, and may include drug therapy.
2. `medquad_ans_0000889-5` (score=0.9947) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Sae...
3. `medquad_ans_0000718-5` (score=0.9928) - These resources address the diagnosis or management of NOMID: - Genetic Testing Registry: Chronic infantile neurological, cutaneous and articular syndrome These resources from MedlinePlus offer information about the d...
4. `medquad_ans_0000071-2__dup1` (score=0.0297) - There is no standard course of treatment for Sotos syndrome. Treatment is symptomatic.
5. `medquad_ans_0000669-5` (score=0.0034) - These resources address the diagnosis or management of monilethrix: - Genetic Testing Registry: Beaded hair These resources from MedlinePlus offer information about the diagnosis and management of various health condi...

**Answer**:

Treatment for Saethre-Chotzen syndrome is symptomatic and supportive, and may include drug therapy [medquad_ans_0000172-2__dup1]. Additional management options include diagnostic tests, surgery and rehabilitation, genetic counseling, and palliative care, as these are resources available for the diagnosis and management of the condition [medquad_ans_0000889-5].

**Supported citations**: `medquad_ans_0000172-2__dup1`, `medquad_ans_0000889-5`
**Unsupported citations**: _none_

### Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=1.000, groundedness=0.497, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000078-1__dup2` (score=0.9214) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...
2. `medquad_ans_0000512-1` (score=0.2314) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effect...
3. `medquad_ans_0000758-1__dup2` (score=0.1210) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
4. `medquad_ans_0000009-1__dup2` (score=0.0159) - Asbestos-related lung diseases are diseases caused by exposure to asbestos (as-BES-tos) fibers. Asbestos is a mineral that, in the past, was widely used in many industries. Asbestos is made up of tiny fibers that can ...
5. `medquad_ans_0000522-1__dup1` (score=0.0140) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...

**Answer**:

Idiopathic pulmonary fibrosis (IPF) is a chronic, progressive lung disease in which scar tissue (fibrosis) builds up in the lungs, making them unable to transport oxygen into the bloodstream effectively. It usually affects people between the ages of 50 and 70 and is characterized by symptoms such as shortness of breath, a persistent dry, hacking cough, loss of appetite, gradual weight loss, and in some cases, widened and rounded tips of the fingers and toes (clubbing) due to oxygen shortage. IPF has no known cause in most cases and is referred to as "idiopathic." When more than one family member has the disease, it is called familial IPF. The disease progresses over time, leading to reduced oxygen supply to the body's organs and tissues, and it is associated with serious complications such as respiratory failure, pulmonary hypertension, heart failure, pulmonary embolism, pneumonia, and lung cancer. Currently,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Scores**: context_relevance=0.167, groundedness=0.000, answer_relevance=0.567

**Top docs**:

1. `medquad_ans_0000118-4__dup1` (score=0.9984) - The NINDS supports and conducts research on brain diseases. Much of this research is aimed at characterizing the agents that cause these disorders, clarifying the mechanisms underlying them, and, ultimately, finding w...
2. `medquad_ans_0000043-4__dup1` (score=0.9984) - The NINDS supports a broad program of research on disorders of the nervous system, including BEB. Much of this research is aimed at increasing understanding of these disorders and finding ways to prevent, treat, and c...
3. `medquad_ans_0000075-4__dup1` (score=0.9983) - The NINDS supports research on movement disorders such as chorea. The goals of this research are to increase understanding of these disorders and to find ways to prevent and treat them.
4. `medquad_ans_0000264-4__dup1` (score=0.9982) - The NINDS conducts and supports a broad range of research on movement disorders including tardive dyskinesia. The goals of this research are to improve understanding of these disorders and to discover ways to treat, p...
5. `medquad_ans_0000137-4__dup1` (score=0.9982) - The NINDS supports and conducts research on TSEs, including GSS. Much of this research is aimed at characterizing the agents that cause these disorders, clarifying the mechanisms underlying them, and, ultimately, find...

**Answer**:

The provided context does not mention Neuroleptic Malignant Syndrome or any research specifically directed at it. Therefore, based on the given information, no research or clinical trials for Neuroleptic Malignant Syndrome are described. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Scores**: context_relevance=0.714, groundedness=0.919, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000203-8` (score=0.9959) - To diagnose kidney stones, the health care provider will perform a physical exam and take a medical history. The medical history may include questions about family history of kidney stones, diet, GI problems, and othe...
2. `medquad_ans_0000204-7` (score=0.9953) - The process of diagnosing any illness begins with consideration of the symptoms. Pain or bloody urine may be the first symptom. Urine, blood, and imaging tests will help determine whether symptoms are caused by a ston...
3. `medquad_ans_0000221-8` (score=0.9934) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other he...
4. `medquad_ans_0000221-6` (score=0.0924) - You may have a kidney stone if you - have pain while urinating - see blood in your urine - feel a sharp pain in your back or lower abdomenthe area between your chest and hips The pain may last for a short or long time...
5. `medquad_ans_0000650-2` (score=0.0335) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

To diagnose kidney stones in adults, a healthcare provider will perform a physical exam and take a medical history, which may include questions about family history, diet, gastrointestinal problems, and other diseases. Diagnostic tests include urinalysis, blood tests, abdominal x-ray, and CT scans. Urinalysis tests a urine sample to detect infection or substances that form stones. Blood tests identify biochemical abnormalities that may lead to kidney stones. An abdominal x-ray uses minimal radiation to show the location of stones in the kidney or urinary tract. CT scans use x-rays and computer technology to create three-dimensional images and can show stone locations and conditions that may have caused the stone to form, possibly with the use of contrast medium [medquad_ans_0000203-8].

**Supported citations**: `medquad_ans_0000203-8`
**Unsupported citations**: _none_

### Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.848, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000205-7` (score=0.2931) - Urine retention may occur either because the bladder wall muscles cannot contract or because the sphincter muscles cannot relax. Catheter. A catheter is a thin tube that can be inserted through the urethra into the bl...
2. `medquad_ans_0000041_1-1` (score=0.0700) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of ...
3. `medquad_ans_0000036_1-1` (score=0.0131) - Key Points - Prostate cancer is a disease in which malignant (cancer) cells form in the tissues of the prostate. - Signs of prostate cancer include a weak flow of urine or frequent urination. - Tests that examine the ...
4. `medquad_ans_0000211-4__dup2` (score=0.0122) - Urinary incontinence in men results when the brain does not properly signal the bladder, the sphincters do not squeeze strongly enough, or both. The bladder muscle may contract too much or not enough because of a prob...
5. `medquad_ans_0000213-3__dup2` (score=0.0044) - Urinary retention can result from - obstruction of the urethra - nerve problems - medications - weakened bladder muscles Obstruction of the Urethra Obstruction of the urethra causes urinary retention by blocking the n...

**Answer**:

The provided context does not define or describe urethral cancer. It mentions urethral cancer as a disease in which malignant (cancer) cells form in the tissues of the urethra, but no further details are given. [medquad_ans_0000041_1-1]

**Supported citations**: `medquad_ans_0000041_1-1`
**Unsupported citations**: _none_

### Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000682-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000400-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0001094-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000961-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention paramyotonia congenita or its mode of inheritance. Therefore, it is not possible to determine from the given information whether paramyotonia congenita is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.530, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000735-2__dup1` (score=0.9987) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound bio...
2. `medquad_ans_0000004-1__dup1` (score=0.0001) - What are the signs and symptoms of ABCD syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for ABCD syndrome. If the information is available, the table below includes how often t...
3. `medquad_ans_0003986-2` (score=0.0001) - What are the signs and symptoms of Methylcobalamin deficiency cbl G type? The Human Phenotype Ontology provides the following list of signs and symptoms for Methylcobalamin deficiency cbl G type. If the information is...
4. `medquad_ans_0005222-1` (score=0.0001) - What are the signs and symptoms of Pyruvate decarboxylase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyruvate decarboxylase deficiency. If the information is availab...
5. `medquad_ans_0003407-1` (score=0.0001) - What are the signs and symptoms of Kniest like dysplasia lethal? The Human Phenotype Ontology provides the following list of signs and symptoms for Kniest like dysplasia lethal. If the information is available, the ta...

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), reduced consciousness/confusion (7.5%), respiratory insufficiency (7.5%), skin ulcer (7.5%), and visual field defect (7.5%). Other associated symptoms may include feeding difficulties in infancy, recurrent skin infections, seborrheic dermatitis, sensorineural hearing impairment, hepatomegaly, hyperammonemia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Scores**: context_relevance=0.429, groundedness=0.917, answer_relevance=0.543

**Top docs**:

1. `medquad_ans_0000966-1__dup2` (score=0.9971) - Summary : We all need clean water. People need it to grow crops and to operate factories, and for drinking and recreation. Fish and wildlife depend on it to survive. Many different pollutants can harm our rivers, stre...
2. `medquad_ans_0000001-7__dup1` (score=0.9961) - Topics
3. `medquad_ans_0000269-1__dup4` (score=0.9954) - Frequently Asked Questions (FAQs) Vector Information
4. `medquad_ans_0000414-1__dup2` (score=0.9950) - Frequently Asked Questions (FAQs) Fact Sheets
5. `medquad_ans_0000298-1__dup1` (score=0.0371) - Summary : We all need to drink water. How much you need depends on your size, activity level, and the weather where you live. The water you drink is a combination of surface water and groundwater. Surface water includ...

**Answer**:

Yes, there is information about water pollution. The three most common pollutants are soil, nutrients, and bacteria. Rain washes soil into streams and rivers, which can kill fish eggs, clog fish gills, and block light, causing plants to die. Nutrients, often from fertilizers, cause algae growth in lakes, ponds, and reservoirs, turning water green. Bacteria, often from sewage spills, can pollute both fresh and salt water. To help protect water supplies, avoid pouring household products like cleansers, medicines, and auto fluids down the drain; dispose of excess grease, diapers, condoms, and personal hygiene products in the garbage; and clean up after pets, as pet waste contains nutrients and germs [medquad_ans_0000966-1__dup2].

**Supported citations**: `medquad_ans_0000966-1__dup2`
**Unsupported citations**: _none_

### Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Scores**: context_relevance=0.429, groundedness=0.000, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000222-3__dup1` (score=0.9984) - The prognosis for those with paresthesia depends on the severity of the sensations and the associated disorders.
2. `medquad_ans_0000201-3__dup1` (score=0.9981) - Improvement in muscle strength usually begins within 3 to 6 weeks after treatment is started. Most patients who receive treatment early experience little, if any, disability. However, there is evidence of slow progres...
3. `medquad_ans_0000185-3__dup1` (score=0.9980) - While in rare cases some patients may regain certain functions, the chances for motor recovery are very limited.
4. `medquad_ans_0000257-3__dup1` (score=0.9976) - Recovery depends upon how quickly treatment is received and how severely the body is compromised. Paralysis may persist for many weeks or be permanent. Most individuals have a good chance of recovery.
5. `medquad_ans_0000203-3__dup1` (score=0.9951) - The disease tends to advance rapidly over the course of 5 to 10 years, with progressive loss of motor skills, eventual confinement to bed, and death. There is no remission from the disease. There is currently no cure.

**Answer**:

The provided context does not contain specific information about the outlook for Foot Drop. Therefore, based on the given text, it is not possible to determine the outlook for Foot Drop. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000145-1` (score=0.7351) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and h...
2. `medquad_ans_0000915-1__dup1` (score=0.3966) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
3. `medquad_ans_0000916-1__dup1` (score=0.3966) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
4. `medquad_ans_0000874-2` (score=0.0041) - This condition affects an estimated 1 in 8,500 females.
5. `medquad_ans_0000989-2` (score=0.0031) - This condition affects an estimated 1 in 50,000 people.

**Answer**:

Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are also affected. The thickened bones can lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. The age at which affected individuals first experience symptoms varies greatly; however, most people with this condition develop pain or weakness by adolescence. Camurati-Engelmann disease is caused by a mutation in the TGFB1 gene which is inherited in an autosomal dominant fashion. In some instances, people have the gene mutation that causes Camurati-Engelmann disease but never develop the characteristic features of this condition. In others, features are present, but a mutation cannot be identified. These cases are referred to as Camurati-Engelmann disease type

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Scores**: context_relevance=1.000, groundedness=0.978, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002080-2` (score=0.9994) - What are the signs and symptoms of Emery-Dreifuss muscular dystrophy, X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Emery-Dreifuss muscular dystrophy, X-linked. If the in...
2. `medquad_ans_0001081-2__dup1` (score=0.0001) - What are the signs and symptoms of Chanarin-Dorfman syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Chanarin-Dorfman syndrome. If the information is available, the table be...
3. `medquad_ans_0003604-2` (score=0.0001) - What are the signs and symptoms of Limb-girdle muscular dystrophy type 2A? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy type 2A. If the information ...
4. `medquad_ans_0003608-2` (score=0.0001) - What are the signs and symptoms of Limb-girdle muscular dystrophy type 2I? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy type 2I. If the information ...
5. `medquad_ans_0006492-1` (score=0.0001) - What are the signs and symptoms of X-linked Charcot-Marie-Tooth disease type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked Charcot-Marie-Tooth disease type 1. If the in...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked include Achilles tendon contracture, atrioventricular block, childhood onset, decreased cervical spine flexion due to contractures of posterior cervical muscles, elbow flexion contracture, elevated serum creatine phosphokinase, juvenile onset, pectus excavatum, primary atrial arrhythmia, slow progression, sudden cardiac death, type 1 muscle fiber atrophy, waddling gait, and X-linked recessive inheritance [medquad_ans_0002080-2].

**Supported citations**: `medquad_ans_0002080-2`
**Unsupported citations**: _none_

### Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Scores**: context_relevance=0.625, groundedness=0.751, answer_relevance=0.725

**Top docs**:

1. `medquad_ans_0000193-5` (score=0.9973) - These resources address the diagnosis or management of Coats plus syndrome: - Genetic Testing Registry: Cerebroretinal microangiopathy with calcifications and cysts These resources from MedlinePlus offer information a...
2. `medquad_ans_0000455-5` (score=0.9970) - These resources address the diagnosis or management of HANAC syndrome: - Gene Review: Gene Review: COL4A1-Related Disorders - Genetic Testing Registry: Angiopathy, hereditary, with nephropathy, aneurysms, and muscle c...
3. `medquad_ans_0000172-5` (score=0.9956) - These resources address the diagnosis or management of Char syndrome: - Gene Review: Gene Review: Char Syndrome - Genetic Testing Registry: Char syndrome - MedlinePlus Encyclopedia: Patent Ductus Arteriosus These reso...
4. `medquad_ans_0000072-5` (score=0.9945) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins Mc...
5. `medquad_ans_0000129-2__dup1` (score=0.9911) - There is no standard protocol to treat FMD. Any treatment to improve blood flow is based on the arteries affected and the progression and severity of the disease. The carotid arteries should be tested if FMD is found ...

**Answer**:

The provided context does not contain specific information about the treatments for arterial tortuosity syndrome. While it mentions resources related to the diagnosis and management of arterial tortuosity syndrome, including a Gene Review and Genetic Testing Registry, it does not detail any specific treatment protocols [medquad_ans_0000072-5]. Therefore, based on the given context, the treatments for arterial tortuosity syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

### Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Scores**: context_relevance=1.000, groundedness=0.970, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001092-1__dup1` (score=0.9994) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
2. `medquad_ans_0001105-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is ava...
3. `medquad_ans_0001110-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is ava...
4. `medquad_ans_0001108-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2J? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2J. If the information is ava...
5. `medquad_ans_0001112-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2O? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2O. If the information is ava...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 1C include decreased motor nerve conduction velocity, distal amyotrophy, distal muscle weakness, distal sensory impairment, hypertrophic nerve changes, hyporeflexia, juvenile onset, onion bulb formation, pes cavus, segmental peripheral demyelination/remyelination, and autosomal dominant inheritance [medquad_ans_0001092-1__dup1].

**Supported citations**: `medquad_ans_0001092-1__dup1`
**Unsupported citations**: _none_

### Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005321-1` (score=0.9994) - What are the signs and symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal arterial macroaneurysm...
2. `medquad_ans_0005876-1` (score=0.0002) - What are the signs and symptoms of STAR syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for STAR syndrome. If the information is available, the table below includes how often t...
3. `medquad_ans_0000870-1` (score=0.0001) - Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) is a disorder that affects blood vessels in the eyes and heart. The condition generally becomes apparent in infancy or childhood. RAMSVPS d...
4. `medquad_ans_0006001-2` (score=0.0001) - What are the signs and symptoms of Takayasu arteritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Takayasu arteritis. If the information is available, the table below includes h...
5. `medquad_ans_0005369-1` (score=0.0001) - What are the signs and symptoms of Rhizomelic syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Rhizomelic syndrome. If the information is available, the table below includes...

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis include exudative retinal detachment and pulmonic stenosis [medquad_ans_0005321-1].

**Supported citations**: `medquad_ans_0005321-1`
**Unsupported citations**: _none_

### Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0001838-1` (score=0.3892) - Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. Symptoms have varied greatly among the few reported cases. Dicarboxylic aminoaciduria is...
2. `medquad_ans_0001702-1` (score=0.0060) - D-2-alpha hydroxyglutaric aciduria is an inherited metabolic condition that is associated with progressive brain damage. Signs and symptoms of this condition include developmental delay, seizures, hypotonia, and abnor...
3. `medquad_ans_0001081-1` (score=0.0029) - 2-hydroxyglutaric aciduria is a condition that causes progressive damage to the brain. The major types of this disorder are called D-2-hydroxyglutaric aciduria (D-2-HGA), L-2-hydroxyglutaric aciduria (L-2-HGA), and co...
4. `medquad_ans_0003443-1` (score=0.0001) - L-2-hydroxyglutaric aciduria is an inherited metabolic condition that is associated with progressive brain damage. Signs and symptoms of this condition typically begin during infancy or early childhood and may include...
5. `medquad_ans_0003998-2` (score=0.0001) - What are the signs and symptoms of Mevalonic aciduria? The Human Phenotype Ontology provides the following list of signs and symptoms for Mevalonic aciduria. If the information is available, the table below includes h...

**Answer**:

Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. It is caused by mutations in the SLC1A1 gene and is inherited in an autosomal recessive fashion [medquad_ans_0001838-1].

**Supported citations**: `medquad_ans_0001838-1`
**Unsupported citations**: _none_

### Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Scores**: context_relevance=0.500, groundedness=0.102, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0001082-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000198-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000853-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000459-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention spondylothoracic dysostosis or its mode of inheritance. Therefore, it is not possible to determine based on the given information whether spondylothoracic dysostosis is inherited.

Answer: Insufficient context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.279, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000542-1` (score=0.9244) - Jervell and Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). This disorder is a form of long QT syndrome, which is a heart ...
2. `medquad_ans_0003296-1` (score=0.8756) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. I...
3. `medquad_ans_0000946-2` (score=0.0012) - This condition is rare; only a few affected individuals have been reported worldwide.
4. `medquad_ans_0000948-2` (score=0.0012) - This condition is rare; only a few affected individuals have been reported worldwide.
5. `medquad_ans_0000346-2` (score=0.0012) - This condition appears to be rare; only a few affected individuals have been reported worldwide.

**Answer**:

Jervell Lange-Nielsen syndrome is a rare condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). It is a form of long QT syndrome, which results in the heart muscle taking longer than usual to recharge between beats. This disorder increases the risk of fainting (syncope) and sudden death, beginning in early childhood. The syndrome is inherited in an autosomal recessive fashion and has two types: type 1 and type 2. Only a few affected individuals have been reported worldwide [medquad_ans_0000542-1, medquad_ans_0003296-1, medquad_ans_0000946-2, medquad_ans_0000948-2, medquad_ans_0000346-2].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000542-1, medquad_ans_0003296-1, medquad_ans_0000946-2, medquad_ans_0000948-2, medquad_ans_0000346-2`

### Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Scores**: context_relevance=0.167, groundedness=0.066, answer_relevance=0.633

**Top docs**:

1. `medquad_ans_0001071-2` (score=0.9995) - This condition is very rare; only a few dozen affected individuals have been identified.
2. `medquad_ans_0000650-2` (score=0.9995) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
3. `medquad_ans_0000904-2` (score=0.9995) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.
4. `medquad_ans_0000989-2` (score=0.9994) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000874-2` (score=0.9994) - This condition affects an estimated 1 in 8,500 females.

**Answer**:

The provided context does not specify the exact prevalence or number of people affected by congenital cataracts, facial dysmorphism, and neuropathy. While several sources provide estimates for rare conditions, none directly link to this specific combination of symptoms. Therefore, based on the given information, it is not possible to determine how many people are affected by congenital cataracts, facial dysmorphism, and neuropathy.

Answer: Insufficient context to determine the number of people affected.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Scores**: context_relevance=0.769, groundedness=0.961, answer_relevance=0.708

**Top docs**:

1. `medquad_ans_0000040-3` (score=0.9988) - Alpha thalassemia X-linked intellectual disability syndrome results from mutations in the ATRX gene. This gene provides instructions for making a protein that plays an essential role in normal development. Although th...
2. `medquad_ans_0001056-3` (score=0.0051) - X-linked intellectual disability, Siderius type is caused by mutations in the PHF8 gene. This gene provides instructions for making a protein that is found in the nucleus of cells, particularly in brain cells before a...
3. `medquad_ans_0001013-4` (score=0.0004) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000342-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000948-4` (score=0.0001) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

Mutations in the ATRX gene are responsible for alpha thalassemia X-linked intellectual disability syndrome. These mutations change the structure of the ATRX protein, which likely prevents it from effectively regulating gene expression. Reduced activity of the HBA1 and HBA2 genes, which are necessary for normal hemoglobin production, causes alpha thalassemia. Abnormal expression of other genes, whose identities have not been identified, probably leads to developmental delay, distinctive facial features, and other signs and symptoms of the syndrome [medquad_ans_0000040-3].

**Supported citations**: `medquad_ans_0000040-3`
**Unsupported citations**: _none_

### Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002622-2` (score=0.9994) - What are the signs and symptoms of Glycogen storage disease type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 6. If the information is available, ...
2. `medquad_ans_0003608-2` (score=0.0001) - What are the signs and symptoms of Limb-girdle muscular dystrophy type 2I? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy type 2I. If the information ...
3. `medquad_ans_0000563-2__dup1` (score=0.0001) - What are the signs and symptoms of Autosomal dominant neuronal ceroid lipofuscinosis 4B? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant neuronal ceroid lipofuscin...
4. `medquad_ans_0002617-2` (score=0.0001) - What are the signs and symptoms of Glycogen storage disease type 1B? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 1B. If the information is available...
5. `medquad_ans_0003607-1` (score=0.0001) - What are the signs and symptoms of Limb-girdle muscular dystrophy type 2H? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy type 2H. If the information ...

**Answer**:

The symptoms of Glycogen storage disease type 6 include hypoglycemia (90%), short stature (90%), hepatomegaly, increased hepatic glycogen content, and postnatal growth retardation [medquad_ans_0002622-2].

**Supported citations**: `medquad_ans_0002622-2`
**Unsupported citations**: _none_

### Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Scores**: context_relevance=1.000, groundedness=0.854, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000012-4__dup2` (score=0.9952) - The exact cause of atherosclerosis isn't known. However, certain traits, conditions, or habits may raise your risk for the disease. These conditions are known as risk factors. The more risk factors you have, the more ...
2. `medquad_ans_0000046-3__dup3` (score=0.9918) - The same risk factors that cause atherosclerosis may cause coronary microvascular disease. Atherosclerosis is a disease in which plaque builds up inside the arteries. Risk factors for atherosclerosis include: Diabetes...
3. `medquad_ans_0000027-3__dup3` (score=0.9900) - Type 2 diabetesthe most common form of diabetesis caused by a combination of factors, including insulin resistance, a condition in which the bodys muscle, fat, and liver cells do not use insulin effectively. Type 2 di...
4. `medquad_ans_0000070-3__dup4` (score=0.9900) - Type 2 diabetesthe most common form of diabetesis caused by a combination of factors, including insulin resistance, a condition in which the bodys muscle, fat, and liver cells do not use insulin effectively. Type 2 di...
5. `medquad_ans_0000037-3__dup3` (score=0.9900) - Type 2 diabetesthe most common form of diabetesis caused by a combination of factors, including insulin resistance, a condition in which the bodys muscle, fat, and liver cells do not use insulin effectively. Type 2 di...

**Answer**:

The causes of diabetes are not detailed in the provided context. The text mentions that diabetes is a disease in which the body's blood sugar level is too high because the body doesn't make enough insulin or doesn't use its insulin properly, but it does not explain the underlying causes of this condition [medquad_ans_0000012-4__dup2]. Therefore, based on the given information, a specific cause of diabetes cannot be identified.

**Supported citations**: `medquad_ans_0000012-4__dup2`
**Unsupported citations**: _none_

### Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Scores**: context_relevance=1.000, groundedness=0.937, answer_relevance=0.782

**Top docs**:

1. `medquad_ans_0000018-14` (score=0.9981) - For information about the clinical trial on this topic, visit: http://www.clinicaltrials.gov/ct/show/NCT00372320?order=1. If you would like to read an interview with Dr. Bruce Baum, the study's principal investigator,...
2. `medquad_ans_0000016-26` (score=0.9974) - National Eye Institute National Institutes of Health 2020 Vision Place Bethesda, MD 20892-3655 301-496-5248 E-mail: 2020@nei.nih.gov www.nei.nih.gov Find eye health organizations that address diabetic eye disease.
3. `medquad_ans_0000016-21` (score=0.9970) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the lab...
4. `medquad_ans_0000207-1__dup2` (score=0.9967) - Summary : Clinical trials are research studies that test how well new medical approaches work in people. Each study answers scientific questions and tries to find better ways to prevent, screen for, diagnose, or treat...
5. `medquad_ans_0000927-1__dup2` (score=0.9964) - Summary : It seems to happen almost every day - you hear about the results of a new medical research study. Sometimes the results of one study seem to disagree with the results of another study. It's important to be c...

**Answer**:

Research on diabetic retinopathy is being conducted by the National Eye Institute (NEI), which is conducting and supporting research to find better ways to detect, treat, and prevent vision loss in people with diabetes. This research includes laboratory studies and patient-based studies. For example, researchers are studying drugs for the treatment of proliferative retinopathy that may reduce the need for laser surgery [medquad_ans_0000016-21]. Additional information about clinical trials on this topic can be found at http://www.clinicaltrials.gov/ct/show/NCT00372320?order=1 [medquad_ans_0000018-14].

**Supported citations**: `medquad_ans_0000016-21`, `medquad_ans_0000018-14`
**Unsupported citations**: _none_

### Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Scores**: context_relevance=0.800, groundedness=0.925, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.9100) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart an...
2. `medquad_ans_0000379-1__dup1` (score=0.8206) - If you've ever gotten a splinter or had sand in your eye, you've had experience with a foreign body. A foreign body is something that is stuck inside you but isn't supposed to be there. You may inhale or swallow a for...
3. `medquad_ans_0000096-1__dup3` (score=0.5364) - Oxygen therapy is a treatment that provides you with extra oxygen, a gas that your body needs to work well. Normally, your lungs absorb oxygen from the air. However, some diseases and conditions can prevent you from g...
4. `medquad_ans_0000567-1__dup2` (score=0.1606) - When you breathe, your lungs take in oxygen from the air and deliver it to the bloodstream. The cells in your body need oxygen to work and grow. During a normal day, you breathe nearly 25,000 times. People with lung d...
5. `medquad_ans_0000676-1__dup2` (score=0.0673) - Oxygen therapy is a treatment that provides you with extra oxygen. Oxygen is a gas that your body needs to function. Normally, your lungs absorb oxygen from the air you breathe. But some conditions can prevent you fro...

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles in the air from fires or toxic fumes. These substances can damage the eyes and respiratory system and may worsen chronic heart and lung diseases. Symptoms may include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, the condition may worsen [medquad_ans_0000516-1__dup1].

**Supported citations**: `medquad_ans_0000516-1__dup1`
**Unsupported citations**: _none_

### Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Scores**: context_relevance=0.125, groundedness=0.000, answer_relevance=0.825

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0001101-2` (score=0.9995) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
3. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0000790-2` (score=0.9995) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
5. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.

**Answer**:

The provided context does not contain specific information about the prevalence or incidence of Zollinger-Ellison Syndrome. Therefore, it is not possible to determine how many people are affected by Zollinger-Ellison Syndrome based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Scores**: context_relevance=0.429, groundedness=0.853, answer_relevance=0.429

**Top docs**:

1. `medquad_ans_0000172-2__dup1` (score=0.9959) - Treatment is symptomatic and supportive, and may include drug therapy.
2. `medquad_ans_0000974-5` (score=0.9949) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Ge...
3. `medquad_ans_0000120-5` (score=0.0036) - These resources address the diagnosis or management of beta-ketothiolase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of acetyl-CoA acetyltransferase These resources from MedlinePlus offer in...
4. `medquad_ans_0000482-5` (score=0.0030) - These resources address the diagnosis or management of histidinemia: - Genetic Testing Registry: Histidinemia These resources from MedlinePlus offer information about the diagnosis and management of various health con...
5. `medquad_ans_0001089-5` (score=0.0030) - These resources address the diagnosis or management of HMG-CoA lyase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of hydroxymethylglutaryl-CoA lyase These resources from MedlinePlus offer inf...

**Answer**:

Treatment for tetrahydrobiopterin (BH4) deficiency is symptomatic and supportive, and may include drug therapy [medquad_ans_0000172-2__dup1]. Additionally, management may involve diagnostic tests, drug therapy, genetic counseling, and palliative care as outlined in resources from MedlinePlus [medquad_ans_0000974-5].

**Supported citations**: `medquad_ans_0000172-2__dup1`, `medquad_ans_0000974-5`
**Unsupported citations**: _none_

### Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Scores**: context_relevance=0.125, groundedness=0.650, answer_relevance=0.325

**Top docs**:

1. `medquad_ans_0001101-2` (score=0.9996) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
2. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
3. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
4. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
5. `medquad_ans_0000904-2` (score=0.9995) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.

**Answer**:

The prevalence of nonsyndromic holoprosencephaly is estimated to range from 1 in 7,500 to 20,000 people [medquad_ans_0001101-2]. Additionally, other estimates suggest it affects 1 in 40,000 to 50,000 people [medquad_ans_0000195-2], 1 in 50,000 people [medquad_ans_0000989-2], 1 in 87,000 people [medquad_ans_0000484-2], and approximately 1 in 35,000 to 50,000 newborns [medquad_ans_0000904-2]. These figures reflect varying estimates across different studies.

**Supported citations**: `medquad_ans_0001101-2`, `medquad_ans_0000195-2`, `medquad_ans_0000989-2`, `medquad_ans_0000484-2`, `medquad_ans_0000904-2`
**Unsupported citations**: _none_

### Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Scores**: context_relevance=0.636, groundedness=0.251, answer_relevance=0.582

**Top docs**:

1. `medquad_ans_0000525-3` (score=0.0013) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt ex...
2. `medquad_ans_0001013-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000342-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000948-4` (score=0.0002) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000799-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The provided context does not contain information about genetic changes related to progressive familial intrahepatic cholestasis. The context discusses genetic changes in the ABCB11 or ABCB4 genes in relation to intrahepatic cholestasis of pregnancy, not progressive familial intrahepatic cholestasis. Therefore, based on the available information, the answer cannot be determined.

Answer: Insufficient context to determine genetic changes related to progressive familial intrahepatic cholestasis.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.9994) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below in...
2. `medquad_ans_0005211-1` (score=0.0001) - What are the signs and symptoms of Pyknoachondrogenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyknoachondrogenesis. If the information is available, the table below includ...
3. `medquad_ans_0001626-1` (score=0.0001) - What are the signs and symptoms of Craniodiaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Craniodiaphyseal dysplasia. If the information is available, the table ...
4. `medquad_ans_0000773-1__dup1` (score=0.0001) - What are the signs and symptoms of Bone dysplasia lethal Holmgren type? The Human Phenotype Ontology provides the following list of signs and symptoms for Bone dysplasia lethal Holmgren type. If the information is ava...
5. `medquad_ans_0003813-2` (score=0.0001) - What are the signs and symptoms of Marshall syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Marshall syndrome. If the information is available, the table below includes how...

**Answer**:

The symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

### Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Scores**: context_relevance=0.429, groundedness=0.150, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000131-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000862-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000410-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000693-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether hepatic veno-occlusive disease with immunodeficiency is inherited. While the context repeatedly describes autosomal recessive inheritance patterns in general terms, it does not mention this specific condition. Therefore, based on the given information, the inheritance pattern of hepatic veno-occlusive disease with immunodeficiency cannot be determined.

Answer: Insufficient context to determine if hepatic veno-occlusive disease with immunodeficiency is inherited.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Scores**: context_relevance=1.000, groundedness=0.914, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005902-1` (score=0.9993) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how o...
2. `medquad_ans_0002781-2` (score=0.0001) - What are the signs and symptoms of Hemophagocytic lymphohistiocytosis? The signs and symptoms of hemophagocytic lymphohistiocytosis typically develop during the first months or years of life. However, in rare cases, a...
3. `medquad_ans_0000706-2__dup1` (score=0.0001) - What are the signs and symptoms of Beta-thalassemia? The Human Phenotype Ontology provides the following list of signs and symptoms for Beta-thalassemia. If the information is available, the table below includes how o...
4. `medquad_ans_0001972-1` (score=0.0001) - What are the signs and symptoms of Dyskeratosis congenita autosomal recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Dyskeratosis congenita autosomal recessive. If the info...
5. `medquad_ans_0003194-1` (score=0.0001) - What are the signs and symptoms of Infantile myofibromatosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Infantile myofibromatosis. If the information is available, the table be...

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. These signs and symptoms are based on the Human Phenotype Ontology, which compiles information from sources such as Orphanet [medquad_ans_0005902-1].

**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

### Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Scores**: context_relevance=0.833, groundedness=0.893, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000001-7__dup1` (score=0.9948) - Topics
2. `medquad_ans_0000159-7` (score=0.9732) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may n...
3. `medquad_ans_0000159-8` (score=0.9253) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed...
4. `medquad_ans_0000155-1__dup4` (score=0.0139) - If you have advanced chronic kidney disease (CKD), you may soon need treatment to do the work your kidneys can no longer do. Learning about your treatment options for kidney failure will help you make the best choice ...
5. `medquad_ans_0000159-4__dup2` (score=0.0088) - Lupus nephritis is diagnosed through urine and blood tests and a kidney biopsy: - Urinalysis. Urinalysis is testing of a urine sample. The urine sample is collected in a special container in a health care providers of...

**Answer**:

For lupus nephritis, treatment involves medications that suppress the immune system to prevent further damage to the kidneys. Standard treatment includes a corticosteroid, usually prednisone, to reduce inflammation in the kidneys, and an immunosuppressive medication such as cyclophosphamide or mycophenolate mofetil, typically used in combination with prednisone [medquad_ans_0000159-8]. If lupus nephritis is causing high blood pressure, medications to lower blood pressure may be needed, as this can significantly slow the progression of kidney disease [medquad_ans_0000159-8]. Additionally, people with kidney disease may need to discuss dietary changes with a healthcare provider, particularly regarding sodium intake, which may be reduced to manage blood pressure [medquad_ans_0000159-7]. While treatment can effectively control lupus nephritis in many

**Supported citations**: `medquad_ans_0000159-8`, `medquad_ans_0000159-7`
**Unsupported citations**: _none_

### Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000029-2__dup1` (score=0.9969) - Medications may ease certain symptoms, such as pain. Surgery is the only treatment available to correct functional disturbances or halt the progression of damage to the central nervous system. More than one surgery ma...
2. `medquad_ans_0000217-2` (score=0.9967) - Treatment involves eliminating or reducing exposure to the toxic substance, followed by symptomatic and supportive therapy.
3. `medquad_ans_0000238-2__dup1` (score=0.9966) - Treatment for individuals with PLS is symptomatic. Muscle relaxants such as baclofen, tizanidine, and the benzodiazepines may reduce spasticity. Other drugs may relieve pain and antidepressants can help treat depressi...
4. `medquad_ans_0000172-2__dup1` (score=0.9963) - Treatment is symptomatic and supportive, and may include drug therapy.
5. `medquad_ans_0000220-2__dup1` (score=0.9960) - There is no specific treatmentfor OPCA. Physicians may try different medications to treat the ataxia, tremor, and rigidity that are associated with the disorder. Other treatments are directed at specific symptoms. Sti...

**Answer**:

The provided context does not contain information about the treatments for Restless Legs Syndrome. Therefore, a specific answer cannot be given based on the available content. [doc_id: not available]

**Supported citations**: _none_
**Unsupported citations**: `doc_id: not available`

### Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Scores**: context_relevance=0.750, groundedness=0.991, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000052-2__dup1` (score=0.9801) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
2. `medquad_ans_0002574-1` (score=0.9311) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...
3. `medquad_ans_0000127-1__dup3` (score=0.8372) - A brain tumor is a growth of abnormal cells in the tissues of the brain. Brain tumors can be benign, with no cancer cells, or malignant, with cancer cells that grow quickly. Some are primary brain tumors, which start ...
4. `medquad_ans_0002573-1` (score=0.6410) - Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue th...
5. `medquad_ans_0000052-3__dup1` (score=0.1073) - Symptoms of brain and spinal cord tumors generally develop slowly and worsen over time unless they are treated. The tumor may be classified as benign or malignant and given a numbered score that reflects its rate of m...

**Answer**:

Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue that helps repair brain damage in response to injury. Glioblastomas are often very aggressive and grow into surrounding brain tissue. Signs and symptoms, such as headache, nausea, vomiting, and/or drowsiness, may develop when the tumor begins to put excess pressure on the brain [medquad_ans_0002573-1].

**Supported citations**: `medquad_ans_0002573-1`
**Unsupported citations**: _none_

### Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0005496-3` (score=0.9944) - What causes schizencephaly? The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, s...
2. `medquad_ans_0003881-3` (score=0.0013) - What causes megalencephaly, polymicrogyria, and hydrocephalus (MPPH) syndrome? The cause of MPPH syndrome is unknown. Infection during pregnancy or fetal accident is thought to be unlikely.
3. `medquad_ans_0000102-3__dup1` (score=0.0012) - The progression of the disorder is usually 10 years or longer.
4. `medquad_ans_0000251-4__dup1` (score=0.0004) - The NINDS conducts and supports a wide range of studies that explore the mechanisms of normal brain development. The knowledge gained from these fundamental studies provides the foundation for understanding how to pre...
5. `medquad_ans_0000650-2` (score=0.0002) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, schizencephaly can affect more than one family member, which supports a genetic cause in some cases. Schizencephaly has also been linked to a variety of non-genetic factors, including young maternal age and certain medications and infections that can cause vascular disruptions (disruption of blood flow or blood supply) in a developing baby [medquad_ans_0005496-3].

**Supported citations**: `medquad_ans_0005496-3`
**Unsupported citations**: _none_

### Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Scores**: context_relevance=0.700, groundedness=0.480, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0000025-3` (score=0.9993) - ALSP is caused by mutations in the CSF1R gene. This gene provides instructions for making a protein called colony stimulating factor 1 receptor (CSF-1 receptor), which is found in the outer membrane of certain types o...
2. `medquad_ans_0000052-3` (score=0.9983) - Mutations in several genes can cause familial ALS and contribute to the development of sporadic ALS. Mutations in the C9orf72 gene account for 30 to 40 percent of familial ALS in the United States and Europe. Worldwid...
3. `medquad_ans_0000013-9__dup1` (score=0.9980) - A heritable disease is caused by a mutation, or harmful change, in a gene inherited from a parent. Genes are small structures in your body's cells that determine how you look and tell your body how to work. Examples o...
4. `medquad_ans_0000551-3` (score=0.0068) - Mutations in the ALS2 gene cause most cases of juvenile primary lateral sclerosis. This gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but its function is not fully u...
5. `medquad_ans_0000521-3` (score=0.0023) - Infantile-onset ascending hereditary spastic paralysis is caused by mutations in the ALS2 gene. This gene provides instructions for making the alsin protein. Alsin is produced in a wide range of tissues, with highest ...

**Answer**:

Genetic changes related to amyotrophic lateral sclerosis (ALS) include mutations in several genes, such as C9orf72, SOD1, TARDBP, and FUS. Mutations in the C9orf72 gene account for 30 to 40 percent of familial ALS in the United States and Europe, while SOD1 gene mutations cause 15 to 20 percent of familial ALS worldwide, and TARDBP and FUS gene mutations each account for about 5 percent of cases. The other genes associated with familial ALS contribute to a small proportion of cases. It is estimated that 60 percent of individuals with familial ALS have an identified genetic mutation. The exact mechanisms by which these mutations lead to motor neuron death are unclear, but they are thought to disrupt axon development, impair axonal transport, or prevent the breakdown of toxic substances, leading to motor neuron damage and death [medquad*ans*

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.920, answer_relevance=0.571

**Top docs**:

1. `medquad_ans_0000718-5` (score=0.9972) - These resources address the diagnosis or management of NOMID: - Genetic Testing Registry: Chronic infantile neurological, cutaneous and articular syndrome These resources from MedlinePlus offer information about the d...
2. `medquad_ans_0000581-5` (score=0.9964) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Te...
3. `medquad_ans_0000582-5` (score=0.9950) - These resources address the diagnosis or management of Larsen syndrome: - Gene Review: Gene Review: FLNB-Related Disorders - Genetic Testing Registry: Larsen syndrome - Genetic Testing Registry: Larsen syndrome, domin...
4. `medquad_ans_0003473-6` (score=0.9531) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injectio...
5. `medquad_ans_0000073-5` (score=0.1646) - These resources address the diagnosis or management of Arts syndrome: - Gene Review: Gene Review: Arts Syndrome - Genetic Testing Registry: Arts syndrome - MedlinePlus Encyclopedia: Hearing Loss - MedlinePlus Encyclop...

**Answer**:

The only specific treatment available for Laron syndrome is subcutaneous injections of insulin-like growth factor 1 (IGF-1), a growth-promoting hormone. IGF-1 stimulates linear growth (height), improves brain growth, and addresses metabolic abnormalities caused by long-term IGF-1 deficiency. It also raises blood glucose levels, reduces cholesterol, and increases muscle growth. IGF-1 and growth hormone (GH) levels should be closely monitored during treatment because overdosage of IGF-1 can cause various health problems [medquad_ans_0003473-6].

**Supported citations**: `medquad_ans_0003473-6`
**Unsupported citations**: _none_

### Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Scores**: context_relevance=0.857, groundedness=0.937, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000096-2__dup1` (score=0.9962) - Treatment for SOD is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems associated with SOD are generally not treatable. Vision, physical, and occupational therapie...
2. `medquad_ans_0000016-14` (score=0.9947) - Research found that that prompt treatment of macular edema with anti-VEGF drugs, with or without laser treatment, resulted in better vision than laser treatment alone or steroid injections. When injected into the eye,...
3. `medquad_ans_0000001-22` (score=0.9922) - Once dry AMD reaches the advanced stage, no form of treatment can prevent vision loss. However, treatment can delay and possibly prevent intermediate AMD from progressing to the advanced stage, in which vision loss oc...
4. `medquad_ans_0000001-27` (score=0.9912) - Wet AMD can be treated with laser surgery, photodynamic therapy, and drugs that are injected into the eye. None of these treatments is a cure for wet AMD. The disease and loss of vision may progress despite treatment.
5. `medquad_ans_0000055-7` (score=0.9745) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need othe...

**Answer**:

Treatments for Diabetic Retinopathy include controlling blood glucose, blood pressure, cholesterol, and triglycerides. If retinopathy does not improve, additional treatments may be needed, such as medicines injected into the eye to block a protein causing abnormal blood vessel growth and fluid leakage, which can reduce fluid leakage and allow the retina to return to normal thickness. Focal laser treatment may also be used to slow fluid leakage by placing small laser burns in areas near the macula where fluid is leaking. In some cases, scatter laser treatment is used in multiple visits to place thousands of laser burns around weak, new blood vessels away from the macula, causing them to shrink [medquad_ans_0000055-7].

**Supported citations**: `medquad_ans_0000055-7`
**Unsupported citations**: _none_

### Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Scores**: context_relevance=0.571, groundedness=0.749, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000201-5` (score=0.9976) - These resources address the diagnosis or management of Cole disease: - Genetic Testing Registry: Cole disease These resources from MedlinePlus offer information about the diagnosis and management of various health con...
2. `medquad_ans_0000945-5` (score=0.9966) - These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliative Care
3. `medquad_ans_0000172-2__dup1` (score=0.9958) - Treatment is symptomatic and supportive, and may include drug therapy.
4. `medquad_ans_0000119-9` (score=0.9946) - Proctitis that is not treated or does not respond to treatment may lead to complications, including - severe bleeding and anemiaa condition in which red blood cells are fewer or smaller than normal, which means less o...
5. `medquad_ans_0000116-7` (score=0.9878) - Treatment depends on the severity of symptoms. The gastroenterologist will - review the medications the person is taking - make recommendations to change or stop certain medications - recommend that the person quit sm...

**Answer**:

The provided context does not contain specific information about the treatments for collagenous colitis. While it mentions treatments for microscopic colitis, including drug therapy, antidiarrheal medications, corticosteroids, anti-inflammatory medications, cholestyramine, antibiotics, immunomodulators, and anti-TNF therapies, there is no direct information linking these treatments to collagenous colitis [medquad_ans_0000172-2\_\_dup1][medquad_ans_0000116-7]. Therefore, based on the given context, the treatments for collagenous colitis cannot be determined.

**Supported citations**: `medquad_ans_0000172-2__dup1`, `medquad_ans_0000116-7`
**Unsupported citations**: _none_

### Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Scores**: context_relevance=0.444, groundedness=0.064, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000432-7` (score=0.9888) - The best way to prevent whipworm infection is to always: - Avoid ingesting soil that may be contaminated with human feces, including where human fecal matter ("night soil") or wastewater is used to fertilize crops. - ...
2. `medquad_ans_0000344-7` (score=0.9848) - Baylisascaris infection can be prevented by avoiding contact with raccoons and their feces. Washing your hands after working or playing outdoors is good practice for preventing a number of diseases. Do not keep, feed,...
3. `medquad_ans_0000265-10` (score=0.9725) - There are no programs to control or eliminate loiasis in affected areas. Your risk of infection may be less in areas where communities receive regular treatment for onchocerciasis or lymphatic filariasis. There are no...
4. `medquad_ans_0000241-7` (score=0.9408) - The best way to avoid hookworm infection is not to walk barefoot in areas where hookworm is common and where there may be human fecal contamination of the soil. Also, avoid other skin contact with such soil and avoid ...
5. `medquad_ans_0000440-7` (score=0.9177) - Wearing shoes and taking other protective measures to avoid skin contact with sand or soil will prevent infection with zoonotic hookworms. Travelers to tropical and subtropical climates, especially where beach exposur...

**Answer**:

The provided context does not contain information on how to prevent Trichinellosis (also known as Trichinosis). Therefore, no answer can be given based on the available information. [context_missing]

**Supported citations**: _none_
**Unsupported citations**: `context_missing`

### Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Scores**: context_relevance=1.000, groundedness=0.949, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000182-3__dup1` (score=0.9962) - A combination of autoimmunity, environmental triggers, and a genetic predisposition can lead to autoimmune hepatitis.
2. `medquad_ans_0000536-3` (score=0.9939) - What causes autoimmune hepatitis? Although the exact cause of autoimmune hepatitis is unknown, evidence suggests that liver injury in a patient with autoimmune hepatitis is the result of a cell-mediated immunologic at...
3. `medquad_ans_0000079-3__dup2` (score=0.0058) - In most cases, an autoimmune response is thought to cause immune thrombocytopenia (ITP). Normally, your immune system helps your body fight off infections and diseases. In ITP, however, your immune system attacks and ...
4. `medquad_ans_0000015-16` (score=0.0042) - Type 1 diabetes is an autoimmune disease. In an autoimmune reaction, antibodies, or immune cells, attach to the body's own healthy tissues by mistake, signaling the body to attack them. At present, scientists do not k...
5. `medquad_ans_0000535-3` (score=0.0029) - What causes autoimmune hemolytic anemia? In about half of cases, the cause of autoimmune hemolytic anemia cannot be determined (idiopathic or primary). This condition can also be caused by or occur with another disord...

**Answer**:

Autoimmune hepatitis is caused by a combination of autoimmunity, environmental triggers, and a genetic predisposition. The liver injury in autoimmune hepatitis results from a cell-mediated immunologic attack, which may be triggered by genetic factors, viral infections, or chemical agents. Autoimmune hepatitis sometimes occurs in relatives of people with autoimmune diseases, further suggesting a genetic cause [medquad_ans_0000182-3\_\_dup1][medquad_ans_0000536-3].

**Supported citations**: `medquad_ans_0000182-3__dup1`, `medquad_ans_0000536-3`
**Unsupported citations**: _none_

### Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Scores**: context_relevance=0.800, groundedness=0.948, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000006_3-1` (score=0.9580) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
2. `medquad_ans_0002574-1` (score=0.8300) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...
3. `medquad_ans_0000186-1__dup2` (score=0.0463) - Brain tumors are abnormal growths inside the skull. They are among the most common types of childhood cancers. Some are benign tumors, which aren't cancer. They can still be serious. Malignant tumors are cancerous. Ch...
4. `medquad_ans_0000144-1__dup2` (score=0.0108) - Cancer begins in the cells, which are the building blocks of your body. Normally, new cells form as you need them, replacing old cells that die. Sometimes, this process goes wrong. New cells form when you don't need t...
5. `medquad_ans_0000006_6-5` (score=0.0034) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on: - The type of tumor and where it is in the brain. - Whether the cancer h...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They are tumors that start in star-shaped brain cells called astrocytes, which are a type of glial cell that holds nerve cells in place, brings food and oxygen to them, and helps protect them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

### Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Scores**: context_relevance=0.857, groundedness=0.975, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000013_3-1__dup3` (score=0.8728) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia in...
2. `medquad_ans_0000040-10` (score=0.0081) - Acute leukemia gets worse quickly. In chronic leukemia, symptoms develop gradually and are generally not as severe as in acute leukemia.
3. `medquad_ans_0000040-15` (score=0.0053) - There are no standard or over-the-counter tests for leukemia. Your doctor can request lab analyses for leukemia that include blood tests that check the levels and types of blood cells and look for changes in the shape...
4. `medquad_ans_0000013_3-2__dup1` (score=0.0009) - Signs and symptoms of myelodysplastic/myeloproliferative neoplasm, unclassifiable, include fever, weight loss, and feeling very tired. These and other signs and symptoms may be caused by MDS/MPN-UC or by other conditi...
5. `medquad_ans_0000013_3-3__dup3` (score=0.0008) - Certain factors affect prognosis (chance of recovery). The prognosis (chance of recovery) for atypical CML depends on the number of red blood cells and platelets in the blood.

**Answer**:

Atypical chronic myelogenous leukemia (CML) is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. The body tells too many blood stem cells to become granulocytes, and some of these stem cells never become mature white blood cells; these immature cells are called blasts. Over time, the granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. Leukemia cells in atypical CML and CML look alike under a microscope, but in atypical CML, the "Philadelphia chromosome" chromosome change is not present [medquad_ans_0000013_3-1__dup3]. Signs and symptoms include easy bruising or bleeding and feeling tired and weak [medquad_ans_0000013_3-1__dup3]. The prognosis for atypical CML depends on the number

**Supported citations**: `medquad_ans_0000013_3-1__dup3`
**Unsupported citations**: _none_

### Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Scores**: context_relevance=0.750, groundedness=0.948, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000996-3` (score=0.9960) - Most cases of trisomy 13 result from having three copies of chromosome 13 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the cha...
2. `medquad_ans_0000823-4` (score=0.9844) - This condition is generally not inherited but arises from gene mutations that occur in early blood-forming cells after conception. These alterations are called somatic mutations.
3. `medquad_ans_0000373-3` (score=0.0800) - Researchers have identified changes in five regions of the X chromosome that are linked to FG syndrome in affected families. Mutations in a gene called MED12, which is located in one of these regions, appear to be the...
4. `medquad_ans_0000398-1__dup1` (score=0.0397) - Genes are the building blocks of heredity. They are passed from parent to child. They hold DNA, the instructions for making proteins. Proteins do most of the work in cells. They move molecules from one place to anothe...
5. `medquad_ans_0000253-3` (score=0.0071) - In most cases, the cause of CCHD is unknown. A variety of genetic and environmental factors likely contribute to this complex condition. Changes in single genes have been associated with CCHD. Studies suggest that the...

**Answer**:

The genetic changes related to trisomy 13 include having three copies of chromosome 13 in each cell instead of the usual two copies, which disrupts normal development and causes characteristic features of the condition [medquad_ans_0000996-3]. In some cases, trisomy 13 occurs when a part of chromosome 13 becomes attached (translocated) to another chromosome during the formation of reproductive cells or very early in fetal development, resulting in individuals having two normal copies of chromosome 13 and an extra copy attached to another chromosome [medquad_ans_0000996-3]. Rarely, only part of chromosome 13 is present in three copies, which may lead to different physical signs and symptoms compared to full trisomy 13 [medquad_ans_0000996-3]. In a small percentage of cases, the extra copy

**Supported citations**: `medquad_ans_0000996-3`
**Unsupported citations**: _none_

### Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Scores**: context_relevance=0.571, groundedness=0.751, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000217-2` (score=0.9965) - Treatment involves eliminating or reducing exposure to the toxic substance, followed by symptomatic and supportive therapy.
2. `medquad_ans_0000172-2__dup1` (score=0.9961) - Treatment is symptomatic and supportive, and may include drug therapy.
3. `medquad_ans_0000054-2__dup1` (score=0.9960) - Generally treatment for individuals with BSS focuses on the underlying cause of the disorder. Early treatment with high-dose steroids may be beneficial in many cases. Other treatment is symptomatic and supportive.
4. `medquad_ans_0000239-3__dup1` (score=0.9906) - There is currently no treatment that can halt progression of any of the TSEs. Treatment is aimed at alleviating symptoms and making the patient as comfortable as possible. A clinical trial of a potential therapy for C...
5. `medquad_ans_0000731-5` (score=0.9866) - These resources address the diagnosis or management of Niemann-Pick disease: - Baby's First Test - Gene Review: Gene Review: Acid Sphingomyelinase Deficiency - Gene Review: Gene Review: Niemann-Pick Disease Type C - G...

**Answer**:

The provided context does not specify the exact treatments for Niemann-Pick disease. It mentions resources that address the diagnosis and management of Niemann-Pick disease, including genetic testing, diagnostic tests, drug therapy, genetic counseling, and palliative care [medquad_ans_0000731-5], but does not detail specific treatments. Therefore, based on the given information, the treatments for Niemann-Pick disease cannot be determined.

**Supported citations**: `medquad_ans_0000731-5`
**Unsupported citations**: _none_

### Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.533, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001714-1` (score=0.9994) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial poly...
2. `medquad_ans_0002480-2` (score=0.0001) - What are the signs and symptoms of Galloway-Mowat syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Galloway-Mowat syndrome. If the information is available, the table below ...
3. `medquad_ans_0005958-1` (score=0.0001) - What are the signs and symptoms of Symphalangism with multiple anomalies of hands and feet? The Human Phenotype Ontology provides the following list of signs and symptoms for Symphalangism with multiple anomalies of h...
4. `medquad_ans_0006103-1` (score=0.0001) - What are the signs and symptoms of Tibia absent polydactyly arachnoid cyst? The Human Phenotype Ontology provides the following list of signs and symptoms for Tibia absent polydactyly arachnoid cyst. If the informatio...
5. `medquad_ans_0000902-1__dup1` (score=0.0001) - What are the signs and symptoms of Camptobrachydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Camptobrachydactyly. If the information is available, the table below includes...

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, autosomal recessive inheritance, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, wide nasal bridge [med

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.880, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003789-3` (score=0.9941) - What causes Mantle cell lymphoma? Most lymphomas are not inherited, but rather an acquired disease in which the DNAwithin the cells has been damaged. Damage to the DNA occurs by a combination of different factors. Man...
2. `medquad_ans_0002483-3` (score=0.4833) - What causes gamma heavy chain disease? The causes or risk factors for gamma heavy chain disease are not known.
3. `medquad_ans_0000102-3__dup1` (score=0.0079) - The progression of the disorder is usually 10 years or longer.
4. `medquad_ans_0004199-3` (score=0.0046) - What causes multiple myeloma? Although the exact underlying cause of multiple myeloma is poorly understood, the specific symptoms of the condition result from abnormal and excessive growth of plasma cells in the bone ...
5. `medquad_ans_0000650-2` (score=0.0006) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Mantle cell lymphoma is caused by DNA damage within cells, which can result from a combination of factors. Many cases are associated with a chromosomal translocation. Additionally, viral infections such as Epstein-Barr virus, HIV, and human herpesvirus 6 have been linked to non-Hodgkin lymphomas, and immunodeficiencies and environmental factors like hair dyes and pesticides may also contribute to the development of NHLs, including mantle cell lymphoma [medquad_ans_0003789-3].

**Supported citations**: `medquad_ans_0003789-3`
**Unsupported citations**: _none_

### Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000487-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000574-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000484-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000495-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Legg-Calvé-Perthes disease or its inheritance pattern. Therefore, based on the given information, it is not possible to determine whether Legg-Calvé-Perthes disease is inherited.

Answer: Insufficient context [doc_id].

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

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
2. `medquad_ans_0003209-1` (score=0.0001) - What are the signs and symptoms of Infundibulopelvic dysgenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Infundibulopelvic dysgenesis. If the information is available, the ta...
3. `medquad_ans_0003551-1` (score=0.0001) - What are the signs and symptoms of Lethal chondrodysplasia Moerman type? The Human Phenotype Ontology provides the following list of signs and symptoms for Lethal chondrodysplasia Moerman type. If the information is a...
4. `medquad_ans_0003780-1` (score=0.0001) - What are the signs and symptoms of Mandibuloacral dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Mandibuloacral dysplasia. If the information is available, the table belo...
5. `medquad_ans_0004611-2` (score=0.0001) - What are the signs and symptoms of Osteogenesis imperfecta? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteogenesis imperfecta. If the information is available, the table below ...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies (90%), elbow dislocation (90%), hypoplasia of the zygomatic bone (90%), scoliosis (90%), omphalocele (7.5%), hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, tongue-like lumbar vertebral deformities [medquad_ans_0005135-1].

**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

### Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.985, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001664-1` (score=0.8285) - Cri du chat syndrome, also known as 5p- (5p minus) syndrome or cat cry syndrome, is a genetic condition that is caused by the deletion of genetic material on the small arm (the p arm) of chromosome 5. Infants with thi...
2. `medquad_ans_0000251-1` (score=0.8085) - Cri-du-chat (cat's cry) syndrome, also known as 5p- (5p minus) syndrome, is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that ...
3. `medquad_ans_0000219-1` (score=0.4092) - Congenital central hypoventilation syndrome (CCHS) is a disorder that affects breathing. People with this disorder take shallow breaths (hypoventilate), especially during sleep, resulting in a shortage of oxygen and a...
4. `medquad_ans_0001488-1` (score=0.0307) - Congenital central hypoventilation syndrome (CCHS) is a disorder of the autonomic nervous system that affects breathing. It causes a person to hypoventilate (especially during sleep), resulting in a shortage of oxygen...
5. `medquad_ans_0000073-1__dup2` (score=0.0056) - Cerebro-oculo-facio-skeletal syndrome (COFS) is a pediatric, genetic, degenerative disorder that involves the brain and the spinal cord. It is characterized by craniofacial and skeletal abnormalities, severely reduced...

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

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003414-3` (score=0.9965) - What causes Konigsmark Knox Hussels syndrome? Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid ar...
2. `medquad_ans_0000171-3__dup1` (score=0.0102) - KTS is often a progressive disorder, and complications may be life-threatening. However, many individuals can live well while managing their symptoms.
3. `medquad_ans_0001071-2` (score=0.0004) - This condition is very rare; only a few dozen affected individuals have been identified.
4. `medquad_ans_0000621-2` (score=0.0004) - This condition is very rare; fewer than 30 cases have been reported.
5. `medquad_ans_0000087-3__dup1` (score=0.0003) - The prognosis is dependent upon the specific syndrome, however, some of the syndromes are fatal if left untreated.

**Answer**:

Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene, specifically a mutation that replaces the amino acid arginine with the amino acid histidine at position 445 in the OPA1 protein, written as Arg445His or R445H [medquad_ans_0003414-3].

**Supported citations**: `medquad_ans_0003414-3`
**Unsupported citations**: _none_

### Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.500, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004584-2` (score=0.9992) - What are the signs and symptoms of Orofaciodigital syndrome 1? Oral features of OFD1 may include a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormali...
2. `medquad_ans_0000886-2__dup1` (score=0.0001) - What are the signs and symptoms of C syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for C syndrome. If the information is available, the table below includes how often the sym...
3. `medquad_ans_0004589-2` (score=0.0001) - What are the signs and symptoms of Orofaciodigital syndrome 2? Although the signs and symptoms that occur in people with orofaciodigital syndrome type 2 may vary, the following findings may be present:Facial findings ...
4. `medquad_ans_0006175-1` (score=0.0001) - What are the signs and symptoms of Tricho-dento-osseous syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Tricho-dento-osseous syndrome 1. If the information is available, ...
5. `medquad_ans_0004592-1` (score=0.0001) - What are the signs and symptoms of Orofaciodigital syndrome 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 5. If the information is available, the table ...

**Answer**:

The symptoms of Orofaciodigital syndrome 1 include oral features such as a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormalities. Facial features may include hypertelorism (increased width between the eyes), a small nose, micrognathia (small jaw), and other facial abnormalities. Finger and toe abnormalities may include shortness (brachydactyly), webbing or joining (syndactyly), abnormally curved fingers (clinodactyly), or other malformations. Brain abnormalities such as cysts and kidney problems like polycystic kidney disease may also occur. About half of individuals have some degree of learning disability, usually mild. Additionally, symptoms include bifid tongue (90%), broad alveolar ridges (90%), cleft upper lip (90%), frontal bossing (90%), hypert

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Scores**: context_relevance=0.833, groundedness=0.817, answer_relevance=0.567

**Top docs**:

1. `medquad_ans_0000190-1__dup2` (score=0.9979) - Summary : Kids, this page is for you. Learn about everything from how the body works to what happens when you go to the hospital. There are quizzes, games and lots of cool web sites for you to explore. Have fun!
2. `medquad_ans_0000863-1__dup1` (score=0.9972) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For exa...
3. `medquad_ans_0000499-1__dup1` (score=0.9965) - Summary : A hysterectomy is surgery to remove a woman's uterus or womb. The uterus is the place where a baby grows when a woman is pregnant. After a hysterectomy, you no longer have menstrual periods and can't become ...
4. `medquad_ans_0000414-1__dup2` (score=0.9936) - Frequently Asked Questions (FAQs) Fact Sheets
5. `medquad_ans_0000001-7__dup1` (score=0.9905) - Topics

**Answer**:

Yes, there is information about surgery. Surgery can be performed for various reasons, such as relieving or preventing pain, reducing symptoms, improving body function, or finding a problem like through a biopsy. Some surgeries, such as heart surgery, can save a life. Laparoscopic surgery, which uses small incisions and a thin tube with a camera, is now commonly used instead of large incisions. However, there are risks after surgery, including infection, excessive bleeding, reaction to anesthesia, or accidental injury. There is also almost always some pain following surgery [medquad_ans_0000863-1__dup1]. Additionally, a hysterectomy is a specific type of surgery to remove a woman's uterus, which can result in the end of menstrual periods and the inability to become pregnant [medquad_ans_0000499-1__dup1].

**Supported citations**: `medquad_ans_0000863-1__dup1`, `medquad_ans_0000499-1__dup1`
**Unsupported citations**: _none_

### Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Scores**: context_relevance=0.875, groundedness=0.977, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0002935-1` (score=0.9831) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affe...
2. `medquad_ans_0000040-18` (score=0.0036) - Treatment depends on a number of factors, including the type of leukemia, the patient's age and general health, where leukemia cells have collected in the body, and whether the leukemia has been treated before. Certai...
3. `medquad_ans_0000040-23` (score=0.0027) - Clinical trials are research studies in which new treatments -- drugs, diagnostics, procedures, vaccines, and other therapies -- are tested in people to see if they are safe, effective, and better than the current sta...
4. `medquad_ans_0001053-5` (score=0.0022) - These resources address the diagnosis or management of XMEN: - MedlinePlus Encyclopedia: Epstein-Barr Virus Test - MedlinePlus Encyclopedia: T Cell Count These resources from MedlinePlus offer information about the di...
5. `medquad_ans_0002934-1` (score=0.0011) - Human T-cell leukemia virus, type 1 (HTLV-1) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, some affected people may later de...

**Answer**:

Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affects the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affected people may later develop neurological problems and/or chronic lung infections. HTLV-2 is spread by blood transfusions, sexual contact, and sharing needles. It can also be spread from mother to child during birth or breastfeeding. There is no cure or treatment for HTLV-2, and it is considered a lifelong condition; however, most infected people remain asymptomatic (show no symptoms) throughout life [medquad_ans_0002935-1].

**Supported citations**: `medquad_ans_0002935-1`
**Unsupported citations**: _none_

### Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Scores**: context_relevance=1.000, groundedness=0.543, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000522-1__dup1` (score=0.5592) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...
2. `medquad_ans_0001152-1` (score=0.0115) - Children's interstitial and diffuse lung disease (chILD) is not a single condition, but a group of rare lung diseases found in infants, children and adolescents that can range from mild to severe. All types of chILD d...
3. `medquad_ans_0000758-1__dup2` (score=0.0098) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
4. `medquad_ans_0000009-1__dup2` (score=0.0084) - Asbestos-related lung diseases are diseases caused by exposure to asbestos (as-BES-tos) fibers. Asbestos is a mineral that, in the past, was widely used in many industries. Asbestos is made up of tiny fibers that can ...
5. `medquad_ans_0000078-1__dup2` (score=0.0068) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...

**Answer**:

Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing in dust or other particles in the air is responsible for some types of interstitial lung diseases. Specific types include black lung disease among coal miners from inhaling coal dust, farmer's lung from inhaling farm dust, asbestosis from inhaling asbestos fibers, siderosis from inhaling iron from mines or welding fumes, and silicosis from inhaling silica dust. Other causes include autoimmune diseases or occupational exposures to molds, gases, or fumes. Some types of interstitial lung disease have no known cause. Treatment depends on the type of exposure and the stage of the disease and may involve medicines, oxygen therapy, or a lung transplant in severe cases [medquad_ans_0000522-1\_\_dup

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000960-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000657-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0001038-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000440-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Melnick-Needles syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Melnick-Needles syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Scores**: context_relevance=0.750, groundedness=0.538, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000908-1__dup2` (score=0.9954) - Your trachea, or windpipe, is one part of your airway system. Airways are pipes that carry oxygen-rich air to your lungs. They also carry carbon dioxide, a waste gas, out of your lungs. When you inhale, air travels fr...
2. `medquad_ans_0000133-1__dup2` (score=0.9724) - The bronchi are two tubes that branch off the trachea, or windpipe. The bronchi carry air to your lungs. The most common problem with the bronchi is bronchitis, an inflammation of the tubes. Bronchitis can be acute or...
3. `medquad_ans_0006138-1` (score=0.9703) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of ...
4. `medquad_ans_0003482-1` (score=0.0528) - Laryngomalacia is an abnormality of the cartilage of the voice box (larynx) that is present at birth. The condition is characterized by "floppy" cartilage collapsing over the larynx when air is drawn into the lungs (i...
5. `medquad_ans_0000171-3__dup1` (score=0.0207) - KTS is often a progressive disorder, and complications may be life-threatening. However, many individuals can live well while managing their symptoms.

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse. There are two forms: a congenital form (called primary TBM) that typically develops during infancy or early childhood, and an acquired form (called secondary TBM) usually seen in adults. Some affected individuals may initially have no signs or symptoms, but the condition is typically progressive and most people eventually develop symptoms such as shortness of breath, cough, sputum retention, and wheezing or stridor with breathing. Most cases of primary TBM are caused by genetic conditions that weaken the airway walls, while secondary TBM often results from trauma, chronic inflammation, or prolonged compression of the airways. Treatment is generally only required in those with signs and symptoms and may include stenting, surgical correction,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Scores**: context_relevance=0.600, groundedness=0.993, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000880-1__dup2` (score=0.9909) - The temporomandibular joint (TMJ) connects your jaw to the side of your head. When it works well, it enables you to talk, chew, and yawn. For people with TMJ dysfunction, problems with the joint and muscles around it ...
2. `medquad_ans_0000001-7__dup1` (score=0.9636) - Topics
3. `medquad_ans_0000531-1__dup2` (score=0.8110) - Your jaw is a set of bones that holds your teeth. It consists of two main parts. The upper part is the maxilla. It doesn't move. The moveable lower part is called the mandible. You move it when you talk or chew. The t...
4. `medquad_ans_0006030-1` (score=0.4411) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people m...
5. `medquad_ans_0000874-2` (score=0.0061) - This condition affects an estimated 1 in 8,500 females.

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

### Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0001090-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000495-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000721-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000053-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Kallmann syndrome or its mode of inheritance. Therefore, it is not possible to determine from the given information whether Kallmann syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Scores**: context_relevance=1.000, groundedness=0.895, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005967-2` (score=0.9994) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how...
2. `medquad_ans_0005970-1` (score=0.0001) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how...
3. `medquad_ans_0000813-1__dup1` (score=0.0001) - What are the signs and symptoms of Brachydactyly type A5? The Human Phenotype Ontology provides the following list of signs and symptoms for Brachydactyly type A5. If the information is available, the table below incl...
4. `medquad_ans_0004592-1` (score=0.0001) - What are the signs and symptoms of Orofaciodigital syndrome 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 5. If the information is available, the table ...
5. `medquad_ans_0006572-2` (score=0.0001) - What are the signs and symptoms of 22q13.3 deletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 22q13.3 deletion syndrome. If the information is available, the table be...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and autosomal dominant inheritance [medquad_ans_0005967-2].

**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

### Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Scores**: context_relevance=0.800, groundedness=0.894, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000012-7__dup1` (score=0.9730) - Chronic obstructive pulmonary disease, or COPD, is a progressive lung disease in which the airways of the lungs become damaged, making it harder to breathe. With COPD, airways become blocked, making it harder to get a...
2. `medquad_ans_0000232-1__dup2` (score=0.8652) - COPD (chronic obstructive pulmonary disease) makes it hard for you to breathe. The two main types are chronic bronchitis and emphysema. The main cause of COPD is long-term exposure to substances that irritate and dama...
3. `medquad_ans_0000133-1__dup2` (score=0.7908) - The bronchi are two tubes that branch off the trachea, or windpipe. The bronchi carry air to your lungs. The most common problem with the bronchi is bronchitis, an inflammation of the tubes. Bronchitis can be acute or...
4. `medquad_ans_0000012-1__dup5` (score=0.4235) - Chronic obstructive pulmonary disease, or COPD, is a progressive lung disease in which the airways of the lungs become damaged, making it hard to breathe. You may also have heard COPD called other names, like emphysem...
5. `medquad_ans_0000197-1__dup3` (score=0.1864) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...

**Answer**:

Chronic bronchitis is one of the two main types of chronic obstructive pulmonary disease (COPD). It is characterized by inflammation of the bronchi, the tubes that carry air to the lungs, and is typically marked by a persistent cough that produces a lot of mucus. It is caused by long-term exposure to irritants such as cigarette smoke, air pollution, chemical fumes, or dust. Chronic bronchitis is part of a broader condition known as COPD, which also includes emphysema and is associated with progressive damage to the airways, making it harder to breathe [medquad_ans_0000232-1__dup2].

**Supported citations**: `medquad_ans_0000232-1__dup2`
**Unsupported citations**: _none_

### Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Scores**: context_relevance=0.750, groundedness=0.817, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000945-5` (score=0.9974) - These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliative Care
2. `medquad_ans_0000937-5` (score=0.9966) - These resources address the diagnosis or management of SMARD1: - Genetic Testing Registry: Spinal muscular atrophy with respiratory distress 1 These resources from MedlinePlus offer information about the diagnosis and...
3. `medquad_ans_0000696-5` (score=0.9964) - These resources address the diagnosis or management of multiple system atrophy: - Genetic Testing Registry: Shy-Drager syndrome - Vanderbilt Autonomic Dysfunction Center These resources from MedlinePlus offer informat...
4. `medquad_ans_0000173-2__dup2` (score=0.9964) - There is no cure for SMA. Treatment consists of managing the symptoms and preventing complications.
5. `medquad_ans_0000159-3__dup1` (score=0.9963) - IBM is generally resistant to all therapies and its rate of progression appears to be unaffected by currently available treatments.

**Answer**:

There is no cure for Spinal Muscular Atrophy (SMA). Treatment consists of managing the symptoms and preventing complications [medquad_ans_0000173-2__dup2]. Specific management may include diagnostic tests, drug therapy, genetic counseling, and palliative care, as these are resources available for the diagnosis and management of various health conditions, including SMA [medquad_ans_0000945-5]. However, the provided context does not specify particular drug therapies or interventions for SMA beyond symptom management.

**Supported citations**: `medquad_ans_0000173-2__dup2`, `medquad_ans_0000945-5`
**Unsupported citations**: _none_

### Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Scores**: context_relevance=0.667, groundedness=0.780, answer_relevance=0.756

**Top docs**:

1. `medquad_ans_0000267-5` (score=0.9944) - These resources address the diagnosis or management of cytogenetically normal acute myeloid leukemia: - Fred Hutchinson Cancer Research Center - National Cancer Institute: Acute Myeloid Leukemia Treatment - St. Jude C...
2. `medquad_ans_0000040-18` (score=0.9944) - Treatment depends on a number of factors, including the type of leukemia, the patient's age and general health, where leukemia cells have collected in the body, and whether the leukemia has been treated before. Certai...
3. `medquad_ans_0000001_6-5` (score=0.9902) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000001_3-6` (score=0.9887) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000001_7-7` (score=0.9868) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not contain information about treatments for Adult Acute Lymphoblastic Leukemia. While the context discusses treatments for cytogenetically normal acute myeloid leukemia and mentions clinical trials involving CAR T-cell therapy for childhood ALL that has relapsed, there is no specific information regarding Adult Acute Lymphoblastic Leukemia. [medquad_ans_0000267-5] [medquad_ans_0000040-18] [medquad_ans_0000001_6-5]

**Supported citations**: `medquad_ans_0000267-5`, `medquad_ans_0000040-18`, `medquad_ans_0000001_6-5`
**Unsupported citations**: _none_

### Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Scores**: context_relevance=1.000, groundedness=0.991, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002439-3` (score=0.9899) - What causes Friedreich ataxia? Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known...
2. `medquad_ans_0000133-3__dup1` (score=0.0002) - Generally, within 15 to 20 years after the appearance of the first symptoms, the person is confined to a wheelchair, and in later stages of the disease, individuals may become completely incapacitated. Friedreich's at...
3. `medquad_ans_0000032-1__dup1` (score=0.0001) - Ataxia often occurs when parts of the nervous system that control movement are damaged. People with ataxia experience a failure of muscle control in their arms and legs, resulting in a lack of balance and coordination...
4. `medquad_ans_0000388-3` (score=0.0001) - Mutations in the FXN gene cause Friedreich ataxia. This gene provides instructions for making a protein called frataxin. Although its role is not fully understood, frataxin appears to be important for the normal funct...
5. `medquad_ans_0000941-1` (score=0.0001) - Spinocerebellar ataxia type 36 (SCA36) is a condition characterized by progressive problems with movement that typically begin in mid-adulthood. People with this condition initially experience problems with coordinati...

**Answer**:

Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known as a GAA trinucleotide repeat. Normally, this segment is repeated 5 to 33 times within the FXN gene. In people with Friedreich ataxia, the GAA segment is repeated 66 to more than 1,000 times. The abnormally long GAA trinucleotide repeat disrupts the production of frataxin, which severely reduces the amount of this protein in cells. Certain nerve and muscle cells cannot function properly with a shortage of frataxin, leading to the characteristic signs and symptoms of Friedreich ataxia [medquad_ans_0002439-3].

**Supported citations**: `medquad_ans_0002439-3`
**Unsupported citations**: _none_

### Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Scores**: context_relevance=0.667, groundedness=0.705, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000352-1__dup2` (score=0.9965) - Your family history includes health information about you and your close relatives. Families have many factors in common, including their genes, environment, and lifestyle. Looking at these factors can help you figure...
2. `medquad_ans_0000169-3` (score=0.9945) - Episodes eventually decrease in frequency and intensity over the course of eight to 12 years.
3. `medquad_ans_0000962-4` (score=0.9878) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. However, some people who inherit the altered gene never develop ...
4. `medquad_ans_0000034-1__dup2` (score=0.9724) - Atrial fibrillation (AF) describes the rapid, irregular beating of the left atrium (upper chamber) of the heart. These rapid contractions of the heart are weaker than normal contractions, resulting in slow flow of blo...
5. `medquad_ans_0000706-2` (score=0.9224) - This condition has been reported in several families of northern Swedish ancestry.

**Answer**:

The provided context does not contain information about familial atrial fibrillation. While it discusses atrial fibrillation (AF) in general, including its symptoms, diagnosis, and possible causes, there is no mention of a familial or inherited form of atrial fibrillation. Therefore, based on the given context, familial atrial fibrillation cannot be defined or described.

[medquad_ans_0000034-1__dup2]

**Supported citations**: `medquad_ans_0000034-1__dup2`
**Unsupported citations**: _none_

### Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Scores**: context_relevance=0.857, groundedness=0.544, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.8245) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the assoc...
2. `medquad_ans_0000645-1__dup2` (score=0.5608) - The brain, spinal cord, and nerves make up the nervous system. Together they control all the workings of the body. When something goes wrong with a part of your nervous system, you can have trouble moving, speaking, s...
3. `medquad_ans_0000922-1` (score=0.0092) - SOST-related sclerosing bone dysplasia is a disorder of bone development characterized by excessive bone formation (hyperostosis). As a result of hyperostosis, bones throughout the body are denser and wider than norma...
4. `medquad_ans_0000874-2` (score=0.0044) - This condition affects an estimated 1 in 8,500 females.
5. `medquad_ans_0000166-4__dup1` (score=0.0038) - The NINDS supports an extensive research program of basic studies to increase understanding of diseases that affect the brain, spinal cord, muscles, and nerves. This research examines the genetics, symptoms, progressi...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary significantly from person to person, even within a single family. Features of the condition are generally present at birth and may include skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected people may also have developmental delay, hearing loss, heart defects, and/or ophthalmoplegia (paralysis of the muscles surrounding the eyes). Osteopathia striata cranial sclerosis is caused by changes (mutations) in the WTX gene and is inherited in an X-linked dominant manner. Treatment is based on the signs and symptoms present in each person [medquad_ans_0004

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004097-2` (score=0.9992) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus eryt...
2. `medquad_ans_0001661-2` (score=0.0010) - What are the signs and symptoms of CREST syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for CREST syndrome. If the information is available, the table below includes how often...
3. `medquad_ans_0004097-1` (score=0.0001) - Mixed connective tissue disease (MCTD) is a rare autoimmune disorder that is characterized by features commonly seen in three different connective tissue disorders: systemic lupus erythematosus, scleroderma, and polym...
4. `medquad_ans_0004174-2` (score=0.0001) - What are the signs and symptoms of Multicentric reticulohistiocytosis? The main symptoms of multicentric reticulohistiocytosis are arthritis and red to purple skin nodules varying in size from 1 to 10 mm. The nodules ...
5. `medquad_ans_0003806-1` (score=0.0001) - What are the signs and symptoms of Marinesco-Sjogren syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Marinesco-Sjogren syndrome. If the information is available, the table ...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. Common symptoms include pain in multiple joints, inflammation of joints (arthritis), muscle weakness, fevers, and fatigue. Other signs and symptoms may include accumulation of fluid in the tissue of the hands (edema), skin findings such as lupus-like rashes, reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia). Dysfunction of the esophagus (hypomotility), abnormalities in lung function leading to breathing difficulties or pulmonary hypertension, heart involvement (such as pericarditis, myocarditis, or aortic insufficiency), neurologic abnormalities (e

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Scores**: context_relevance=0.429, groundedness=0.020, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000172-2__dup1` (score=0.9925) - Treatment is symptomatic and supportive, and may include drug therapy.
2. `medquad_ans_0000055-32` (score=0.9901) - People can receive treatment as outpatients (they live at home and visit the doctor or other provider) or through inpatient services (they live temporarily at a special facility where they get treatment). The support ...
3. `medquad_ans_0000239-3__dup1` (score=0.9806) - There is currently no treatment that can halt progression of any of the TSEs. Treatment is aimed at alleviating symptoms and making the patient as comfortable as possible. A clinical trial of a potential therapy for C...
4. `medquad_ans_0000017-2` (score=0.9703) - Currently there are no medicines that can slow the progression of AD. However, four FDA-approved medications are used to treat AD symptoms. These drugs help individuals carry out the activities of daily living by main...
5. `medquad_ans_0000098-2__dup1` (score=0.9580) - Drugs to specifically treat Alzheimers disease and some other progressive dementias are now available. Although these drugs do not halt the disease or reverse existing brain damage, they can improve symptoms and slow ...

**Answer**:

The provided context does not address treatments for Alzheimer's caregiving. Therefore, no answer can be given based on the available information. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Scores**: context_relevance=0.900, groundedness=0.075, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000148-3` (score=0.9990) - Mutations in the ACTA1, TPM2, or TPM3 genes can cause cap myopathy. These genes provide instructions for producing proteins that play important roles in skeletal muscles. The ACTA1 gene provides instructions for makin...
2. `medquad_ans_0000291-3` (score=0.9984) - Mutations in the HSPB1 and HSPB8 genes cause distal hereditary motor neuropathy, type II. These genes provide instructions for making proteins called heat shock protein beta-1 and heat shock protein beta-8. Heat shock...
3. `medquad_ans_0000511-3` (score=0.9972) - Idiopathic inflammatory myopathy is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Researchers have identi...
4. `medquad_ans_0000686-3` (score=0.0418) - Mutations in the RYR1 and SEPN1 genes cause multiminicore disease. The severe, classic form of multiminicore disease is usually caused by mutations in the SEPN1 gene. This gene provides instructions for making a prote...
5. `medquad_ans_0000293-3` (score=0.0024) - A mutation in the MATR3 gene has been identified in people with distal myopathy 2. This gene provides instructions for making a protein called matrin 3, which is found in the nucleus of the cell as part of the nuclear...

**Answer**:

The provided context does not contain information about genetic changes related to idiopathic inflammatory myopathy. Therefore, no answer can be provided based on the given context. [context_missing]

**Supported citations**: _none_
**Unsupported citations**: `context_missing`

### Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0005225-5` (score=0.9947) - Is genetic testing available for pyruvate kinase deficiency? Yes. GeneTests lists laboratories offering clinical genetic testing for this condition. Clinical genetic tests are ordered to help diagnose an affected pers...
2. `medquad_ans_0005223-5` (score=0.9899) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for...
3. `medquad_ans_0003868-5` (score=0.0164) - How is medium-chain acyl-coenzyme A dehydrogenase deficiency (MCADD) diagnosed? MCADD is now included in many newborn screening programs. If a newborn screening result for MCADD is not in the normal range, additional ...
4. `medquad_ans_0005926-5` (score=0.0002) - How is succinic semialdehyde dehydrogenase deficiency diagnosed? The diagnosis of succinic semialdehyde dehydrogenase (SSADH) deficiency is based upon a thorough clinical exam, the identification of features consisten...
5. `medquad_ans_0000855-2` (score=0.0001) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.

**Answer**:

The diagnosis of pyruvate dehydrogenase (PDH) deficiency may be considered in any individual with early-onset neurological disease, especially if it appears to be associated with structural abnormalities in the brain and unexplained lactic acidosis. When lactic acid (also called lactate) and pyruvate in the blood do not seem to be significantly high, an important clue to the diagnosis may be high concentrations of lactate and/or pyruvate in the cerebrospinal fluid (the fluid that surrounds the brain and spinal cord). Additionally, magnetic resonance spectroscopy (MRS) of the brain may show concentrations of lactate in the central nervous system. Analysis of serum and urine amino acids usually shows hyperalaninemia (high levels of the amino acid alanine). When lactic acidosis is present, other disorders involving pyruvate abnormalities are part of the differential diagnosis. However, in all of these conditions,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.925, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006562-2` (score=0.9991) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb a...
2. `medquad_ans_0001242-1` (score=0.0003) - What are the signs and symptoms of Chromosome 17q11.2 deletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Chromosome 17q11.2 deletion syndrome. If the information is a...
3. `medquad_ans_0006572-2` (score=0.0001) - What are the signs and symptoms of 22q13.3 deletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 22q13.3 deletion syndrome. If the information is available, the table be...
4. `medquad_ans_0006610-1` (score=0.0001) - What are the signs and symptoms of 8p23.1 duplication syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 8p23.1 duplication syndrome. If the information is available, the tabl...
5. `medquad_ans_0001293-1` (score=0.0001) - What are the signs and symptoms of Chromosome 6q25 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Chromosome 6q25 microdeletion syndrome. If the information ...

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

**Scores**: context_relevance=1.000, groundedness=0.543, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003860-3` (score=0.9952) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
2. `medquad_ans_0001013-4` (score=0.0106) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000292-4` (score=0.0082) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Some people who have the altered gene never develop the conditio...
4. `medquad_ans_0000948-4` (score=0.0007) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000469-4` (score=0.0006) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located on the long (q) arm of the X chromosome. The size of the duplication can vary; however, this does not appear to affect the severity of the condition. People with larger duplications have signs and symptoms that are similar to people with smaller duplications. The MECP2 gene encodes a protein that is important for normal brain functioning. Although it plays many roles, one of its most important functions is to regulate other genes in the brain by switching them on and off. A duplication of the MECP2 gene leads to the production of excess protein, which is unable to properly regulate the expression of other genes. This results in irregular brain activity, leading to the signs and symptoms of MECP2 duplication syndrome [medquad_ans_000386

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 187 (`0004633-2`)

**Question**: What are the symptoms of Osteopetrosis autosomal dominant type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.536, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004633-2` (score=0.9994) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 2. If the informatio...
2. `medquad_ans_0004635-2` (score=0.0002) - What are the signs and symptoms of Osteopetrosis autosomal recessive 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 2. If the information is ava...
3. `medquad_ans_0004632-2` (score=0.0001) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 1. If the informatio...
4. `medquad_ans_0000096-1__dup1` (score=0.0001) - What are the signs and symptoms of Acroosteolysis dominant type? The Human Phenotype Ontology provides the following list of signs and symptoms for Acroosteolysis dominant type. If the information is available, the ta...
5. `medquad_ans_0001170-2` (score=0.0001) - What are the signs and symptoms of Chondrodysplasia punctata 2 X-linked dominant? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 2 X-linked dominant. If th...

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

**Scores**: context_relevance=0.222, groundedness=0.000, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000209-3` (score=0.9969) - With treatment, muscle weakness may improve or be reversed.
2. `medquad_ans_0000043-2__dup2` (score=0.9968) - In most cases of BEB the treatment of choice is botulinum toxin injections which relax the muscles and stop the spasms. Other treatment options include medications (drug therapy) or surgery--either local surgery of th...
3. `medquad_ans_0000172-2__dup1` (score=0.9967) - Treatment is symptomatic and supportive, and may include drug therapy.
4. `medquad_ans_0000255-2__dup1` (score=0.9961) - Treatment may include such medications as baclofen, diazepam, tizanidine or clonazepam. Physical therapy regimens may include muscle stretching and range of motion exercises to help prevent shrinkage or shortening of ...
5. `medquad_ans_0000201-3__dup1` (score=0.9959) - Improvement in muscle strength usually begins within 3 to 6 weeks after treatment is started. Most patients who receive treatment early experience little, if any, disability. However, there is evidence of slow progres...

**Answer**:

The provided context does not contain information about the treatments for Lambert Eaton myasthenic syndrome. Therefore, a specific answer cannot be given based on the available context. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000445-2__dup1` (score=0.9993) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes...
2. `medquad_ans_0004582-1` (score=0.0001) - What are the signs and symptoms of Ornithine translocase deficiency syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Ornithine translocase deficiency syndrome. If the inform...
3. `medquad_ans_0002602-1` (score=0.0001) - What are the signs and symptoms of Glutamine deficiency, congenital? The Human Phenotype Ontology provides the following list of signs and symptoms for Glutamine deficiency, congenital. If the information is available...
4. `medquad_ans_0001971-1` (score=0.0001) - What are the signs and symptoms of Dyskeratosis congenita autosomal dominant? The Human Phenotype Ontology provides the following list of signs and symptoms for Dyskeratosis congenita autosomal dominant. If the inform...
5. `medquad_ans_0001972-1` (score=0.0001) - What are the signs and symptoms of Dyskeratosis congenita autosomal recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Dyskeratosis congenita autosomal recessive. If the info...

**Answer**:

The symptoms of Arginase deficiency include aminoaciduria (90%), behavioral abnormality (90%), cognitive impairment (90%), neurological speech impairment (90%), EEG abnormality (50%), hemiplegia/hemiparesis (50%), hyperammonemia (50%), and seizures (50%). Other reported symptoms are anorexia, autosomal recessive inheritance, diaminoaciduria, hyperactivity, intellectual disability, irritability, oroticaciduria, postnatal growth retardation, progressive spastic quadriplegia, and vomiting [medquad_ans_0000445-2__dup1].

**Supported citations**: `medquad_ans_0000445-2__dup1`
**Unsupported citations**: _none_

### Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Scores**: context_relevance=0.778, groundedness=0.991, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000945-5` (score=0.9974) - These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliative Care
2. `medquad_ans_0000465-5` (score=0.9925) - These resources address the diagnosis or management of HLRCC: - Gene Review: Gene Review: Hereditary Leiomyomatosis and Renal Cell Cancer - Genetic Testing Registry: Hereditary leiomyomatosis and renal cell cancer - M...
3. `medquad_ans_0000022-2__dup1` (score=0.9864) - Treatment for VHL varies according to the location and size of the tumor. In general, the objective of treatment is to treat the tumors before they grow to a size large enough to cause permanent problems by putting pr...
4. `medquad_ans_0006376-6` (score=0.9676) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but ar...
5. `medquad_ans_0000737-5` (score=0.0261) - These resources address the diagnosis or management of nonsyndromic paraganglioma: - Genetic Testing Registry: Pheochromocytoma These resources from MedlinePlus offer information about the diagnosis and management of ...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms but are still small so they don't cause permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All people with VHL disease should be carefully followed by a physician or medical team familiar with the disorder [medquad_ans_0006376-6].

**Supported citations**: `medquad_ans_0006376-6`
**Unsupported citations**: _none_

### Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Scores**: context_relevance=0.800, groundedness=0.701, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004452-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
2. `medquad_ans_0004453-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0004454-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004451-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004450-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

The provided context does not mention "Noonan syndrome 3" or any specific subtype of Noonan syndrome beyond general references to subtypes based on the responsible gene. Therefore, based on the available information, Noonan syndrome 3 cannot be identified or described.

Answer: Insufficient context to determine what Noonan syndrome 3 is. [medquad_ans_0004451-1]

**Supported citations**: `medquad_ans_0004451-1`
**Unsupported citations**: _none_

### Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005865-2` (score=0.9991) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may...
2. `medquad_ans_0002966-1` (score=0.0004) - What are the signs and symptoms of Hydrocephalus, costovertebral dysplasia, and Sprengel anomaly? The Human Phenotype Ontology provides the following list of signs and symptoms for Hydrocephalus, costovertebral dyspla...
3. `medquad_ans_0004790-1` (score=0.0001) - What are the signs and symptoms of Pelvic dysplasia arthrogryposis of lower limbs? The Human Phenotype Ontology provides the following list of signs and symptoms for Pelvic dysplasia arthrogryposis of lower limbs. If ...
4. `medquad_ans_0005632-1` (score=0.0001) - What are the signs and symptoms of Sillence syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Sillence syndrome. If the information is available, the table below includes how...
5. `medquad_ans_0002328-2` (score=0.0001) - What are the signs and symptoms of Femoral facial syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Femoral facial syndrome. If the information is available, the table below ...

**Answer**:

The symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additional features that may be present include abnormality of the shoulder (seen in approximately 90% of patients), cleft palate (in about 7.5% of patients), scoliosis, hemivertebrae, neck muscle hypoplasia, rib segmentation abnormalities, cervical segmentation defects, spina bifida occulta, and abnormalities of the collarbone or chest organs [medquad_ans_0005865-2]. The condition may also be associated with limb length discrepancy or abnormalities in the spine or ribs [medquad_ans_0005865-2].

**Supported citations**: `medquad_ans_0005865-2`
**Unsupported citations**: _none_

### Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.493, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000487-3` (score=0.9991) - HGPPS is caused by mutations in the ROBO3 gene. This gene provides instructions for making a protein that is important for the normal development of certain nerve pathways in the brain. These include motor nerve pathw...
2. `medquad_ans_0000882-3` (score=0.9988) - Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene. This gene provides instructions for making a protein whose function is not well understood, although it is involved in chemical signaling p...
3. `medquad_ans_0000979-3` (score=0.9966) - Mutations in the RBM8A gene cause TAR syndrome. The RBM8A gene provides instructions for making a protein called RNA-binding motif protein 8A. This protein is believed to be involved in several important cellular func...
4. `medquad_ans_0000373-3` (score=0.0161) - Researchers have identified changes in five regions of the X chromosome that are linked to FG syndrome in affected families. Mutations in a gene called MED12, which is located in one of these regions, appear to be the...
5. `medquad_ans_0000606-3` (score=0.0060) - Mutations in the OCRL gene cause Lowe syndrome. The OCRL gene provides instructions for making an enzyme that helps modify fat (lipid) molecules called membrane phospholipids. By controlling the levels of specific mem...

**Answer**:

Genetic changes related to Robinow syndrome include mutations in the ROR2 gene, which cause autosomal recessive Robinow syndrome. These mutations prevent cells from making any functional ROR2 protein, disrupting development before birth and leading to characteristic features of the syndrome. Additionally, autosomal dominant Robinow syndrome can be caused by mutations in the WNT5A or DVL1 gene, with the osteosclerotic form resulting from DVL1 gene mutations. The proteins from these genes are part of the same chemical signaling pathways as the ROR2 protein, and mutations alter the production or function of these proteins, impairing essential chemical signaling during early development. In some cases, individuals with the signs and symptoms of Robinow syndrome do not have an identified mutation in the ROR2, WNT5A, or DVL1 gene, and the cause remains unknown [doc_id: medquad_ans_0000882-

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Scores**: context_relevance=1.000, groundedness=0.160, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000157-12` (score=0.9904) - Before you were on dialysis, your doctor may have told you to follow a low-protein diet. Being on dialysis changes this. Most people on dialysis are encouraged to eat as much high-quality protein as they can. Protein ...
2. `medquad_ans_0000157-11` (score=0.9904) - Before you were on dialysis, your doctor may have told you to follow a low-protein diet. Being on dialysis changes this. Most people on dialysis are encouraged to eat as much high-quality protein as they can. Protein ...
3. `medquad_ans_0000174-11` (score=0.9894) - Eating the right foods can help you feel better while on peritoneal dialysis. Talk with your dialysis centers dietitian to find a meal plan that works for you. Your dietary needs will depend on your treatment and othe...
4. `medquad_ans_0000058-11__dup1` (score=0.9894) - Kidney failure, also called end-stage kidney disease or ESRD, means your kidneys no longer work well enough to do their job. You will need treatment to replace the work your kidneys have stopped doing.
5. `medquad_ans_0000179-6` (score=0.9805) - Peritoneal dialysis uses the lining of your belly to filter your blood inside your body. You can do peritoneal dialysis at home because it uses your body to filter. A doctor will place a soft tube called a catheter in...

**Answer**:

The provided context does not contain the specific title or content of "Kidney Failure: Eat Right to Feel Right on Hemodialysis." Therefore, it is not possible to answer the question based on the given information. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Scores**: context_relevance=0.222, groundedness=0.682, answer_relevance=0.311

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
3. `medquad_ans_0000681-2` (score=0.9995) - The exact prevalence of MPS IV is unknown, although it is estimated to occur in 1 in 200,000 to 300,000 individuals.
4. `medquad_ans_0000790-2` (score=0.9995) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
5. `medquad_ans_0000874-2` (score=0.9995) - This condition affects an estimated 1 in 8,500 females.

**Answer**:

The exact prevalence of mucolipidosis type IV is unknown, although it is estimated to occur in 1 in 200,000 to 300,000 individuals [medquad_ans_0000681-2]. Other estimates range from 1 in 25,000 to 300,000 individuals [medquad_ans_0000790-2]. However, no specific data is provided for mucolipidosis type IV in the given context, and the information provided is for MPS IV, which may be a different condition. Therefore, based on the available context, the prevalence of mucolipidosis type IV cannot be accurately determined.

**Supported citations**: `medquad_ans_0000681-2`, `medquad_ans_0000790-2`
**Unsupported citations**: _none_

### Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Scores**: context_relevance=1.000, groundedness=0.994, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.9368) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chi...
2. `medquad_ans_0000193-1__dup4` (score=0.0106) - Cirrhosis\* is scarring of the liver. Scar tissue forms because of injury or long-term disease. Scar tissue replaces healthy liver tissue and blocks the normal flow of blood through the liver. A healthy liver - makes p...
3. `medquad_ans_0000193-6` (score=0.0084) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need t...
4. `medquad_ans_0000184-1__dup3` (score=0.0045) - Cirrhosis is a condition in which the liver slowly deteriorates and is unable to function normally due to chronic, or long lasting, injury. Scar tissue replaces healthy liver tissue and partially blocks the flow of bl...
5. `medquad_ans_0000712-1__dup1` (score=0.0038) - Piercings and tattoos are body decorations that go back to ancient times. Body piercing involves making a hole in the skin so that you can insert jewelry. This is often in the earlobe, but can be in other parts of the...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams. [medquad_ans_0000795-1__dup1]

**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

### Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Scores**: context_relevance=0.556, groundedness=0.912, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0001045-3` (score=0.9992) - Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. The ABCD1 gene provides instructions for producing the adrenoleukodystrophy protein (ALDP), which is involved in transporting certain fat molecules call...
2. `medquad_ans_0001013-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000342-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0001054-4` (score=0.0001) - This condition is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each cell. In males...
5. `medquad_ans_0000455-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. These mutations result in a deficiency of the adrenoleukodystrophy protein (ALDP), which is responsible for transporting very long-chain fatty acids (VLCFAs) into peroxisomes. The lack of this protein disrupts the transport and breakdown of VLCFAs, leading to their accumulation in the body. This accumulation is believed to be toxic to the adrenal cortex and myelin, and may trigger an inflammatory response in the brain, contributing to the destruction of myelin and the signs and symptoms of the disorder [medquad_ans_0001045-3]. The condition is inherited in an X-linked pattern, meaning the mutated gene is located on the X chromosome, and in males, one altered copy of the gene is sufficient to cause the disorder [medquad_ans_0001054-4].

**Supported citations**: `medquad_ans_0001045-3`, `medquad_ans_0001054-4`
**Unsupported citations**: _none_

### Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Scores**: context_relevance=0.125, groundedness=0.798, answer_relevance=0.325

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
3. `medquad_ans_0000790-2` (score=0.9995) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
4. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000770-2` (score=0.9995) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.

**Answer**:

The exact incidence of transthyretin amyloidosis is unknown, but estimates vary. It is estimated to affect 1 in 87,000 people [medquad_ans_0000484-2], 1 in 40,000 to 50,000 people [medquad_ans_0000195-2], 1 in 50,000 people [medquad_ans_0000989-2], and ranges from 1 in 25,000 to 1 in 300,000 individuals [medquad_ans_0000790-2]. Additionally, it is estimated to affect 1 to 3 per million people worldwide [medquad_ans_0000770-2].

**Supported citations**: `medquad_ans_0000484-2`, `medquad_ans_0000195-2`, `medquad_ans_0000989-2`, `medquad_ans_0000790-2`, `medquad_ans_0000770-2`
**Unsupported citations**: _none_

### Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Scores**: context_relevance=1.000, groundedness=0.787, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.9994) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the informatio...
2. `medquad_ans_0000525-1__dup1` (score=0.0001) - What are the signs and symptoms of Auriculo-condylar syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Auriculo-condylar syndrome. If the information is available, the table ...
3. `medquad_ans_0006175-1` (score=0.0001) - What are the signs and symptoms of Tricho-dento-osseous syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Tricho-dento-osseous syndrome 1. If the information is available, ...
4. `medquad_ans_0001740-1` (score=0.0001) - What are the signs and symptoms of Deafness oligodontia syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness oligodontia syndrome. If the information is available, the ...
5. `medquad_ans_0006174-1` (score=0.0001) - What are the signs and symptoms of Tricho-dento-osseous syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tricho-dento-osseous syndrome. If the information is available, the ...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include a range of dental and oral features. According to the Human Phenotype Ontology (HPO), the specific symptoms are listed, and their frequency is estimated based on studies. The frequency of a symptom is typically given as a percentage or a fraction (e.g., 25/25 means all patients in a study had the symptom). However, the provided context does not list the exact symptoms of Dens in dente and palatal invaginations beyond stating that the HPO has compiled information on their occurrence [medquad_ans_0001780-1]. Therefore, the specific symptoms are not detailed in the available text.

**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

### Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Scores**: context_relevance=0.125, groundedness=0.750, answer_relevance=0.325

**Top docs**:

1. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
2. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
3. `medquad_ans_0000904-2` (score=0.9995) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.
4. `medquad_ans_0000874-2` (score=0.9995) - This condition affects an estimated 1 in 8,500 females.
5. `medquad_ans_0000886-2` (score=0.9995) - This condition is uncommon; it occurs in an estimated 1 in 100,000 to 125,000 newborns.

**Answer**:

The exact incidence of periventricular heterotopia is unknown, but it is estimated to affect 1 in 87,000 people [medquad_ans_0000484-2]. Additionally, it is thought to affect approximately 1 in 35,000 to 50,000 newborns [medquad_ans_0000904-2], and it occurs in an estimated 1 in 100,000 to 125,000 newborns [medquad_ans_0000886-2]. However, the most specific estimate provided for the general population is 1 in 87,000 [medquad_ans_0000484-2].

**Supported citations**: `medquad_ans_0000484-2`, `medquad_ans_0000904-2`, `medquad_ans_0000886-2`
**Unsupported citations**: _none_

### Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Scores**: context_relevance=0.273, groundedness=0.593, answer_relevance=0.636

**Top docs**:

1. `medquad_ans_0000214-4__dup1` (score=0.9987) - The NINDS conducts and supports a wide range of studies that explore the complex systems of brain development. These studies include the identification of the mechanism of action of the known causes of NMD as well as ...
2. `medquad_ans_0000113-4__dup1` (score=0.9987) - The NINDS conducts and supports an extensive research program on seizures and seizure-related disorders. Much of this research is aimed at increasing scientific understanding of these disorders and finding ways to pre...
3. `medquad_ans_0000023-4__dup1` (score=0.9986) - The NINDS supports and conducts studies aimed at understanding neurological conditions that can damage the brain, such as cerebral hypoxia. The goals of these studies are to find ways to prevent and treat these condit...
4. `medquad_ans_0000171-4__dup1` (score=0.9985) - The NINDS supports research on congenital disorders such as KTS with the goal of finding new means to treat and prevent them.
5. `medquad_ans_0000118-4__dup1` (score=0.9985) - The NINDS supports and conducts research on brain diseases. Much of this research is aimed at characterizing the agents that cause these disorders, clarifying the mechanisms underlying them, and, ultimately, finding w...

**Answer**:

The provided context does not mention Landau-Kleffner Syndrome or any specific research or clinical trials related to it. While the NINDS supports research on congenital disorders such as KTS (which may refer to a condition like Klinefelter syndrome or another congenital disorder), the context does not specify the research being conducted for Landau-Kleffner Syndrome [medquad_ans_0000171-4__dup1]. Therefore, based on the available information, the research for Landau-Kleffner Syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000171-4__dup1`
**Unsupported citations**: _none_

### Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0004219-1` (score=0.9721) - Muscular dystrophy (MD) refers to a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or ...
2. `medquad_ans_0000205-1__dup2` (score=0.9314) - The muscular dystrophies (MD) are a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or ...
3. `medquad_ans_0000631-1__dup1` (score=0.8258) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...
4. `medquad_ans_0000710-1` (score=0.0819) - Myotonic dystrophy is part of a group of inherited disorders called muscular dystrophies. It is the most common form of muscular dystrophy that begins in adulthood. Myotonic dystrophy is characterized by progressive m...
5. `medquad_ans_0000602-1` (score=0.0265) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifi...

**Answer**:

Muscular dystrophy (MD) is a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or childhood, while others may not appear until middle age or later. The disorders differ in terms of the distribution and extent of muscle weakness (some forms of MD also affect cardiac muscle), age of onset, rate of progression, and pattern of inheritance [medquad_ans_0004219-1]. Duchenne MD is the most common form and primarily affects boys, caused by the absence of dystrophin, a protein involved in maintaining the integrity of muscle, with onset between 3 and 5 years and rapid progression, leading to loss of walking ability by age 12 and later requiring a respirator [medquad_ans_0000205-1__dup2]. Facioscapulohumeral

**Supported citations**: `medquad_ans_0004219-1`, `medquad_ans_0000205-1__dup2`
**Unsupported citations**: _none_

### Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.342, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.8062) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that t...
2. `medquad_ans_0005582-1` (score=0.0004) - What are the signs and symptoms of Short rib-polydactyly syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Short rib-polydactyly syndrome type 3. If the information is...
3. `medquad_ans_0004054-1` (score=0.0002) - What are the signs and symptoms of Microsomia hemifacial radial defects? The Human Phenotype Ontology provides the following list of signs and symptoms for Microsomia hemifacial radial defects. If the information is a...
4. `medquad_ans_0005020-1` (score=0.0002) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the tabl...
5. `medquad_ans_0005585-1` (score=0.0001) - What are the signs and symptoms of Short rib-polydactyly syndrome type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Short rib-polydactyly syndrome type 4. If the information is...

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding with no other physical anomalies or intellectual impairment, or it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits, with surgical management depending on the complexity of the deformity [1] [2].

**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

### Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Scores**: context_relevance=1.000, groundedness=0.862, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002519-2` (score=0.9994) - What are the signs and symptoms of Geniospasm? The Human Phenotype Ontology provides the following list of signs and symptoms for Geniospasm. If the information is available, the table below includes how often the sym...
2. `medquad_ans_0000525-1__dup1` (score=0.0001) - What are the signs and symptoms of Auriculo-condylar syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Auriculo-condylar syndrome. If the information is available, the table ...
3. `medquad_ans_0000085-1__dup1` (score=0.0001) - What are the signs and symptoms of Acromegaloid facial appearance syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Acromegaloid facial appearance syndrome. If the informatio...
4. `medquad_ans_0000756-1__dup1` (score=0.0001) - What are the signs and symptoms of Blepharophimosis with ptosis, syndactyly, and short stature? The Human Phenotype Ontology provides the following list of signs and symptoms for Blepharophimosis with ptosis, syndacty...
5. `medquad_ans_0003846-1` (score=0.0001) - What are the signs and symptoms of McDonough syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for McDonough syndrome. If the information is available, the table below includes h...

**Answer**:

The symptoms of Geniospasm include chin myoclonus. This information is derived from the Human Phenotype Ontology, which lists chin myoclonus as a sign or symptom associated with Geniospasm [medquad_ans_0002519-2].

**Supported citations**: `medquad_ans_0002519-2`
**Unsupported citations**: _none_

### Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=0.800, groundedness=0.033, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000963-3` (score=0.9984) - Surfactant dysfunction is caused by mutations in one of several genes, including SFTPB, SFTPC, and ABCA3. Each of these genes is involved in the production of surfactant. The production and release of surfactant is a ...
2. `medquad_ans_0000512-3` (score=0.9928) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of devel...
3. `medquad_ans_0000041-3` (score=0.8595) - Mutations in the SERPINA1 gene cause alpha-1 antitrypsin deficiency. This gene provides instructions for making a protein called alpha-1 antitrypsin, which protects the body from a powerful enzyme called neutrophil el...
4. `medquad_ans_0000825-3` (score=0.0097) - Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition. The FLCN gene provides instructions for making a protein called follicu...
5. `medquad_ans_0000828-3` (score=0.0033) - Mutations in the SCN5A and TRPM4 genes cause most cases of progressive familial heart block types IA and IB, respectively. The proteins produced from these genes are channels that allow positively charged atoms (catio...

**Answer**:

The provided context does not contain any information about genetic changes related to idiopathic pulmonary fibrosis. Therefore, it is not possible to answer the question based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000100-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0001005-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000237-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000150-4` (score=0.9990) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Potocki-Shaffer syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Potocki-Shaffer syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Scores**: context_relevance=0.714, groundedness=0.747, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000190-1__dup2` (score=0.9980) - Summary : Kids, this page is for you. Learn about everything from how the body works to what happens when you go to the hospital. There are quizzes, games and lots of cool web sites for you to explore. Have fun!
2. `medquad_ans_0000076-1__dup3` (score=0.9978) - Summary : There are many new responsibilities when you have a baby. One of them is to make sure they get the checkups that they need. Well-baby exams are important in making sure that your baby is growing and developi...
3. `medquad_ans_0000414-1__dup2` (score=0.9976) - Frequently Asked Questions (FAQs) Fact Sheets
4. `medquad_ans_0000897-1__dup1` (score=0.9972) - Summary : Most young children get sick. It is hard for parents to know what is serious. You can learn what the common warning signs are. In the end, trust your intuition. If you are worried about your toddler, call yo...
5. `medquad_ans_0000189-1__dup2` (score=0.9970) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...

**Answer**:

No, there is no information about child care in the provided context. [medquad_ans_0000190-1__dup2] discusses general child health, [medquad_ans_0000076-1__dup3] covers well-baby exams and immunizations, [medquad_ans_0000897-1__dup1] addresses warning signs in toddlers, and [medquad_ans_0000189-1__dup2] lists health concerns for school-age children. None of these sources mention child care.

**Supported citations**: `medquad_ans_0000190-1__dup2`, `medquad_ans_0000076-1__dup3`, `medquad_ans_0000897-1__dup1`, `medquad_ans_0000189-1__dup2`
**Unsupported citations**: _none_

### Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Scores**: context_relevance=0.200, groundedness=0.790, answer_relevance=0.440

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0001092-2` (score=0.9995) - This condition is detected in an estimated 1 in 36,000 newborns worldwide.
3. `medquad_ans_0000904-2` (score=0.9995) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.
4. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000886-2` (score=0.9995) - This condition is uncommon; it occurs in an estimated 1 in 100,000 to 125,000 newborns.

**Answer**:

The exact incidence of alternating hemiplegia of childhood is unknown, but it is estimated to affect 1 in 87,000 people [medquad_ans_0000484-2]. Additionally, it is detected in an estimated 1 in 36,000 newborns worldwide [medquad_ans_0001092-2], and thought to affect approximately 1 in 35,000 to 50,000 newborns [medquad_ans_0000904-2]. Another estimate suggests it affects 1 in 50,000 people [medquad_ans_0000989-2], and occurs in 1 in 100,000 to 125,000 newborns [medquad_ans_0000886-2

**Supported citations**: `medquad_ans_0000484-2`, `medquad_ans_0001092-2`, `medquad_ans_0000904-2`, `medquad_ans_0000989-2`
**Unsupported citations**: _none_

### Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003687-2` (score=0.9992) - What are the signs and symptoms of Lupus? You can read about the signs and symptoms of lupus from MedlinePlus and the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS). The Human Phenotype ...
2. `medquad_ans_0000159-3__dup2` (score=0.9977) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time...
3. `medquad_ans_0006001-2` (score=0.0001) - What are the signs and symptoms of Takayasu arteritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Takayasu arteritis. If the information is available, the table below includes h...
4. `medquad_ans_0002327-2` (score=0.0001) - What are the signs and symptoms of Felty's syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Felty's syndrome. If the information is available, the table below includes how o...
5. `medquad_ans_0000966-1` (score=0.0001) - Systemic lupus erythematosus (SLE) is a chronic disease that causes inflammation in connective tissues, such as cartilage and the lining of blood vessels, which provide strength and flexibility to structures throughou...

**Answer**:

The symptoms of Lupus include abdominal pain, abnormality of temperature regulation, abnormality of the heart valves, abnormality of the pericardium, alopecia, arthralgia, arthritis, autoimmunity, chest pain, cutaneous photosensitivity, skin rash, thrombocytopenia, thrombophlebitis, abnormal pyramidal signs, abnormal tendon morphology, abnormality of the autonomic nervous system, abnormality of the endocardium, abnormality of the pleura, anorexia, arterial thrombosis, aseptic leukocyturia, bone marrow hypocellularity, conjunctival telangiectasia, cranial nerve paralysis, cutis marmorata, dry skin, eczema, edema of the lower limbs, glomerulopathy, hallucinations, hematuria, hepatomegaly, hyperkeratosis, hypoproteinemia, increased antibody level in blood

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.531, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000194-1` (score=0.2765) - Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features of this disorder include a failure to gain weight and grow at the expected rate (failure to thrive), a...
2. `medquad_ans_0001397-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
3. `medquad_ans_0001398-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
4. `medquad_ans_0001395-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
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

**Scores**: context_relevance=0.111, groundedness=0.717, answer_relevance=0.378

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0001101-2` (score=0.9995) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
3. `medquad_ans_0000904-2` (score=0.9995) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.
4. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.

**Answer**:

The exact incidence of hyperkalemic periodic paralysis is unknown, but it is estimated to affect 1 in 87,000 people [medquad_ans_0000484-2]. Additionally, other estimates suggest it affects 1 in 35,000 to 50,000 newborns [medquad_ans_0000904-2], 1 in 40,000 to 50,000 people [medquad_ans_0000195-2], and 1 in 50,000 people [medquad_ans_0000989-2]. The prevalence ranges from 1 in 7,500 to 20,000 people [medquad_ans_0001101-2]. Due to the variation in estimates,

**Supported citations**: `medquad_ans_0000484-2`, `medquad_ans_0000904-2`, `medquad_ans_0000195-2`, `medquad_ans_0000989-2`, `medquad_ans_0001101-2`
**Unsupported citations**: _none_

### Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Scores**: context_relevance=0.800, groundedness=0.991, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0005212-1` (score=0.8399) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end o...
2. `medquad_ans_0000121-12` (score=0.0115) - Eating, diet, and nutrition can play a role in causing, preventing, and treating some of the diseases and disorders of the digestive system that are affected by smoking, including heartburn and GERD, liver diseases, C...
3. `medquad_ans_0000203-3__dup1` (score=0.0090) - The disease tends to advance rapidly over the course of 5 to 10 years, with progressive loss of motor skills, eventual confinement to bed, and death. There is no remission from the disease. There is currently no cure.
4. `medquad_ans_0000874-2` (score=0.0087) - This condition affects an estimated 1 in 8,500 females.
5. `medquad_ans_0000989-2` (score=0.0067) - This condition affects an estimated 1 in 50,000 people.

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end of the bone), widening of the ribs and clavicles (collarbones), platyspondyly (flattening of the bones of the spine) and cortical thinning. Cranial involvement is minimal with some showing mild hyperostosis (excessive new bone formation) of the skull base and thickening of the frontal and occipital bones. Pyle disease is passed through families in an autosomal recessive manner [medquad_ans_0005212-1].

**Supported citations**: `medquad_ans_0005212-1`
**Unsupported citations**: _none_

### Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Scores**: context_relevance=0.800, groundedness=0.946, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000051-3__dup2` (score=0.9954) - Blood clots can form in your body's deep veins if: A vein's inner lining is damaged. Injuries caused by physical, chemical, or biological factors can damage the veins. Such factors include surgery, serious injuries, i...
2. `medquad_ans_0000108-3__dup2` (score=0.0331) - Major Causes Pulmonary embolism (PE) usually begins as a blood clot in a deep vein of the leg. This condition is called deep vein thrombosis. The clot can break free, travel through the bloodstream to the lungs, and b...
3. `medquad_ans_0000039-5__dup2` (score=0.0004) - While new technology and advances in surgical techniques have greatly reduced the risks involved with knee replacements, there are still some risks you should be aware of. Two of the most common possible problems are ...
4. `medquad_ans_0000108-4__dup2` (score=0.0002) - Pulmonary embolism (PE) occurs equally in men and women. The risk increases with age. For every 10 years after age 60, the risk of having PE doubles. Certain inherited conditions, such as factor V Leiden, increase the...
5. `medquad_ans_0000104-1__dup4` (score=0.0001) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent b...

**Answer**:

Deep Vein Thrombosis (DVT) is caused by:

1. Damage to a vein's inner lining, which can result from injuries caused by physical, chemical, or biological factors such as surgery, serious injuries, inflammation, or immune responses [medquad_ans_0000051-3__dup2].
2. Sluggish or slow blood flow, which may occur due to lack of motion, such as after surgery, being ill and bedridden for a long time, or prolonged travel (e.g., long car or airplane trips) [medquad_ans_0000051-3\_\_dup2][medquad_ans_0000108-3__dup2].
3. Blood that is thicker or more likely to clot than normal, which can be due to inherited conditions (e.g., factor V Leiden) or the use of hormone therapy or birth control

**Supported citations**: `medquad_ans_0000051-3__dup2`, `medquad_ans_0000108-3__dup2`
**Unsupported citations**: _none_

### Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Scores**: context_relevance=0.714, groundedness=0.689, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0004081-1` (score=0.5514) - Cytochrome C oxidase deficiency (COX deficiency) is a condition that can affect several parts of the body including the skeletal muscles, heart, brain and liver. The range and severity of signs and symptoms can vary w...
2. `medquad_ans_0004086-1` (score=0.0234) - Mitochondrial genetic disorders refer to a group of conditions that affect the mitochondria (the structures in each cell of the body that are responsible for making energy). People with these conditions can present at...
3. `medquad_ans_0001445-1` (score=0.0088) - Combined oxidative phosphorylation deficiency 16, also know as infantile hypertrophic cardiomyopathy, is characterized by decreased levels of mitochondrial complexes. The symptoms and signs described include an enlarg...
4. `medquad_ans_0005223-1` (score=0.0047) - Pyruvate dehydrogenase deficiency is metabolic disorder associated with abnormal function of the mitochondria in cells, thus depriving the body of energy. Progressive neurological symptoms usually start in infancy but...
5. `medquad_ans_0004079-1` (score=0.0011) - Complex II deficiency is a mitochondrial disease. Mitochondria are specialized compartments in cells that create more than 90% of the energy needed by the body. In mitochondrial diseases, the mitochondria don't work c...

**Answer**:

The provided context does not contain information about Mitochondrial complex IV deficiency. While it mentions Cytochrome C oxidase deficiency (COX deficiency) and links it to mitochondrial dysfunction, it does not explicitly define or describe Mitochondrial complex IV deficiency. Since Cytochrome C oxidase is the fourth complex in the mitochondrial electron transport chain, COX deficiency may be related to complex IV deficiency, but this connection is not explicitly stated or confirmed in the provided context.

Answer: Insufficient context to determine what Mitochondrial complex IV deficiency is. [medquad_ans_0004081-1]

**Supported citations**: `medquad_ans_0004081-1`
**Unsupported citations**: _none_

### Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Scores**: context_relevance=1.000, groundedness=0.530, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000078-1` (score=0.4373) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...
2. `medquad_ans_0005761-1` (score=0.0036) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal...
3. `medquad_ans_0000080-1` (score=0.0003) - Ataxia-telangiectasia is a rare inherited disorder that affects the nervous system, immune system, and other body systems. This disorder is characterized by progressive difficulty with coordinating movements (ataxia) ...
4. `medquad_ans_0000941-1` (score=0.0002) - Spinocerebellar ataxia type 36 (SCA36) is a condition characterized by progressive problems with movement that typically begin in mid-adulthood. People with this condition initially experience problems with coordinati...
5. `medquad_ans_0000078-2` (score=0.0002) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.

**Answer**:

Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement, with the hallmark being difficulty coordinating movements (ataxia), which is often the first symptom. Most affected individuals also have oculomotor apraxia, which makes it difficult to move their eyes side-to-side, requiring them to turn their head to see peripheral vision. There are multiple types of this condition, which are very similar but caused by mutations in different genes. The two most common types (types 1 and 2) share features including involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, ataxia begins around age 4, while in type 2, it begins around age 15. Chorea and myoclonus tend to disappear gradually in type 1 but persist throughout life in type 2. Individuals

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Scores**: context_relevance=0.909, groundedness=0.956, answer_relevance=0.909

**Top docs**:

1. `medquad_ans_0000221-1` (score=0.9399) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that fo...
2. `medquad_ans_0001071-2` (score=0.0017) - This condition is very rare; only a few dozen affected individuals have been identified.
3. `medquad_ans_0000346-2` (score=0.0015) - This condition appears to be rare; only a few affected individuals have been reported worldwide.
4. `medquad_ans_0002637-1` (score=0.0003) - Goldenhar disease is a condition that is present at birth and mainly affects the development of the eye, ear and spine. Affected individuals commonly have a partially formed ear (microtia) or totally absent ear (anoti...
5. `medquad_ans_0000248-2` (score=0.0003) - Craniofacial microsomia has been estimated to occur in between 1 in 5,600 and 1 in 26,550 newborns. However, this range may be an underestimate because not all medical professionals agree on the criteria for diagnosis...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects the development of the ears and teeth. In people with this condition, the structures that form the inner ear are usually completely absent (labyrinthine aplasia). This leads to a form of hearing loss called sensorineural deafness that is present from birth (congenital). Due to the inner ear's role in balance and hearing, motor skills such as sitting and crawling may be delayed in affected infants. Additionally, individuals often have abnormally small outer ears (microtia) with narrow ear canals and unusually small, widely spaced teeth (microdontia) [medquad_ans_0000221-1]. The condition is very rare, with only a few dozen affected individuals identified worldwide [medquad_ans_0001071-2] and

**Supported citations**: `medquad_ans_0000221-1`, `medquad_ans_0001071-2`
**Unsupported citations**: _none_

### Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.599, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000948-4` (score=0.9989) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
2. `medquad_ans_0000441-4` (score=0.9988) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has a parent and other family ...
3. `medquad_ans_0000342-4` (score=0.9988) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000403-4` (score=0.9988) - This condition has an autosomal dominant inheritance pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. All reported cases have resulted from new mutations in the gene ...
5. `medquad_ans_0000785-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Apert syndrome specifically. Therefore, based on the given information, it is not possible to determine whether Apert syndrome is inherited.

Note: While the context discusses autosomal dominant and autosomal recessive inheritance patterns, it does not link these patterns to Apert syndrome. [medquad_ans_0000948-4], [medquad_ans_0000441-4], [medquad_ans_0000342-4], [medquad_ans_0000403-4], and [medquad_ans_0000785-4] describe inheritance patterns generally but do not specify the condition as Apert syndrome. Thus, the context is insufficient to answer the question.

**Supported citations**: `medquad_ans_0000948-4`, `medquad_ans_0000441-4`, `medquad_ans_0000342-4`, `medquad_ans_0000403-4`, `medquad_ans_0000785-4`
**Unsupported citations**: _none_

### Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Scores**: context_relevance=0.750, groundedness=0.544, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.1319) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organi...
2. `medquad_ans_0000874-2` (score=0.0112) - This condition affects an estimated 1 in 8,500 females.
3. `medquad_ans_0000341-3__dup2` (score=0.0073) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact ...
4. `medquad_ans_0000989-2` (score=0.0067) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000650-2` (score=0.0065) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air that contains airborne barnyard dust contaminated by dried placental material, birth fluids, and excreta of infected animals. Other modes of transmission to humans, including tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission, are rare. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, the affected people with acute Q fever infection develop a chronic Q fever. Treatment of the acute form is

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Scores**: context_relevance=0.111, groundedness=0.000, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000989-2` (score=0.9992) - This condition affects an estimated 1 in 50,000 people.
2. `medquad_ans_0000650-2` (score=0.9991) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
3. `medquad_ans_0000874-2` (score=0.9991) - This condition affects an estimated 1 in 8,500 females.
4. `medquad_ans_0000934-2` (score=0.9991) - This condition affects fewer than 1 in 150,000 males and is very rare in females.
5. `medquad_ans_0000359-2` (score=0.9989) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.

**Answer**:

The provided context does not specify who is at risk for Childhood Interstitial Lung Disease. Therefore, based on the given information, it is not possible to determine the at-risk population for this condition. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Scores**: context_relevance=0.667, groundedness=0.959, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.9011) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in lat...
2. `medquad_ans_0000874-2` (score=0.0242) - This condition affects an estimated 1 in 8,500 females.
3. `medquad_ans_0000555-1` (score=0.0113) - Kawasaki disease is a sudden and time-limited (acute) illness that affects infants and young children. Affected children develop a prolonged fever lasting several days, a skin rash, and swollen lymph nodes in the neck...
4. `medquad_ans_0000102-3__dup1` (score=0.0091) - The progression of the disorder is usually 10 years or longer.
5. `medquad_ans_0000650-2` (score=0.0062) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels. It is generally diagnosed in late childhood or early adulthood and specifically affects the temporal arteries, which are located at the lower sides of the skull, directly underneath the temple. Affected individuals often have no signs or symptoms aside from a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically with no family history. It is generally treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

### Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.538, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0006577-1` (score=0.9565) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particul...
2. `medquad_ans_0001082-1` (score=0.9285) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as...
3. `medquad_ans_0006575-1` (score=0.0036) - 2-methyl-3-hydroxybutyric aciduria is an inherited disorder in which the body cannot effectively process the amino acid isoleucine. Signs and symptoms of this condition usually develop in infancy or early childhood an...
4. `medquad_ans_0001093-2` (score=0.0007) - 3-methylglutaconyl-CoA hydratase deficiency is a rare disorder; at least 20 cases have been reported in the scientific literature.
5. `medquad_ans_0001082-2` (score=0.0004) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyry...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of the amino acid isoleucine. The inability to process isoleucine correctly leads to the buildup of the amino acid in the body, which can cause a variety of health problems that vary widely from severe and life-threatening to mild or absent. Signs and symptoms can begin a few days after birth or later in childhood and often include poor feeding, lack of energy (lethargy), vomiting, and irritability. These symptoms may progress to serious medical problems such as difficulty breathing, seizures, and coma. Additional issues can include poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills like standing and walking. Symptoms may be triggered by prolonged periods without food (fasting),

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Scores**: context_relevance=1.000, groundedness=0.537, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005840-2` (score=0.9994) - What are the signs and symptoms of Spondyloepiphyseal dysplasia congenita? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia congenita. If the information ...
2. `medquad_ans_0005845-1` (score=0.0001) - What are the signs and symptoms of Spondyloepiphyseal dysplasia-brachydactyly and distinctive speech? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia-bra...
3. `medquad_ans_0005819-2` (score=0.0001) - What are the signs and symptoms of Spondylocostal dysostosis 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylocostal dysostosis 1. If the information is available, the tabl...
4. `medquad_ans_0005135-1` (score=0.0001) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
5. `medquad_ans_0000069-1__dup1` (score=0.0001) - What are the signs and symptoms of Acrocapitofemoral dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Acrocapitofemoral dysplasia. If the information is available, the tabl...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include abnormal form of the vertebral bodies (90%), abnormality of epiphysis morphology (90%), micromelia (90%), narrow chest (90%), short neck (90%), short stature (90%), short thorax (90%), skeletal dysplasia (90%), broad forehead (50%), cleft palate (50%), hyperlordosis (50%), hypertelorism (50%), malar flattening (50%), osteoarthritis (50%), talipes (50%), cataract (7.5%), glaucoma (7.5%), hearing impairment (7.5%), kyphosis (7.5%), myopia (7.5%), nystagmus (7.5%), retinal detachment (7.5%), and scoliosis (7.5%)

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Scores**: context_relevance=1.000, groundedness=0.537, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005219-3` (score=0.9912) - What causes pyridoxine-dependent epilepsy? Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each ...
2. `medquad_ans_0000650-2` (score=0.0012) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
3. `medquad_ans_0000800-2` (score=0.0008) - This condition is very rare; only a few affected individuals have been reported worldwide.
4. `medquad_ans_0000874-2` (score=0.0007) - This condition affects an estimated 1 in 8,500 females.
5. `medquad_ans_0000955-2` (score=0.0007) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.

**Answer**:

Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The ALDH7A1 gene provides instructions for making an enzyme called α-aminoadipic semialdehyde (α-AASA) dehydrogenase, also known as antiquitin. This enzyme is involved in the breakdown of the protein building block (amino acid) lysine in the brain. When antiquitin is deficient, a molecule that interferes with vitamin B6 function builds up in various tissues. Pyridoxine plays a role in many processes in the body, such as the breakdown of amino acids and the production of chemicals that transmit signals in the brain (neurotransmitters). It is unclear how a lack of pyridoxine causes the seizures that are characteristic of this condition. Some individuals with pyridox

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Scores**: context_relevance=0.875, groundedness=0.929, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000650-2` (score=0.9990) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
2. `medquad_ans_0000073-3__dup1` (score=0.9980) - COFS is a fatal disease. Most children do not live beyond five years.
3. `medquad_ans_0000006_2-5` (score=0.9975) - Certain factors affect prognosis (chance of recovery). The prognosis (chance of recovery) depends on the following: - Whether there are any cancer cells left after surgery. - The type of tumor. - Where the tumor is in...
4. `medquad_ans_0000161-3__dup2` (score=0.9974) - IRD is a fatal disease, but some children will survive into their teens and twenties, and possibly even beyond.
5. `medquad_ans_0000004_6-2` (score=0.9949) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...

**Answer**:

Children who are infected with the Epstein-Barr virus, the human immunodeficiency virus (HIV), have certain diseases of the immune system, have a personal history of mononucleosis ("mono"), have a parent or sibling with a personal history of Hodgkin lymphoma, or were exposed to common infections in early childhood are at increased risk for childhood Hodgkin lymphoma [medquad_ans_0000004_6-2].

**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

### Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Scores**: context_relevance=0.800, groundedness=0.639, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000095-1__dup4` (score=0.8953) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation ...
2. `medquad_ans_0001795-1` (score=0.7929) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow...
3. `medquad_ans_0000796-2` (score=0.0024) - PRS superactivity is believed to be a rare disorder. Approximately 30 families with the condition have been reported. More than two thirds of these families are affected by the milder form of the disease.
4. `medquad_ans_0000102-3__dup1` (score=0.0018) - The progression of the disorder is usually 10 years or longer.
5. `medquad_ans_0000095-2__dup3` (score=0.0006) - Dermatitis herpetiformis is caused by the deposit of immunoglobulin A (IgA) in the skin, which triggers further immunologic reactions resulting in lesion formation. DH is an external manifestation of an abnormal immun...

**Answer**:

Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions, which are more common on the knees, elbows, buttocks, and shoulder blades [medquad_ans_0001795-1]. It is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp, and the face and groin can also be involved [medquad_ans_0000095-1]. A burning sensation may precede lesion formation, and lesions are usually scratched off by the time a patient comes in for a physical exam, appearing as erosions and excoriations [medquad_ans_0000095-1]. The condition is caused by the deposit of immunoglobulin A (IgA) in the skin, triggering immunologic reactions that result in

**Supported citations**: `medquad_ans_0001795-1`
**Unsupported citations**: `medquad_ans_0000095-1`

### Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Scores**: context_relevance=0.714, groundedness=0.822, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000209-3` (score=0.9968) - With treatment, muscle weakness may improve or be reversed.
2. `medquad_ans_0000293-5` (score=0.9957) - These resources address the diagnosis or management of distal myopathy 2: - Genetic Testing Registry: Myopathy, distal, 2 - MedlinePlus Encyclopedia: Muscular Dystrophy - National Institute of Neurological Disorders a...
3. `medquad_ans_0000208-2__dup1` (score=0.9928) - Treatments for the myopathies depend on the disease or condition and specific causes. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders. Treatment for other disor...
4. `medquad_ans_0000709-5` (score=0.9926) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Regi...
5. `medquad_ans_0000575-5` (score=0.9907) - These resources address the diagnosis or management of Laing distal myopathy: - Gene Review: Gene Review: Laing Distal Myopathy - Genetic Testing Registry: Myopathy, distal, 1 These resources from MedlinePlus offer in...

**Answer**:

Treatments for myotonia congenita may include drug therapy, as indicated by the resources from MedlinePlus that cover diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care [medquad_ans_0000709-5]. Specific treatments are not detailed in the provided context.

**Supported citations**: `medquad_ans_0000709-5`
**Unsupported citations**: _none_

### Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Scores**: context_relevance=1.000, groundedness=0.464, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000133-13` (score=0.8860) - - Wilson disease is a genetic disease that prevents the body from removing extra copper. - Normally, the liver filters extra copper and releases it into bile. In Wilson disease, the liver does not filter copper correc...
2. `medquad_ans_0000133-11` (score=0.8679) - People with Wilson disease should reduce their dietary copper intake by avoiding foods that are high in copper, such as - shellfish - liver - mushrooms - nuts - chocolate People should not eat these foods during the i...
3. `medquad_ans_0000133-12` (score=0.0097) - A person cannot prevent Wilson disease; however, people with a family history of Wilson disease, especially those with an affected sibling or parent, should talk with a health care provider about testing. A health car...
4. `medquad_ans_0000650-2` (score=0.0036) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
5. `medquad_ans_0000276-3__dup1` (score=0.0021) - Early onset of the disease may foretell a worse prognosis than later onset. If the disorder is detected early and treated appropriately, an individual with WD can usually enjoy normal health and a normal lifespan. If ...

**Answer**:

For Wilson disease, treatment involves a lifelong effort to reduce and control copper levels in the body. This may include medications, changes in eating, diet, and nutrition, and in some cases, a liver transplant. People should reduce dietary copper intake by avoiding high-copper foods such as shellfish, liver, mushrooms, nuts, and chocolate, especially during initial treatment; however, they should consult a health care provider to determine if these foods can be consumed in moderation during maintenance treatment. Additionally, individuals should check the copper levels in their tap water, as water from copper pipes or wells may contain copper residue; running water helps lower copper levels. People should not use copper containers or cookware to store or prepare food or drinks. It is also important to discuss the use of complementary and alternative medical practices, including vitamins and dietary supplements, with a health care provider, as some supplements contain copper. If a supplement is recommended, a pharmacist can suggest copper-free options.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.020, answer_relevance=0.756

**Top docs**:

1. `medquad_ans_0000117-3` (score=0.9994) - Mutations in two genes, KCNQ2 and KCNQ3, have been found to cause BFNS. Mutations in the KCNQ2 gene are a much more common cause of the condition than mutations in the KCNQ3 gene. The KCNQ2 and KCNQ3 genes provide ins...
2. `medquad_ans_0000390-3` (score=0.6480) - Mutations in the ALX3 gene cause frontonasal dysplasia type 1, ALX4 gene mutations cause type 2, and ALX1 gene mutations cause type 3. These genes provide instructions for making proteins that are necessary for normal...
3. `medquad_ans_0000373-3` (score=0.0134) - Researchers have identified changes in five regions of the X chromosome that are linked to FG syndrome in affected families. Mutations in a gene called MED12, which is located in one of these regions, appear to be the...
4. `medquad_ans_0000983-3` (score=0.0009) - Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. This gene provides instructions for making a protein that acts as a channel across cell membranes. This channel, known as CaV1....
5. `medquad_ans_0000174-3` (score=0.0003) - Mutations in the CHD7 gene cause more than half of all cases of CHARGE syndrome. The CHD7 gene provides instructions for making a protein that most likely regulates gene activity (expression) by a process known as chr...

**Answer**:

The provided context does not contain any information about genetic changes related to Lennox-Gastaut syndrome. Therefore, it is not possible to answer the question based on the given context. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Scores**: context_relevance=0.100, groundedness=0.847, answer_relevance=0.420

**Top docs**:

1. `medquad_ans_0001101-2` (score=0.9996) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
2. `medquad_ans_0000989-2` (score=0.9996) - This condition affects an estimated 1 in 50,000 people.
3. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
4. `medquad_ans_0000874-2` (score=0.9995) - This condition affects an estimated 1 in 8,500 females.
5. `medquad_ans_0000904-2` (score=0.9995) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.

**Answer**:

The exact incidence of arginine:glycine amidinotransferase deficiency is unknown, but it is estimated to affect 1 in 87,000 people [medquad_ans_0000484-2].

**Supported citations**: `medquad_ans_0000484-2`
**Unsupported citations**: _none_

### Example 231 (`0002399-2`)

**Question**: What are the symptoms of Focal dermal hypoplasia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0002399-2` (score=0.9990) - What are the signs and symptoms of Focal dermal hypoplasia? Focal dermal hypoplasia is usually evident from birth and primarily affects the skin, skeleton, eyes, and face. The signs and symptoms of vary widely, althou...
2. `medquad_ans_0002401-1` (score=0.0004) - What are the signs and symptoms of Focal facial dermal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Focal facial dermal dysplasia. If the information is available, the ...
3. `medquad_ans_0006149-1` (score=0.0001) - What are the signs and symptoms of Transient bullous dermolysis of the newborn? The Human Phenotype Ontology provides the following list of signs and symptoms for Transient bullous dermolysis of the newborn. If the in...
4. `medquad_ans_0000946-2__dup1` (score=0.0001) - What are the signs and symptoms of Cardiofaciocutaneous syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Cardiofaciocutaneous syndrome. If the information is available, the ...
5. `medquad_ans_0002727-1` (score=0.0001) - What are the signs and symptoms of Hard skin syndrome Parana type? The Human Phenotype Ontology provides the following list of signs and symptoms for Hard skin syndrome Parana type. If the information is available, th...

**Answer**:

The symptoms of Focal dermal hypoplasia include skin abnormalities such as streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), telangiectases, and streaks of slightly darker or lighter skin. These can cause pain, itching, irritation, or lead to skin infections. With age, most develop wart-like growths (papillomas) around the nostrils, lips, anus, and female genitalia, and may also be present in the throat, causing problems with swallowing, breathing, or sleeping. Other features include small, ridged fingernails and toenails, sparse or brittle scalp hair, hand and foot abnormalities such as missing fingers or toes (oligodactyly), webbed or fused fingers or toes (syndactyly), and a deep split in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Scores**: context_relevance=0.091, groundedness=0.000, answer_relevance=0.818

**Top docs**:

1. `medquad_ans_0000650-2` (score=0.9981) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
2. `medquad_ans_0000790-2` (score=0.9971) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
3. `medquad_ans_0000874-2` (score=0.9969) - This condition affects an estimated 1 in 8,500 females.
4. `medquad_ans_0000989-2` (score=0.9968) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000359-2` (score=0.9946) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.

**Answer**:

The provided context does not contain information about who is at risk for Parasites - Trichinellosis (also known as Trichinosis). Therefore, based on the given context, the risk population for Trichinellosis cannot be determined. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001924-2` (score=0.9994) - What are the signs and symptoms of Dravet syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Dravet syndrome. If the information is available, the table below includes how oft...
2. `medquad_ans_0002138-1` (score=0.0001) - What are the signs and symptoms of Epilepsy progressive myoclonic type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Epilepsy progressive myoclonic type 3. If the information is...
3. `medquad_ans_0005014-1` (score=0.0001) - What are the signs and symptoms of PPM-X syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for PPM-X syndrome. If the information is available, the table below includes how often...
4. `medquad_ans_0004898-1` (score=0.0001) - What are the signs and symptoms of Pitt-Hopkins-like syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pitt-Hopkins-like syndrome. If the information is available, the table ...
5. `medquad_ans_0005556-1` (score=0.0001) - What are the signs and symptoms of SeSAME syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for SeSAME syndrome. If the information is available, the table below includes how oft...

**Answer**:

The signs and symptoms of Dravet syndrome include absence seizures, ataxia, autosomal dominant inheritance, cerebral atrophy, cortical visual impairment, epileptic encephalopathy, focal seizures with impairment of consciousness or awareness, generalized myoclonic seizures, hemiclonic seizures, infantile onset, mental deterioration, motor delay, postnatal microcephaly, and status epilepticus [medquad_ans_0001924-2].

**Supported citations**: `medquad_ans_0001924-2`
**Unsupported citations**: _none_

### Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Scores**: context_relevance=0.889, groundedness=0.965, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000826-3` (score=0.9974) - Between 10 and 15 percent of all cases of prion disease are caused by mutations in the PRNP gene. Because they can run in families, these forms of prion disease are classified as familial. Familial prion diseases, whi...
2. `medquad_ans_0000013-9__dup1` (score=0.0074) - A heritable disease is caused by a mutation, or harmful change, in a gene inherited from a parent. Genes are small structures in your body's cells that determine how you look and tell your body how to work. Examples o...
3. `medquad_ans_0000276-3` (score=0.0019) - DRPLA is caused by a mutation in the ATN1 gene. This gene provides instructions for making a protein called atrophin 1. Although the function of atrophin 1 is unclear, it likely plays an important role in nerve cells ...
4. `medquad_ans_0001037-3` (score=0.0017) - Wilson disease is caused by mutations in the ATP7B gene. This gene provides instructions for making a protein called copper-transporting ATPase 2, which plays a role in the transport of copper from the liver to other ...
5. `medquad_ans_0000441-3` (score=0.0014) - GRN-related frontotemporal dementia results from mutations in the GRN gene. This gene provides instructions for making a protein called granulin (also known as progranulin). Granulin is active in many different tissue...

**Answer**:

The genetic changes related to prion disease are mutations in the PRNP gene. Between 10 and 15 percent of all cases of prion disease are caused by such mutations, which result in the production of an abnormally shaped protein called PrPSc from one copy of the gene. These mutations lead to familial forms of prion disease, including familial Creutzfeldt-Jakob disease (CJD), Gerstmann-Strussler-Scheinker syndrome (GSS), and fatal familial insomnia (FFI) [medquad_ans_0000826-3].

**Supported citations**: `medquad_ans_0000826-3`
**Unsupported citations**: _none_

### Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0006430-1` (score=0.9994) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information ...
2. `medquad_ans_0004067-1` (score=0.0001) - What are the signs and symptoms of Miles-Carpenter x-linked mental retardation syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Miles-Carpenter x-linked mental retardation s...
3. `medquad_ans_0005971-1` (score=0.0001) - What are the signs and symptoms of Syndactyly-polydactyly-earlobe syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly-polydactyly-earlobe syndrome. If the informatio...
4. `medquad_ans_0003657-1` (score=0.0001) - What are the signs and symptoms of Loeys-Dietz syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Loeys-Dietz syndrome type 2. If the information is available, the tabl...
5. `medquad_ans_0003655-2` (score=0.0001) - What are the signs and symptoms of Loeys-Dietz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Loeys-Dietz syndrome. If the information is available, the table below includ...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, cleft palate, cleft upper lip, high palate, hydronephrosis, hypoplasia of the radius, hypotelorism, long face, mesomelia, narrow face, oligodactyly (hands), proximal placement of thumb, and proximal radial head dislocation [medquad_ans_0006430-1].

**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

### Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Scores**: context_relevance=1.000, groundedness=0.883, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.9994) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the ta...
2. `medquad_ans_0004524-1` (score=0.0001) - What are the signs and symptoms of Okamoto syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Okamoto syndrome. If the information is available, the table below includes how o...
3. `medquad_ans_0004615-2` (score=0.0001) - What are the signs and symptoms of Osteogenesis imperfecta type III? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteogenesis imperfecta type III. If the information is available...
4. `medquad_ans_0005253-1` (score=0.0001) - What are the signs and symptoms of Ramos Arroyo Clark syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Ramos Arroyo Clark syndrome. If the information is available, the tabl...
5. `medquad_ans_0004954-2` (score=0.0001) - What are the signs and symptoms of Polyglucosan body disease, adult? The Human Phenotype Ontology provides the following list of signs and symptoms for Polyglucosan body disease, adult. If the information is available...

**Answer**:

The symptoms of Premature aging Okamoto type include abnormality of the blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face [medquad_ans_0005029-1]. The frequency of these symptoms is based on data from the Human Phenotype Ontology, which compiles information from sources such as Orphanet, with frequencies often expressed as percentages or fractions representing the proportion of patients exhibiting each symptom [medquad_ans_0005029-1].

**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

### Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Scores**: context_relevance=0.111, groundedness=0.713, answer_relevance=0.378

**Top docs**:

1. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
2. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
3. `medquad_ans_0000874-2` (score=0.9995) - This condition affects an estimated 1 in 8,500 females.
4. `medquad_ans_0000091-2` (score=0.9995) - This condition is rare, affecting fewer than 1 per million people.
5. `medquad_ans_0000650-2` (score=0.9995) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

The exact incidence of thrombocytopenia-absent radius syndrome is unknown, but it is estimated to affect 1 in 87,000 people [medquad_ans_0000484-2]. Additionally, it is estimated to affect 1 in 50,000 to 100,000 people [medquad_ans_0000650-2]. Another estimate suggests it affects 1 in 8,500 females [medquad_ans_0000874-2]. However, one source notes it is rare, affecting fewer than 1 per million people [medquad_ans_0000091-2]. Due to the variation in estimates, the most specific available range is 1 in 50,000 to 100,000 people [medquad_ans_0

**Supported citations**: `medquad_ans_0000484-2`, `medquad_ans_0000650-2`, `medquad_ans_0000874-2`, `medquad_ans_0000091-2`
**Unsupported citations**: _none_

### Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Scores**: context_relevance=0.455, groundedness=0.000, answer_relevance=0.673

**Top docs**:

1. `medquad_ans_0000214-4__dup1` (score=0.9984) - The NINDS conducts and supports a wide range of studies that explore the complex systems of brain development. These studies include the identification of the mechanism of action of the known causes of NMD as well as ...
2. `medquad_ans_0000012-4__dup1` (score=0.9984) - The National Institute of Neurological Disorders and Stroke (NINDS) and other institutes of the National Institutes of Health (NIH) support research related to AGS through grants to major medical institutions across t...
3. `medquad_ans_0000140-4__dup1` (score=0.9983) - The National Institute of Neurological Disorders and Stroke (NINDS) and other institutes at the National Institutes of Health conduct research related to GN and support additional research through grants to major rese...
4. `medquad_ans_0000137-4__dup1` (score=0.9983) - The NINDS supports and conducts research on TSEs, including GSS. Much of this research is aimed at characterizing the agents that cause these disorders, clarifying the mechanisms underlying them, and, ultimately, find...
5. `medquad_ans_0000227-4__dup1` (score=0.9983) - NINDS supports research on gene-linked disorders, including the leukodystrophies. The goals of this research are to increase scientific understanding of these disorders and to find ways to prevent, treat, and ultimate...

**Answer**:

The provided context does not mention research or clinical trials specifically for Tay-Sachs Disease. Therefore, based on the given information, it is not possible to answer the question. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000008-1` (score=0.4357) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achro...
2. `medquad_ans_0000055-3__dup2` (score=0.0006) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision -...
3. `medquad_ans_0000008-2` (score=0.0004) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on o...
4. `medquad_ans_0003512-1` (score=0.0001) - What are the signs and symptoms of Leber congenital amaurosis 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Leber congenital amaurosis 2. If the information is available, the ta...
5. `medquad_ans_0000763-2__dup1` (score=0.0001) - What are the signs and symptoms of Blue cone monochromatism? The Human Phenotype Ontology provides the following list of signs and symptoms for Blue cone monochromatism. If the information is available, the table belo...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. Achromatopsia also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals may have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia is different from common forms of color vision deficiency (color blindness), in which people can perceive color but have difficulty distinguishing between certain colors, such as red and green [medquad_ans_00

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Scores**: context_relevance=0.143, groundedness=0.000, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000945-5` (score=0.9967) - These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliative Care
2. `medquad_ans_0000763-5` (score=0.9951) - These resources address the diagnosis or management of OSMED: - Genetic Testing Registry: Otospondylomegaepiphyseal dysplasia These resources from MedlinePlus offer information about the diagnosis and management of va...
3. `medquad_ans_0000217-2` (score=0.9942) - Treatment involves eliminating or reducing exposure to the toxic substance, followed by symptomatic and supportive therapy.
4. `medquad_ans_0000029-2__dup1` (score=0.9940) - Medications may ease certain symptoms, such as pain. Surgery is the only treatment available to correct functional disturbances or halt the progression of damage to the central nervous system. More than one surgery ma...
5. `medquad_ans_0000172-2__dup1` (score=0.9921) - Treatment is symptomatic and supportive, and may include drug therapy.

**Answer**:

The provided context does not contain information about the treatments for Knee Replacement. Therefore, a specific answer cannot be generated based on the given resources. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.694, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0004124-1` (score=0.0034) - What are the signs and symptoms of MORM syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for MORM syndrome. If the information is available, the table below includes how often t...
2. `medquad_ans_0002359-2` (score=0.0003) - What are the signs and symptoms of FG syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for FG syndrome. If the information is available, the table below includes how often the s...
3. `medquad_ans_0003152-1` (score=0.0001) - What are the signs and symptoms of IMAGe syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IMAGe syndrome. If the information is available, the table below includes how often...
4. `medquad_ans_0001692-2` (score=0.0001) - What are the signs and symptoms of Cushing's syndrome? The signs and symptoms of Cushing's syndrome may include: Upper body obesity Severe fatigue Muscle weakness High blood pressure Backache Elevated blood sugar Easy...
5. `medquad_ans_0000886-2__dup1` (score=0.0001) - What are the signs and symptoms of C syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for C syndrome. If the information is available, the table below includes how often the sym...

**Answer**:

The provided context does not contain information about the symptoms of Metabolic Syndrome. The context discusses MORM syndrome and FG syndrome, not Metabolic Syndrome. [medquad_ans_0004124-1], [medquad_ans_0002359-2]

**Supported citations**: `medquad_ans_0004124-1`, `medquad_ans_0002359-2`
**Unsupported citations**: _none_

### Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Scores**: context_relevance=1.000, groundedness=0.957, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005211-1` (score=0.9994) - What are the signs and symptoms of Pyknoachondrogenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyknoachondrogenesis. If the information is available, the table below includ...
2. `medquad_ans_0000393-2__dup1` (score=0.0001) - What are the signs and symptoms of Antecubital pterygium? The Human Phenotype Ontology provides the following list of signs and symptoms for Antecubital pterygium. If the information is available, the table below incl...
3. `medquad_ans_0006103-1` (score=0.0001) - What are the signs and symptoms of Tibia absent polydactyly arachnoid cyst? The Human Phenotype Ontology provides the following list of signs and symptoms for Tibia absent polydactyly arachnoid cyst. If the informatio...
4. `medquad_ans_0005818-2` (score=0.0001) - What are the signs and symptoms of Spondylocarpotarsal synostosis syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylocarpotarsal synostosis syndrome. If the informatio...
5. `medquad_ans_0004979-1` (score=0.0001) - What are the signs and symptoms of Popliteal pterygium syndrome lethal type? The Human Phenotype Ontology provides the following list of signs and symptoms for Popliteal pterygium syndrome lethal type. If the informat...

**Answer**:

The symptoms of Pyknoachondrogenesis include abnormal vertebral ossification, abnormality of pelvic girdle bone morphology, abnormality of the mouth, abnormality of the sacrum, depressed nasal ridge, enlarged thorax, increased bone mineral density, low-set, posteriorly rotated ears, macrocephaly, micromelia, palpebral edema, premature birth, short stature, short thorax, thickened nuchal skin fold, and stillbirth. These symptoms are reported in approximately 90% of patients with this condition [medquad_ans_0005211-1].

**Supported citations**: `medquad_ans_0005211-1`
**Unsupported citations**: _none_

### Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.543, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003965-2` (score=0.9993) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the ta...
2. `medquad_ans_0003966-1` (score=0.0163) - What are the signs and symptoms of Metachromatic leukodystrophy due to saposin B deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy due to sapos...
3. `medquad_ans_0000340-1__dup1` (score=0.0001) - What are the signs and symptoms of Anemia sideroblastic and spinocerebellar ataxia? The Human Phenotype Ontology provides the following list of signs and symptoms for Anemia sideroblastic and spinocerebellar ataxia. I...
4. `medquad_ans_0004944-1` (score=0.0001) - What are the signs and symptoms of Polycystic lipomembranous osteodysplasia with sclerosing leukoencephalopathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Polycystic lipomembra...
5. `medquad_ans_0005790-1` (score=0.0001) - What are the signs and symptoms of Spinocerebellar ataxia autosomal recessive 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia autosomal recessive 5. If the ...

**Answer**:

The symptoms of Metachromatic leukodystrophy include behavioral abnormality, cognitive impairment, decreased nerve conduction velocity, developmental regression, gait disturbance, genu recurvatum, incoordination, muscle weakness, neurological speech impairment, peripheral neuropathy, reduced consciousness/confusion, seizures, amaurosis fugax, hyperreflexia, hypertonia, limitation of joint mobility, muscular hypotonia, nystagmus, optic atrophy, aganglionic megacolon, ataxia, Babinski sign, bulbar palsy, cholecystitis, chorea, delusions, dysarthria, dystonia, EMG: neuropathic changes, emotional lability, gallbladder dysfunction, hallucinations, hyporeflexia, increased CSF protein, intellectual disability, loss of speech, mental deterioration, peripheral demyelination, progressive peripheral neuropathy, spastic tetraplegia, tetraplegia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0001014-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000111-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000677-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000685-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention chronic granulomatous disease or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether chronic granulomatous disease is inherited.

Answer: Insufficient context [medquad_ans_0000785-4, medquad_ans_0001014-4, medquad_ans_0000111-4, medquad_ans_0000677-4, medquad_ans_0000685-4].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000785-4, medquad_ans_0001014-4, medquad_ans_0000111-4, medquad_ans_0000677-4, medquad_ans_0000685-4`

### Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000255-3__dup1` (score=0.9984) - The prognosis for those with spasticity depends on the severity of the spasticity and the associated disorder(s).
2. `medquad_ans_0000222-3__dup1` (score=0.9984) - The prognosis for those with paresthesia depends on the severity of the sensations and the associated disorders.
3. `medquad_ans_0000217-3` (score=0.9983) - The prognosis depends upon the length and degree of exposure and the severity of neurological injury. In some instances, exposure to neurotoxicants can be fatal. In others, patients may survive but not fully recover. ...
4. `medquad_ans_0000359-2` (score=0.9981) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
5. `medquad_ans_0000087-3__dup1` (score=0.9980) - The prognosis is dependent upon the specific syndrome, however, some of the syndromes are fatal if left untreated.

**Answer**:

The provided context does not contain information about the outlook for Central Cord Syndrome. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.631, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000094-3__dup3` (score=0.9953) - The cause of cyclic vomiting syndrome is unknown. However, some experts believe that some possible problems with bodily functions may contribute to the cause, such as the following: - gastrointestinal motilitythe way ...
2. `medquad_ans_0000094-12` (score=0.0025) - During the prodrome and vomiting phases of cyclic vomiting syndrome, a person will generally take in little or no nutrition by mouth. During the recovery phase, the person may be quite hungry as soon as the vomiting s...
3. `medquad_ans_0000261-3` (score=0.0017) - Although the causes of cyclic vomiting syndrome have yet to be determined, researchers have proposed several factors that may contribute to the disorder. These factors include changes in brain function, hormonal abnor...
4. `medquad_ans_0000094-11` (score=0.0004) - A person should stay away from known triggers, especially during the well phase, as well as - get adequate sleep to prevent exhaustion - treat sinus problems or allergies - seek help on reducing stress and anxiety - a...
5. `medquad_ans_0000650-2` (score=0.0004) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

The cause of cyclic vomiting syndrome is unknown. However, some experts believe that possible problems with bodily functions may contribute to the cause, such as gastrointestinal motility, central nervous system function, autonomic nervous system function, hormone imbalances, and in children, an abnormal inherited gene [medquad_ans_0000094-3__dup3]. Researchers have also proposed that the condition may be related to changes in brain function, hormonal abnormalities, and gastrointestinal problems, and many believe it is a migraine-like condition involving changes in signaling between nerve cells in certain areas of the brain [medquad_ans_00000261-3].

**Supported citations**: `medquad_ans_0000094-3__dup3`
**Unsupported citations**: `medquad_ans_00000261-3`

### Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Scores**: context_relevance=0.750, groundedness=0.710, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000414-1__dup2` (score=0.9983) - Frequently Asked Questions (FAQs) Fact Sheets
2. `medquad_ans_0000189-1__dup2` (score=0.9981) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...
3. `medquad_ans_0000269-1__dup4` (score=0.9980) - Frequently Asked Questions (FAQs) Vector Information
4. `medquad_ans_0000001-7__dup1` (score=0.9972) - Topics
5. `medquad_ans_0000797-1__dup1` (score=0.9540) - Summary : Your child spends more time at school than anywhere else except home. Schools can have a major effect on children's health. Schools can teach children about health, and promote healthy behaviors. Physical ed...

**Answer**:

The provided context does not contain specific information about child mental health. While it mentions that "your child's health includes physical, mental and social well-being" [medquad_ans_0000189-1__dup2], no further details about mental health specifically are given. Therefore, based on the available information, there is no explicit information about child mental health.

**Supported citations**: `medquad_ans_0000189-1__dup2`
**Unsupported citations**: _none_

### Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.536, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001618-2` (score=0.9976) - What are the signs and symptoms of Cowden syndrome? Cowden syndrome is characterized primarily by multiple, noncancerous growths (called hamartomas) on various parts of the body. Approximately 99% of people affected b...
2. `medquad_ans_0003718-2` (score=0.0001) - What are the signs and symptoms of Lynch syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lynch syndrome. If the information is available, the table below includes how often...
3. `medquad_ans_0005115-2` (score=0.0001) - What are the signs and symptoms of Proteus syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Proteus syndrome. If the information is available, the table below includes how o...
4. `medquad_ans_0000964-1__dup1` (score=0.0001) - What are the signs and symptoms of Carney triad? The Human Phenotype Ontology provides the following list of signs and symptoms for Carney triad. If the information is available, the table below includes how often the...
5. `medquad_ans_0003528-1` (score=0.0001) - What are the signs and symptoms of Legius syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Legius syndrome. If the information is available, the table below includes how oft...

**Answer**:

The symptoms of Cowden syndrome include multiple noncancerous growths (hamartomas) on the skin and/or in the mouth, with approximately 99% of affected individuals having these by the third decade of life. A majority also develop hamartomatous polyps in the gastrointestinal tract. Other common symptoms include benign diseases of the breast, thyroid, and endometrium; a rare noncancerous brain tumor called Lhermitte-Duclos disease; enlarged head (macrocephaly); autism spectrum disorder; intellectual disability; and vascular abnormalities. Specific signs and symptoms reported in the Human Phenotype Ontology include abnormality of the pupil, abnormality of the tongue, aplasia/hypoplasia of the cerebellum, arteriovenous malformation, cognitive impairment, conjunctival hamartoma, dental malocclusion, epibulbar dermoid, exostoses, foot polydactyly,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Scores**: context_relevance=0.333, groundedness=0.016, answer_relevance=0.689

**Top docs**:

1. `medquad_ans_0000945-5` (score=0.9974) - These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliative Care
2. `medquad_ans_0000209-3` (score=0.9969) - With treatment, muscle weakness may improve or be reversed.
3. `medquad_ans_0000173-2__dup2` (score=0.9962) - There is no cure for SMA. Treatment consists of managing the symptoms and preventing complications.
4. `medquad_ans_0000201-3__dup1` (score=0.9956) - Improvement in muscle strength usually begins within 3 to 6 weeks after treatment is started. Most patients who receive treatment early experience little, if any, disability. However, there is evidence of slow progres...
5. `medquad_ans_0000208-2__dup1` (score=0.9941) - Treatments for the myopathies depend on the disease or condition and specific causes. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders. Treatment for other disor...

**Answer**:

The provided context does not contain specific information about the treatments for Fukuyama congenital muscular dystrophy. Therefore, based on the given resources, it is not possible to answer the question. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.805, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000004_7-3` (score=0.9978) - Tests that examine the body and lymph system are used to detect (find) and diagnose childhood non-Hodgkin lymphoma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to ...
2. `medquad_ans_0000004_6-4` (score=0.9976) - Tests that examine the lymph system are used to detect (find) and diagnose childhood Hodgkin lymphoma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general...
3. `medquad_ans_0000006_7-5` (score=0.9750) - Imaging studies and tests are used to detect (find) and diagnose childhood CNS germ cell tumors. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general signs...
4. `medquad_ans_0000003_6-2` (score=0.1201) - Tests are used to detect (find) and diagnose childhood vascular tumors. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general signs of health, including che...
5. `medquad_ans_0000007_3-4` (score=0.0313) - Tests that examine the liver and the blood are used to detect (find) and diagnose childhood liver cancer and find out whether the cancer has spread. The following tests and procedures may be used: - Physical exam and ...

**Answer**:

The provided context does not contain information on how to diagnose Childhood Hodgkin Lymphoma. It only discusses tests used to detect and diagnose childhood non-Hodgkin lymphoma [medquad_ans_0000004_7-3]. Therefore, based on the given context, the diagnosis of Childhood Hodgkin Lymphoma cannot be determined.

**Supported citations**: `medquad_ans_0000004_7-3`
**Unsupported citations**: _none_

### Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000572-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000857-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000789-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0001074-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention KBG syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether KBG syndrome is inherited.

Answer: Insufficient context [medquad_ans_0000785-4, medquad_ans_0000572-4, medquad_ans_0000857-4, medquad_ans_0000789-4, medquad_ans_0001074-4].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000785-4, medquad_ans_0000572-4, medquad_ans_0000857-4, medquad_ans_0000789-4, medquad_ans_0001074-4`

### Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Scores**: context_relevance=0.667, groundedness=0.533, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000012-1__dup2` (score=0.9683) - Aicardi-Goutieres syndrome (AGS) is an inherited encephalopathy that affects newborn infants and usually results in severe mental and physical handicap. There are two forms of the syndrome: an early-onset form that is...
2. `medquad_ans_0000194-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
3. `medquad_ans_0000196-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000193-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
5. `medquad_ans_0000195-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...

**Answer**:

Aicardi-Goutieres syndrome (AGS) is an inherited encephalopathy that affects newborn infants and usually results in severe mental and physical handicap. There are two forms of the syndrome: an early-onset form that is severe and a late-onset form that has less impact upon neurological function. The early-onset form affects about 20 percent of all babies who have AGS and is characterized by neurological and liver abnormalities, such as enlargement of the liver and spleen and elevated liver enzymes. Jittery behavior and poor feeding ability mimic congenital viral infection. The later-onset form begins after the first weeks or months of normal development, with symptoms including progressive decline in head growth, weak or stiffened muscles (spasticity), cognitive and developmental delays, irritability, inconsolable crying, intermittent fever, seizures, loss of developmental skills, puffy swelling on the fingers, toes, and ears resembling chilblains,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Scores**: context_relevance=0.750, groundedness=0.067, answer_relevance=0.817

**Top docs**:

1. `medquad_ans_0000008-30` (score=0.9967) - Clinical trials are research studies on people to find out whether a new drug or treatment is both safe and effective. New therapies are tested on people only after laboratory and animal studies show promising results...
2. `medquad_ans_0000003_6-4` (score=0.9965) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
3. `medquad_ans_0000007_1-7` (score=0.9962) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
4. `medquad_ans_0000024_5-8` (score=0.9962) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
5. `medquad_ans_0000001_5-7` (score=0.9962) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...

**Answer**:

The provided context does not mention any specific research or clinical trials being conducted for Renal Cell Cancer. Therefore, based on the given information, it is not possible to answer the question. [doc_id] is not applicable due to lack of relevant information.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Scores**: context_relevance=0.714, groundedness=0.040, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000190-8` (score=0.9898) - A healthy diet is important in all stages of cirrhosis because malnutrition is common in people with this disease. Malnutrition is a condition that occurs when the body does not get enough nutrients. Cirrhosis may lea...
2. `medquad_ans_0000184-8` (score=0.9898) - A healthy diet is important in all stages of cirrhosis because malnutrition is common in people with this disease. Malnutrition is a condition that occurs when the body does not get enough nutrients. Cirrhosis may lea...
3. `medquad_ans_0000191-8` (score=0.9736) - - Primary sclerosing cholangitis (PSC) inflames, scars, and blocks bile ducts inside and outside the liver. - When bile ducts become blocked, bile builds up in the liver and damages liver cells. - PSC can lead to vita...
4. `medquad_ans_0000193-9` (score=0.9567) - - Cirrhosis is scarring of the liver. Scar tissue replaces healthy liver tissue. - Some common causes of cirrhosis include heavy alcohol use, hepatitis infections, and nonalcoholic fatty liver disease. - In the early ...
5. `medquad_ans_0000190-9` (score=0.9146) - - Primary biliary cirrhosis is a chronic disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. - When chronic inflammation damages the bile ducts, bile and toxi...

**Answer**:

The provided context does not contain information about what to do for Primary Biliary Cirrhosis. The context discusses cirrhosis in general and primary sclerosing cholangitis (PSC), but it does not address Primary Biliary Cirrhosis specifically. Therefore, based on the given information, the question cannot be answered.

[doc_id] is not applicable due to insufficient context.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Scores**: context_relevance=0.750, groundedness=0.950, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000193-1__dup3` (score=0.9860) - Food or small objects can cause choking if they get caught in your throat and block your airway. This keeps oxygen from getting to your lungs and brain. If your brain goes without oxygen for more than four minutes, yo...
2. `medquad_ans_0000299-1__dup2` (score=0.9778) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People ...
3. `medquad_ans_0000243-1__dup1` (score=0.3722) - Summary : When someone's blood flow or breathing stops, seconds count. Permanent brain damage or death can happen quickly. If you know how to perform cardiopulmonary resuscitation (CPR), you could save a life. CPR is ...
4. `medquad_ans_0000169-3` (score=0.0320) - Episodes eventually decrease in frequency and intensity over the course of eight to 12 years.
5. `medquad_ans_0000379-1__dup1` (score=0.0149) - If you've ever gotten a splinter or had sand in your eye, you've had experience with a foreign body. A foreign body is something that is stuck inside you but isn't supposed to be there. You may inhale or swallow a for...

**Answer**:

Drowning occurs when a person gets too much water in their lungs. It can happen in as little as an inch or two of water. Babies can drown in a sink or bathtub, and preschoolers are most likely to drown in a swimming pool. People with seizure disorders are also at risk in the water. Drowning can happen quickly and silently [medquad_ans_0000299-1__dup2].

**Supported citations**: `medquad_ans_0000299-1__dup2`
**Unsupported citations**: _none_

### Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Scores**: context_relevance=0.857, groundedness=0.991, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0002058-1` (score=0.9604) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragi...
2. `medquad_ans_0002049-1` (score=0.8987) - Ehlers-Danlos syndrome (EDS) is a group of inherited connective tissue disorders that is caused by abnormalities in the structure, production, and/or processing of collagen. There are 6 major forms of EDS: hypermobili...
3. `medquad_ans_0000313-1` (score=0.2940) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and sympt...
4. `medquad_ans_0000313-2` (score=0.0003) - Although it is difficult to estimate the overall frequency of Ehlers-Danlos syndrome, the combined prevalence of all types of this condition may be about 1 in 5,000 individuals worldwide. The hypermobility and classic...
5. `medquad_ans_0002049-5` (score=0.0001) - How is Ehlers-Danlos syndrome diagnosed? A diagnosis of Ehlers-Danlos syndrome is typically based on the presence of characteristic signs and symptoms. Depending on the subtype suspected, some of the following tests m...

**Answer**:

Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder caused by defects in a protein called collagen. It is caused by changes (mutations) in the PLOD1 gene and is inherited in an autosomal recessive manner. Common signs and symptoms include hyperextensible skin that is fragile and bruises easily, joint hypermobility, severe hypotonia at birth, progressive kyphoscoliosis (kyphosis and scoliosis), and fragility of the sclera. Treatment is focused on preventing serious complications and relieving associated signs and symptoms [medquad_ans_0002058-1].

**Supported citations**: `medquad_ans_0002058-1`
**Unsupported citations**: _none_

### Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.971, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001860-2` (score=0.9994) - What are the signs and symptoms of Dihydropteridine reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropteridine reductase deficiency. If the information is...
2. `medquad_ans_0001862-2` (score=0.0008) - What are the signs and symptoms of Dihydropyrimidine dehydrogenase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropyrimidine dehydrogenase deficiency. If the info...
3. `medquad_ans_0000706-2__dup1` (score=0.0001) - What are the signs and symptoms of Beta-thalassemia? The Human Phenotype Ontology provides the following list of signs and symptoms for Beta-thalassemia. If the information is available, the table below includes how o...
4. `medquad_ans_0006561-2` (score=0.0001) - What are the signs and symptoms of 17-beta hydroxysteroid dehydrogenase 3 deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for 17-beta hydroxysteroid dehydrogenase 3 deficienc...
5. `medquad_ans_0001902-1` (score=0.0001) - What are the signs and symptoms of Dopamine beta hydroxylase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dopamine beta hydroxylase deficiency. If the information is a...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include cognitive impairment (90%), microcephaly (90%), cerebral calcification, choreoathetosis, dysphagia, dystonia, episodic fever, excessive salivation, hyperphenylalaninemia, hypertonia, infantile onset, intellectual disability, irritability, muscular hypotonia, myoclonus, progressive neurologic deterioration, seizures, tremor, and variable expressivity [medquad_ans_0001860-2].

**Supported citations**: `medquad_ans_0001860-2`
**Unsupported citations**: _none_

### Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Scores**: context_relevance=0.143, groundedness=0.000, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0001101-2` (score=0.9994) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
2. `medquad_ans_0000989-2` (score=0.9994) - This condition affects an estimated 1 in 50,000 people.
3. `medquad_ans_0000484-2` (score=0.9994) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
4. `medquad_ans_0000195-2` (score=0.9994) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
5. `medquad_ans_0000650-2` (score=0.9994) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

The provided context does not contain any information about monilethrix. Therefore, it is not possible to determine how many people are affected by monilethrix based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Scores**: context_relevance=0.600, groundedness=0.544, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.3190) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...
2. `medquad_ans_0000874-2` (score=0.0111) - This condition affects an estimated 1 in 8,500 females.
3. `medquad_ans_0000171-3__dup1` (score=0.0098) - KTS is often a progressive disorder, and complications may be life-threatening. However, many individuals can live well while managing their symptoms.
4. `medquad_ans_0000989-2` (score=0.0092) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000650-2` (score=0.0059) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the results of newborn screening tests. Most people with the condition have no symptoms (asymptomatic). For those who do show symptoms, the onset of the disease is usually between the ages of 3 and 9; occasionally the disease may present in adulthood. Mental development is usually normal, though a few cases with intellectual impairment have been reported. The signs and symptoms of Hartnup disease include skin photosensitivity, neurologic findings, psychiatric symptoms, and ocular (eye) findings. Hartnup disease is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. People with Hartnup disease may benefit from a high-protein diet, protection from sunlight, vitamin supplementation, and avoidance of

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Scores**: context_relevance=0.417, groundedness=0.064, answer_relevance=0.817

**Top docs**:

1. `medquad_ans_0000683-3` (score=0.9993) - Mutations in the GUSB gene cause MPS VII. This gene provides instructions for producing the beta-glucuronidase (-glucuronidase) enzyme, which is involved in the breakdown of large sugar molecules called glycosaminogly...
2. `medquad_ans_0000948-4` (score=0.0005) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
3. `medquad_ans_0001013-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000791-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000670-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The provided context does not mention glycogen storage disease type VII or any genetic changes related to it. The context discusses mucopolysaccharidosis type VII (MPS VII), which is caused by mutations in the GUSB gene leading to a deficiency of beta-glucuronidase enzyme and accumulation of glycosaminoglycans (GAGs). This is distinct from glycogen storage disease type VII.

Therefore, based on the given information, there are no genetic changes related to glycogen storage disease type VII. [No supporting source available]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source available`

### Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.704, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000217-2` (score=0.9972) - Treatment involves eliminating or reducing exposure to the toxic substance, followed by symptomatic and supportive therapy.
2. `medquad_ans_0000172-2__dup1` (score=0.9968) - Treatment is symptomatic and supportive, and may include drug therapy.
3. `medquad_ans_0000230-2__dup1` (score=0.9968) - There is no specific treatment for PVL. Treatment is symptomatic and supportive. Children with PVL should receive regular medical screenings to determine appropriate interventions.
4. `medquad_ans_0000159-3__dup1` (score=0.9958) - IBM is generally resistant to all therapies and its rate of progression appears to be unaffected by currently available treatments.
5. `medquad_ans_0000977-5` (score=0.9933) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megal...

**Answer**:

The treatments for thiamine-responsive megaloblastic anemia syndrome include drug therapy, specifically thiamine (vitamin B1) supplementation, as indicated by the resources addressing its diagnosis and management. These resources mention drug therapy as part of the management approach [medquad_ans_0000977-5]. However, the provided context does not explicitly detail thiamine therapy as a specific treatment for this syndrome. Therefore, based on the available information, drug therapy is included in the management, but more specific details are not provided.

**Supported citations**: `medquad_ans_0000977-5`
**Unsupported citations**: _none_

### Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Scores**: context_relevance=0.857, groundedness=0.532, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000001-7__dup1` (score=0.9937) - Topics
2. `medquad_ans_0000172-7` (score=0.9934) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing simple kidney cysts.
3. `medquad_ans_0000167-17` (score=0.9896) - A dietitian specializes in helping people who have kidney disease choose the right foods and plan healthy meals. People with any kind of kidney disease, including PKD, should talk with a dietitian about foods that sho...
4. `medquad_ans_0000167-18` (score=0.8975) - - Polycystic kidney disease (PKD) is a genetic disorder that causes numerous cysts to grow in the kidneys. - A gene mutation, or defect, causes polycystic kidney disease. - Autosomal dominant PKD is the most common fo...
5. `medquad_ans_0000155-1__dup4` (score=0.3167) - If you have advanced chronic kidney disease (CKD), you may soon need treatment to do the work your kidneys can no longer do. Learning about your treatment options for kidney failure will help you make the best choice ...

**Answer**:

People with polycystic kidney disease (PKD) should talk with a dietitian about foods that should be added to their diet and foods that might be harmful. PKD may require diet changes for blood pressure control. Following a healthy eating plan, such as the Dietary Approaches to Stop Hypertension (DASH) eating plan, can help lower blood pressure. The DASH plan is low in fat and cholesterol, features fat-free or low-fat milk and dairy products, fish, poultry, and nuts, suggests less red meat, sweets, added sugars, and sugar-containing beverages, and is rich in nutrients, protein, and fiber. Individuals should also limit sodium intake to 2,300 mg or less each day. As kidney damage progresses, they may need to eat foods lower in phosphorus and potassium. Foods high in potassium include bananas, oranges, potatoes, and tomatoes; lower-potassium options include apples, peaches,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Scores**: context_relevance=0.636, groundedness=0.794, answer_relevance=0.782

**Top docs**:

1. `medquad_ans_0000253-4__dup1` (score=0.9979) - The goals of research on disorders such as Sjgren's syndrome focus on increasing knowledge and understanding of the disorder, improving diagnostic techniques, testing interventions, and finding ways to treat, prevent,...
2. `medquad_ans_0000018-14` (score=0.9978) - For information about the clinical trial on this topic, visit: http://www.clinicaltrials.gov/ct/show/NCT00372320?order=1. If you would like to read an interview with Dr. Bruce Baum, the study's principal investigator,...
3. `medquad_ans_0000159-4__dup1` (score=0.9975) - The National Institute of Neurological Disorders and Stroke (NINDS), National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS), National Institute of Environmental Health Sciences (NIEHS) and other...
4. `medquad_ans_0000090-4__dup1` (score=0.9973) - Much of the on-going research on RMDs is aimed at prevention and rehabilitation. The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) funds research on RMDs.
5. `medquad_ans_0000008-30` (score=0.9969) - Clinical trials are research studies on people to find out whether a new drug or treatment is both safe and effective. New therapies are tested on people only after laboratory and animal studies show promising results...

**Answer**:

The provided context does not specify research or clinical trials being conducted specifically for Rheumatoid Arthritis (RMDs). However, it mentions that much of the ongoing research on RMDs is aimed at prevention and rehabilitation, and that the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) funds research on RMDs [medquad_ans_0000090-4__dup1]. Additionally, research related to inflammatory myopathies, viral infection as a precursor, and treatment safety and efficacy is being explored by NIH institutes, though this is not directly related to Rheumatoid Arthritis [medquad_ans_0000159-4__dup1].

No specific clinical trials for Rheumatoid Arthritis are detailed in the provided context. [medquad_ans_0000090-4__dup1]

**Supported citations**: `medquad_ans_0000090-4__dup1`, `medquad_ans_0000159-4__dup1`
**Unsupported citations**: _none_

### Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Scores**: context_relevance=0.429, groundedness=0.000, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000217-2` (score=0.9975) - Treatment involves eliminating or reducing exposure to the toxic substance, followed by symptomatic and supportive therapy.
2. `medquad_ans_0000029-2__dup1` (score=0.9973) - Medications may ease certain symptoms, such as pain. Surgery is the only treatment available to correct functional disturbances or halt the progression of damage to the central nervous system. More than one surgery ma...
3. `medquad_ans_0000749-5` (score=0.9971) - These resources address the diagnosis or management of Ohdo syndrome, SBBYS variant: - Gene Review: Gene Review: KAT6B-Related Disorders - Genetic Testing Registry: Young Simpson syndrome These resources from MedlineP...
4. `medquad_ans_0000172-2__dup1` (score=0.9967) - Treatment is symptomatic and supportive, and may include drug therapy.
5. `medquad_ans_0000601-5` (score=0.9962) - These resources address the diagnosis or management of Liddle syndrome: - Genetic Testing Registry: Pseudoprimary hyperaldosteronism - Merck Manual for Health Care Professionals These resources from MedlinePlus offer ...

**Answer**:

The provided context does not contain any information about the treatments for Glass-Chapman-Hockley syndrome. Therefore, it is not possible to answer the question based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.697, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000527-5` (score=0.9975) - These resources address the diagnosis or management of IMAGe syndrome: - Gene Review: Gene Review: IMAGe Syndrome - Genetic Testing Registry: Intrauterine growth retardation, metaphyseal dysplasia, adrenal hypoplasia ...
2. `medquad_ans_0000380-5` (score=0.9974) - These resources address the diagnosis or management of FOXG1 syndrome: - Genetic Testing Registry: Rett syndrome, congenital variant These resources from MedlinePlus offer information about the diagnosis and managemen...
3. `medquad_ans_0000868-5` (score=0.9961) - These resources address the diagnosis or management of Renpenning syndrome: - Genetic Testing Registry: Renpenning syndrome 1 - Greenwood Genetics Center: X-Linked Intellectual Disability - Kennedy Krieger Institute: ...
4. `medquad_ans_0000860-5` (score=0.9949) - These resources address the diagnosis or management of recombinant 8 syndrome: - Genetic Testing Registry: Recombinant chromosome 8 syndrome These resources from MedlinePlus offer information about the diagnosis and m...
5. `medquad_ans_0000227-5` (score=0.9947) - These resources address the diagnosis or management of CHILD syndrome: - Gene Review: Gene Review: NSDHL-Related Disorders - Genetic Testing Registry: Child syndrome These resources from MedlinePlus offer information ...

**Answer**:

The provided context does not specify treatments for Renpenning syndrome. It lists resources related to diagnosis and management, such as the Genetic Testing Registry and Greenwood Genetics Center, and mentions the Kennedy Krieger Institute's focus on genetic disorders of cognition and behavior, but no specific treatments are described [medquad_ans_0000868-5]. Therefore, based on the given information, the treatments for Renpenning syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000868-5`
**Unsupported citations**: _none_

### Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Scores**: context_relevance=1.000, groundedness=0.988, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001105-1` (score=0.9994) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is ava...
2. `medquad_ans_0001104-2` (score=0.0007) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2F? The subtypes of Charcot-Marie-Tooth type 2, including type 2F, have similar signs and symptoms. Affected individuals usually become symptomatic b...
3. `medquad_ans_0001110-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is ava...
4. `medquad_ans_0001099-1__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B1? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B1. If the information is a...
5. `medquad_ans_0006493-1` (score=0.0001) - What are the signs and symptoms of X-linked Charcot-Marie-Tooth disease type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked Charcot-Marie-Tooth disease type 2. If the in...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 2G include onion bulb formation (7.5%), areflexia, autosomal recessive inheritance, axonal degeneration/regeneration, decreased motor nerve conduction velocity, decreased number of peripheral myelinated nerve fibers, distal amyotrophy, distal muscle weakness, distal sensory impairment, flexion contracture, neonatal onset, pes cavus, spinal deformities, split hand, and vocal cord paresis [medquad_ans_0001105-1].

**Supported citations**: `medquad_ans_0001105-1`
**Unsupported citations**: _none_

### Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001393-2` (score=0.9988) - What signs and symptoms are associated with coccygodynia? The classic symptom is pain when pressure is applied to the tailbone, such as when sitting on a hard chair. Symptoms usually improve with relief of pressure wh...
2. `medquad_ans_0000054-3__dup2` (score=0.6230) - Common Symptoms Some people with P.A.D. do not have any symptoms. Others may have a number of signs and symptoms. People who have P.A.D. may notice symptoms when walking or climbing stairs. These symptoms may include ...
3. `medquad_ans_0000696-1__dup2` (score=0.0080) - Pelvic pain occurs mostly in the lower abdomen area. The pain might be steady, or it might come and go. If the pain is severe, it might get in the way of your daily activities. If you're a woman, you might feel a dull...
4. `medquad_ans_0001163-2` (score=0.0001) - What are the signs and symptoms of Chondrocalcinosis 2? The signs and symptoms of chondrocalcinosis 2 vary from person to person. Some affected people may not have any symptoms of the condition aside from the appearan...
5. `medquad_ans_0005261-2` (score=0.0001) - What are the signs and symptoms of Reactive arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Reactive arthritis. If the information is available, the table below includes h...

**Answer**:

The symptoms of coccygodynia include pain when pressure is applied to the tailbone, such as when sitting on a hard chair, immediate and severe pain when moving from sitting to standing, pain during bowel movements, pain during sex, and a deep ache in the region of the tailbone [medquad_ans_0001393-2].

**Supported citations**: `medquad_ans_0001393-2`
**Unsupported citations**: _none_

### Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Scores**: context_relevance=1.000, groundedness=0.888, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0004025-1` (score=0.9994) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgen...
2. `medquad_ans_0004026-1` (score=0.0001) - What are the signs and symptoms of Microcephaly, holoprosencephaly, and intrauterine growth retardation? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, holoprosencepha...
3. `medquad_ans_0004509-1` (score=0.0001) - What are the signs and symptoms of Oculoectodermal syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Oculoectodermal syndrome. If the information is available, the table belo...
4. `medquad_ans_0000259-2__dup1` (score=0.0001) - What are the signs and symptoms of Alpha-thalassemia x-linked intellectual disability syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Alpha-thalassemia x-linked intellectua...
5. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, and microcephaly. Preaxial hand polydactyly, proptosis, and seizures are also listed as possible symptoms. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which draws from Orphanet and other sources, with frequencies often expressed as percentages or fractions [medquad_ans_0004025-1].

**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

### Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.985, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003402-5` (score=0.9964) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to d...
2. `medquad_ans_0000170-3__dup1` (score=0.0001) - The prognosis for most individuals with Klippel-Feil Syndrome is good if the disorder is treated early and appropriately. Activities that can injure the neck should be avoided.
3. `medquad_ans_0000170-2__dup1` (score=0.0001) - Treatment for Klippel-Feil Syndrome is symptomatic and may include surgery to relieve cervical or craniocervical instability and constriction of the spinal cord, and to correct scoliosis. Physical therapy may also be ...
4. `medquad_ans_0005837-1` (score=0.0001) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia x-linked with mental deterioration? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplas...
5. `medquad_ans_0003402-1` (score=0.0001) - Klippel Feil syndrome (KFS) is a congenital, musculoskeletal condition characterized by the fusion of at least two vertebrae of the neck. Common symptoms include a short neck, low hairline at the back of the head, and...

**Answer**:

Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal abnormalities, and additional imaging studies may be needed to assess the extent of the abnormality. Other initial exams are needed to detect additional physical abnormalities or underlying conditions, including examination of the chest to rule out involvement of the heart and lungs, examination of the chest wall to detect possible rib anomalies, MRI for spinal stenosis or neurological deficits, ultrasound of the kidneys for renal abnormalities, hearing evaluation due to the high incidence of hearing loss, and various lab tests to assess organ function [medquad_ans_0003402-5].

**Supported citations**: `medquad_ans_0003402-5`
**Unsupported citations**: _none_

### Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.833, groundedness=0.957, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001497-1` (score=0.9537) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severit...
2. `medquad_ans_0000464-1__dup1` (score=0.5711) - A hernia happens when part of an internal organ or tissue bulges through a weak area of muscle. Most hernias are in the abdomen. There are several types of hernias, including - Inguinal, in the groin. This is the the ...
3. `medquad_ans_0000222-1` (score=0.0661) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
4. `medquad_ans_0002458-1` (score=0.0068) - Fryns syndrome is a condition that affects the development of many parts of the body. Signs and symptoms vary widely among affected individuals. Many affected individuals have a defect in the diaphragm muscle such as ...
5. `medquad_ans_0000468-1__dup1` (score=0.0057) - A hiatal hernia is a condition in which the upper part of the stomach bulges through an opening in the diaphragm. The diaphragm is the muscle wall that separates the stomach from the chest. The diaphragm helps keep ac...

**Answer**:

Congenital diaphragmatic hernia (CDH) is a birth defect characterized by the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest. It ranges in severity from a thinned area in the diaphragm to its complete absence. This defect results in an abnormal opening that allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs. This crowding can lead to underdevelopment of the lungs (pulmonary hypoplasia), potentially causing life-threatening breathing difficulties from birth. CDH may occur with other malformations and can present with symptoms such as breathing problems or abdominal pain, though in about 1 percent of cases, it is asymptomatic and may be detected incidentally during imaging for other reasons [medquad_ans_0001497-1]. It is often classified by position,

**Supported citations**: `medquad_ans_0001497-1`
**Unsupported citations**: _none_

### Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Scores**: context_relevance=0.800, groundedness=0.537, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000168-1__dup2` (score=0.7742) - Charcot-Marie-Tooth disease (CMT) is a group of genetic nerve disorders. It is named after the three doctors who first identified it. In the United States, CMT affects about 1 in 2,500 people. CMT affects your periphe...
2. `medquad_ans_0000074-1__dup1` (score=0.6808) - Charcot-Marie-Tooth disease (CMT) is one of the most common inherited neurological disorders, affecting approximately 1 in 2,500 people in theUnited States. CMT, also known as hereditary motor and sensory neuropathy (...
3. `medquad_ans_0001088-1__dup1` (score=0.0761) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
4. `medquad_ans_0000173-1` (score=0.0431) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
5. `medquad_ans_0001104-6` (score=0.0002) - How might Charcot-Marie-Tooth disease type 2F be treated? Treatment for Charcot-Marie-Tooth disease type 2 mainly focuses on the specific symptoms present. Affected individuals are often managed by a team of various s...

**Answer**:

Charcot-Marie-Tooth disease (CMT) is a group of genetic nerve disorders that affect the peripheral nerves, which carry movement and sensation signals between the brain and spinal cord and the rest of the body. It is one of the most common inherited neurological disorders, affecting approximately 1 in 2,500 people in the United States. CMT is caused by mutations in genes that affect the normal function of the peripheral nerves, which lie outside the brain and spinal cord and supply the muscles and sensory organs in the limbs. Common symptoms include weakness of the foot and lower leg muscles, foot drop, high-stepped gait, foot deformities such as high arches and hammertoes, loss of lower leg muscle bulk, numbness in the foot or leg, and a "slapping" gait. In later stages, weakness and muscle atrophy may also affect the hands, leading to difficulty with fine motor skills. There is no

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000577-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0001018-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000676-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0001093-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether monilethrix is inherited. While several documents describe autosomal recessive inheritance patterns, none explicitly mention monilethrix or its mode of inheritance. Therefore, based on the given information, it is not possible to determine if monilethrix is inherited.

[No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Scores**: context_relevance=0.875, groundedness=0.460, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000217-5` (score=0.9801) - When a child with VUR gets a UTI, bacteria can move into the kidney and lead to scarring. Scarring of the kidney can be associated with high blood pressure and kidney failure. However, most children with VUR who get a...
2. `medquad_ans_0000215-8` (score=0.3362) - Only a health care provider can determine whether a child has a UTI. A urine sample will be collected and examined. The way urine is collected depends on the childs age: - If the child is not yet toilet trained, the h...
3. `medquad_ans_0000223-3__dup2` (score=0.1193) - Any child can get a UTI, though girls get UTIs more often than boys. Children with a condition called vesicoureteral reflux (VUR) are at higher risk for UTIs. VUR causes urine to reflux at the point where one or both ...
4. `medquad_ans_0000215-3__dup2` (score=0.0924) - Most UTIs are caused by bacteria that live in the bowel. The bacterium Escherichia coli (E. coli) causes the vast majority of UTIs. The urinary tract has several systems to prevent infection. The points where the uret...
5. `medquad_ans_0000223-8` (score=0.0410) - Bacteria-fighting medicines called antibiotics are used to treat a UTI. While the lab is doing the urine culture, the health care provider may begin treatment with an antibiotic that treats the bacteria most likely to...

**Answer**:

Urinary Tract Infections (UTIs) in children are infections that occur in the urinary tract, which includes the bladder and urethra. Girls are more likely than boys to develop UTIs, and children with a condition called vesicoureteral reflux (VUR) are at a higher risk. VUR causes urine to reflux at the point where one or both ureters attach to the bladder, which can allow bacteria to move into the kidney and potentially lead to scarring, high blood pressure, or kidney failure. However, most children with VUR who experience a UTI recover without long-term complications. Diagnosis requires a urine sample collected and examined by a health care provider, with collection methods varying by age and including catheterization or needle aspiration in infants, and urine collection in a container for older children. The urine is analyzed under a microscope and cultured to identify bacteria, and sensitivity testing may be performed to determine the most effective antibiotic treatment

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Scores**: context_relevance=0.750, groundedness=0.922, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000874-2` (score=0.9980) - This condition affects an estimated 1 in 8,500 females.
2. `medquad_ans_0000171-3__dup1` (score=0.9971) - KTS is often a progressive disorder, and complications may be life-threatening. However, many individuals can live well while managing their symptoms.
3. `medquad_ans_0000167-7` (score=0.9944) - The complications of autosomal dominant PKD include the following: - Pain. Cyst infection, other types of urinary tract infections (UTIs), bleeding into cysts, kidney stones, or stretching of the fibrous tissue around...
4. `medquad_ans_0000135-6__dup1` (score=0.6068) - People with acquired cystic kidney disease may develop the following complications: - an infected cyst, which can cause fever and back pain. - blood in the urine, which can signal that a cyst in the kidney is bleeding...
5. `medquad_ans_0000135-5__dup2` (score=0.0328) - A person with acquired cystic kidney disease often has no symptoms. However, the complications of acquired cystic kidney disease can have signs and symptoms.

**Answer**:

The complications of autosomal dominant Polycystic Kidney Disease (PKD) include:

- Pain, caused by cyst infection, urinary tract infections (UTIs), bleeding into cysts, kidney stones, or stretching of the fibrous tissue around the kidney due to cyst growth [medquad_ans_0000167-7].
- High blood pressure, present in about half of people with autosomal dominant PKD and normal kidney function between ages 20 and 35, and in almost 100% of those with kidney failure; it increases the risk of heart disease, stroke, and further kidney damage [medquad_ans_0000167-7].
- Kidney failure, which may present with symptoms such as little or no urination, edema, drowsiness, fatigue, itching, dry skin, headaches, weight loss, nausea, vomiting, sleep problems, trouble

**Supported citations**: `medquad_ans_0000167-7`
**Unsupported citations**: _none_

### Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Scores**: context_relevance=0.800, groundedness=0.911, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000001-7__dup1` (score=0.9770) - Topics
2. `medquad_ans_0000649-1__dup1` (score=0.8358) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contami...
3. `medquad_ans_0000100-3__dup3` (score=0.1056) - Anyone can get a foodborne illness. However, some people are more likely to develop foodborne illnesses than others, including - infants and children - pregnant women and their fetuses - older adults - people with wea...
4. `medquad_ans_0000073-3__dup1` (score=0.0054) - COFS is a fatal disease. Most children do not live beyond five years.
5. `medquad_ans_0000123-1__dup3` (score=0.0038) - Viral gastroenteritis is inflammation of the lining of the stomach, small intestine, and large intestine. Several different viruses can cause viral gastroenteritis, which is highly contagious and extremely common. Vir...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by a group of related viruses. Infection leads to inflammation of the stomach and intestines, with common symptoms including diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms are fever, headache, or body aches. The illness can spread from person to person, through contaminated food or water, or by touching a contaminated surface. It can be serious, especially in young children and older adults. There is no specific medicine to treat norovirus infections; treatment focuses on bed rest and consuming plenty of liquids to prevent dehydration. Prevention measures include proper hand washing and safe food preparation [medquad_ans_0000649-1__dup1].

**Supported citations**: `medquad_ans_0000649-1__dup1`
**Unsupported citations**: _none_

### Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.740

**Top docs**:

1. `medquad_ans_0000823-4` (score=0.9749) - This condition is generally not inherited but arises from gene mutations that occur in early blood-forming cells after conception. These alterations are called somatic mutations.
2. `medquad_ans_0000174-3` (score=0.0021) - Mutations in the CHD7 gene cause more than half of all cases of CHARGE syndrome. The CHD7 gene provides instructions for making a protein that most likely regulates gene activity (expression) by a process known as chr...
3. `medquad_ans_0001013-4` (score=0.0003) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000342-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000292-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Some people who have the altered gene never develop the conditio...

**Answer**:

The provided context does not contain information about the genetic changes related to congenital diaphragmatic hernia. Therefore, a specific answer cannot be generated from the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.920, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004534-3` (score=0.9950) - What causes Omenn syndrome? Omenn syndrome is a genetically heterogeneous condition (meaning that it may be caused by a number of different genes). While most cases are attributed to mutations in the RAG genes (RAG-1 ...
2. `medquad_ans_0001071-2` (score=0.0005) - This condition is very rare; only a few dozen affected individuals have been identified.
3. `medquad_ans_0000650-2` (score=0.0004) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
4. `medquad_ans_0000346-2` (score=0.0003) - This condition appears to be rare; only a few affected individuals have been reported worldwide.
5. `medquad_ans_0000904-2` (score=0.0002) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.

**Answer**:

Omenn syndrome is a genetically heterogeneous condition that may be caused by mutations in several genes, including RAG1 and RAG2 (located on chromosome band 11p13), ARTEMIS, ADA, ILRA2, ILRA7, CHD7, and DNA ligase 4. Recent reports have also described cases of Omenn syndrome in the absence of RAG mutations [medquad_ans_0004534-3]. Additionally, some cases have been found in association with 22q11 microdeletion syndrome [medquad_ans_0004534-3].

**Supported citations**: `medquad_ans_0004534-3`
**Unsupported citations**: _none_

### Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Scores**: context_relevance=0.375, groundedness=0.000, answer_relevance=0.575

**Top docs**:

1. `medquad_ans_0001037-4` (score=0.0015) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000969-4` (score=0.0015) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000663-4` (score=0.0015) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000233-4` (score=0.0015) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000569-4` (score=0.0015) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain information about autosomal recessive cerebellar ataxia type 1. Therefore, the answer cannot be determined from the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Scores**: context_relevance=0.375, groundedness=0.000, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000029-2__dup1` (score=0.9960) - Medications may ease certain symptoms, such as pain. Surgery is the only treatment available to correct functional disturbances or halt the progression of damage to the central nervous system. More than one surgery ma...
2. `medquad_ans_0000255-2__dup1` (score=0.9947) - Treatment may include such medications as baclofen, diazepam, tizanidine or clonazepam. Physical therapy regimens may include muscle stretching and range of motion exercises to help prevent shrinkage or shortening of ...
3. `medquad_ans_0000112-2__dup1` (score=0.9939) - No one treatment has been found to be universally effective. Instead, doctors use a variety of therapies (medications, surgery, and other treatments such as physical therapy, splinting, stress management, and biofeedb...
4. `medquad_ans_0000074-2__dup1` (score=0.9936) - There is no cure for CMT, but physical therapy, occupational therapy, braces and other orthopedic devices, and orthopedic surgery can help people cope with the disabling symptoms of the disease. In addition, pain-kill...
5. `medquad_ans_0000232-3__dup1` (score=0.9933) - With treatment, most people recover from pinched nerve. However, in some cases, the damage is irreversible.

**Answer**:

The provided context does not contain information about the treatments for Brachial Plexus Injuries. Therefore, a specific answer cannot be given based on the available information. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Scores**: context_relevance=0.667, groundedness=0.832, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000205-2__dup2` (score=0.9976) - Many events or conditions can damage nerves and nerve pathways. Some of the most common causes are - vaginal childbirth - infections of the brain or spinal cord - diabetes - stroke - accidents that injure the brain or...
2. `medquad_ans_0000218-4__dup2` (score=0.9972) - Urine leakage has many possible causes.
3. `medquad_ans_0000216-1__dup2` (score=0.9961) - The urinary tract is the bodys drainage system for removing wastes and extra fluid. The urinary tract includes two kidneys, two ureters, a bladder, and a urethra. The kidneys are two bean-shaped organs, each about the...
4. `medquad_ans_0000141-3__dup2` (score=0.9940) - During fetal development, a babys kidneys first appear as buds inside the pelvisthe bowl-shaped bone that supports the spine and holds up the digestive, urinary, and reproductive organsnear the bladder. As the kidneys...
5. `medquad_ans_0000216-8` (score=0.9936) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PU...

**Answer**:

The provided context does not specify the causes of urine blockage in newborns. While it mentions that some children are born with nerve problems that can keep the bladder from releasing urine, leading to urinary infections or kidney damage [medquad_ans_0000205-2__dup2], it does not directly address urine blockage in newborns or its specific causes. Therefore, based on the given information, the cause of urine blockage in newborns cannot be determined.

**Supported citations**: `medquad_ans_0000205-2__dup2`
**Unsupported citations**: _none_

### Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Scores**: context_relevance=0.111, groundedness=0.714, answer_relevance=0.378

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9996) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
3. `medquad_ans_0000790-2` (score=0.9995) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
4. `medquad_ans_0000874-2` (score=0.9995) - This condition affects an estimated 1 in 8,500 females.
5. `medquad_ans_0000091-2` (score=0.9995) - This condition is rare, affecting fewer than 1 per million people.

**Answer**:

The exact incidence of intranuclear rod myopathy is unknown, but estimates vary. One source indicates it affects 1 in 87,000 people [medquad_ans_0000484-2], another estimates 1 in 50,000 [medquad_ans_0000989-2], and the prevalence is uncertain with ranges from 1 in 25,000 to 1 in 300,000 [medquad_ans_0000790-2]. Additionally, it affects an estimated 1 in 8,500 females [medquad_ans_0000874-2], and one source notes it affects fewer than 1 per million people [medquad_ans_0000091-2]. Due to the variability in estimates, a precise number cannot be determined

**Supported citations**: `medquad_ans_0000484-2`, `medquad_ans_0000989-2`, `medquad_ans_0000790-2`, `medquad_ans_0000874-2`, `medquad_ans_0000091-2`
**Unsupported citations**: _none_

### Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Scores**: context_relevance=1.000, groundedness=0.910, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000058-3__dup4` (score=0.9849) - Blood pressure is the force of blood flow inside your blood vessels. Blood pressure is written with two numbers separated by a slash. For example, a blood pressure result of 130/80 is said as 130 over 80. The first nu...
2. `medquad_ans_0000036-12` (score=0.9806) - Abnormal blood pressure is higher than 120/80 mmHg. If either your systolic or diastolic blood pressure is higher than normal (120/80) but not high enough to be considered high blood pressure (140/90), you have pre-hy...
3. `medquad_ans_0000036-21` (score=0.9779) - In most cases, the goal is probably to keep your blood pressure below 140/90 mmHg (130/80 if you have diabetes or chronic kidney disease). Normal blood pressure is less than 120/80. Ask your doctor what your blood pre...
4. `medquad_ans_0000036-10` (score=0.9690) - Normal blood pressure for adults is defined as a systolic pressure below 120 mmHg and a diastolic pressure below 80 mmHg. It is normal for blood pressures to change when you sleep, wake up, or are excited or nervous. ...
5. `medquad_ans_0000565-1__dup1` (score=0.9669) - You've probably heard that high blood pressure is a problem. Sometimes blood pressure that is too low can also cause problems. Blood pressure is the force of your blood pushing against the walls of your arteries. Each...

**Answer**:

High blood pressure is defined as a blood pressure reading of 140/90 mmHg or higher. It is characterized by increased pressure in the blood vessels, which forces the heart to work harder and can lead to strain on the heart, damage to blood vessels, and an increased risk of heart attack, stroke, eye problems, and kidney problems. The goal for most adults is to keep blood pressure below 140/90 mmHg, and for individuals with diabetes or chronic kidney disease, the goal is to keep it below 130/80 mmHg [medquad_ans_0000036-21]. Abnormal blood pressure is higher than 120/80 mmHg, and if the pressure is above normal but not yet in the high blood pressure range, it is classified as pre-hypertension [medquad_ans_00000

**Supported citations**: `medquad_ans_0000036-21`
**Unsupported citations**: _none_

### Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000053-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000495-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000961-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000236-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000286-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention X-linked dystonia-parkinsonism or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether X-linked dystonia-parkinsonism is inherited. [No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.973, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000702-1__dup1` (score=0.9065) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ket...
2. `medquad_ans_0000120-1` (score=0.8601) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process keto...
3. `medquad_ans_0000125-4` (score=0.0015) - This condition is inherited in an autosomal recessive pattern, which means both copies of the BTD gene in each cell have mutations. The parents of an individual with biotinidase deficiency each carry one copy of the m...
4. `medquad_ans_0001071-2` (score=0.0015) - This condition is very rare; only a few dozen affected individuals have been identified.
5. `medquad_ans_0001013-4` (score=0.0012) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. Signs and symptoms typically appear between the ages of 6 months and 24 months and include episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, known as ketoacidotic attacks, sometimes lead to coma and are frequently triggered by infections, periods without food (fasting), or increased intake of protein-rich foods. The condition is inherited in an autosomal recessive fashion and is caused by mutations in the ACAT1 gene [medquad_ans_0000702-1__dup1].

**Supported citations**: `medquad_ans_0000702-1__dup1`
**Unsupported citations**: _none_

### Example 285 (`0000006_5-7`)

**Question**: What are the stages of Childhood Brain Stem Glioma ?

**Scores**: context_relevance=0.889, groundedness=0.954, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0000006_5-7` (score=0.9969) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the...
2. `medquad_ans_0000006_3-7` (score=0.9921) - Key Points - The grade of the tumor is used to plan cancer treatment. - Low-grade astrocytomas - High-grade astrocytomas - An MRI is done after surgery. The grade of the tumor is used to plan cancer treatment. Staging...
3. `medquad_ans_0000006_7-2` (score=0.0136) - Key Points - Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and s...
4. `medquad_ans_0000006_2-8` (score=0.0031) - Key Points - There are different types of treatment for children with brain and spinal cord tumors. - Children with brain or spinal cord tumors should have their treatment planned by a team of health care providers wh...
5. `medquad_ans_0000006_1-7` (score=0.0026) - Key Points - There is no standard staging system for adult brain and spinal cord tumors. - Imaging tests may be repeated after surgery to help plan more treatment. There is no standard staging system for adult brain a...

**Answer**:

There is no standard staging system for childhood brain stem glioma [medquad_ans_0000006_5-7].

**Supported citations**: `medquad_ans_0000006_5-7`
**Unsupported citations**: _none_

### Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.533, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000266-1__dup2` (score=0.3112) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal c...
2. `medquad_ans_0000934-2` (score=0.0063) - This condition affects fewer than 1 in 150,000 males and is very rare in females.
3. `medquad_ans_0000054-4__dup1` (score=0.0036) - The NINDS supports and conducts a wide range of research on spinal cord disorders such as BSS. The goal of this research is to find ways to prevent, treat, and, ultimately, cure these disorders.
4. `medquad_ans_0000812-2` (score=0.0019) - This condition appears to be rare; it has been reported in only a few individuals and families worldwide.
5. `medquad_ans_0000158-4__dup2` (score=0.0019) - NINDS supports research on brain and spinal cord disorders that can cause hypertonia. The goals of this research are to learn more about how the nervous system adapts after injury or disease and to find ways to preven...

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or may develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course. In children, symptoms may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is believed to result from improper growth of the neural tube during fetal development and is closely linked to spina bifida. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. This delayed presentation is related to the degree of strain on the spinal cord over

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Scores**: context_relevance=0.375, groundedness=0.876, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000989-2` (score=0.9986) - This condition affects an estimated 1 in 50,000 people.
2. `medquad_ans_0000222-3__dup1` (score=0.9983) - The prognosis for those with paresthesia depends on the severity of the sensations and the associated disorders.
3. `medquad_ans_0000114-3__dup2` (score=0.9982) - ESS is not a life-threatening condition. Most often, and particularly among those with primary ESS, the disorder does not cause health problems and does not affect life expectancy.
4. `medquad_ans_0000029-2__dup1` (score=0.9979) - Medications may ease certain symptoms, such as pain. Surgery is the only treatment available to correct functional disturbances or halt the progression of damage to the central nervous system. More than one surgery ma...
5. `medquad_ans_0000171-3__dup1` (score=0.9979) - KTS is often a progressive disorder, and complications may be life-threatening. However, many individuals can live well while managing their symptoms.

**Answer**:

The outlook for Carpal Tunnel Syndrome (CTS) varies. Many individuals can live well while managing their symptoms, and some have no noticeable symptoms that interfere with daily activities. Medications may help ease symptoms such as pain, and surgery is the only treatment available to correct functional disturbances or halt progression of central nervous system damage, with more than one surgery possibly needed in some cases [medquad_ans_0000029-2__dup1]. However, the condition may be progressive, and complications can be life-threatening [medquad_ans_0000171-3__dup1]. Overall, while complications may arise, many people manage the condition effectively.

**Supported citations**: `medquad_ans_0000029-2__dup1`, `medquad_ans_0000171-3__dup1`
**Unsupported citations**: _none_

### Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.000, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000297-5` (score=0.9970) - These resources address the diagnosis or management of Donohue syndrome: - Genetic Testing Registry: Leprechaunism syndrome These resources from MedlinePlus offer information about the diagnosis and management of vari...
2. `medquad_ans_0000298-5` (score=0.9970) - These resources address the diagnosis or management of DOORS syndrome: - Gene Review: Gene Review: TBC1D24-Related Disorders These resources from MedlinePlus offer information about the diagnosis and management of var...
3. `medquad_ans_0000749-5` (score=0.9970) - These resources address the diagnosis or management of Ohdo syndrome, SBBYS variant: - Gene Review: Gene Review: KAT6B-Related Disorders - Genetic Testing Registry: Young Simpson syndrome These resources from MedlineP...
4. `medquad_ans_0000052-2__dup1` (score=0.9969) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
5. `medquad_ans_0000172-2__dup1` (score=0.9967) - Treatment is symptomatic and supportive, and may include drug therapy.

**Answer**:

The provided context does not contain information about the treatments for Donnai-Barrow syndrome. Therefore, it is not possible to answer the question based on the given context. [no_context]

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

**Scores**: context_relevance=0.833, groundedness=0.954, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0001900-1` (score=0.9606) - Dominant optic atrophy (DOA) is an inherited optic nerve disorder characterized by degeneration of the optic nerves. It typically starts during the first decade of life. Affected people usually develop moderate visual...
2. `medquad_ans_0000585-1__dup1` (score=0.9521) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
3. `medquad_ans_0000931-4` (score=0.0015) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000263-4` (score=0.0015) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000133-4` (score=0.0015) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. It is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations, and the parents of an individual with the condition each carry one copy of the mutated gene but typically do not show signs and symptoms [medquad_ans_0000585-1\_\_dup1][medquad_ans_0000931-4][medquad_ans_0000263-4][medquad_ans_0000133-4].

**Supported citations**: `medquad_ans_0000585-1__dup1`, `medquad_ans_0000931-4`, `medquad_ans_0000263-4`, `medquad_ans_0000133-4`
**Unsupported citations**: _none_

### Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Scores**: context_relevance=0.900, groundedness=0.977, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000425-3` (score=0.9992) - Mutations in the PHKA1, PHKA2, PHKB, or PHKG2 genes are known to cause GSD IX. These genes provide instructions for making pieces (subunits) of an enzyme called phosphorylase b kinase. The enzyme is made up of 16 subu...
2. `medquad_ans_0000794-3` (score=0.9991) - Phosphoglycerate kinase deficiency is caused by mutations in the PGK1 gene. This gene provides instructions for making an enzyme called phosphoglycerate kinase, which is involved in a critical energy-producing process...
3. `medquad_ans_0000795-3` (score=0.9983) - Phosphoglycerate mutase deficiency is caused by mutations in the PGAM2 gene. This gene provides instructions for making an enzyme called phosphoglycerate mutase, which is involved in a critical energy-producing proces...
4. `medquad_ans_0000984-3` (score=0.0809) - As the condition name suggests, mutations in the TK2 gene cause TK2-MDS. The TK2 gene provides instructions for making an enzyme called thymidine kinase 2 that functions within cell structures called mitochondria, whi...
5. `medquad_ans_0000397-3` (score=0.0110) - Mutations in the GALT, GALK1, and GALE genes cause galactosemia. These genes provide instructions for making enzymes that are essential for processing galactose obtained from the diet. These enzymes break down galacto...

**Answer**:

The genetic changes related to phosphoglycerate kinase deficiency are mutations in the PGK1 gene [medquad_ans_0000794-3]. These mutations reduce the activity of the phosphoglycerate kinase enzyme, which disrupts energy production in cells [medquad_ans_0000794-3].

**Supported citations**: `medquad_ans_0000794-3`
**Unsupported citations**: _none_

### Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Scores**: context_relevance=0.700, groundedness=0.930, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000850-3` (score=0.9993) - The primary genetic cause of PVOD is mutations in the EIF2AK4 gene. Mutations in other genes may cause a small percentage of cases. Other suspected causes of PVOD include viral infection and exposure to toxic chemical...
2. `medquad_ans_0000805-3` (score=0.9992) - Mutations in the TREM2 gene or the TYROBP gene (also called DAP12) can cause PLOSL. The proteins produced from these two genes work together to activate certain kinds of cells. These proteins appear to be particularly...
3. `medquad_ans_0000047-3` (score=0.9990) - ACD/MPV can be caused by mutations in the FOXF1 gene. The protein produced from the FOXF1 gene is a transcription factor, which means that it attaches (binds) to specific regions of DNA and helps control the activity ...
4. `medquad_ans_0000687-3` (score=0.9987) - Mutations in the TEK gene (also called the TIE2 gene) cause VMCM. The TEK gene provides instructions for making a protein called TEK receptor tyrosine kinase. This receptor protein triggers chemical signals needed for...
5. `medquad_ans_0000870-3` (score=0.9987) - RAMSVPS is caused by a mutation in the IGFBP7 gene. This gene provides instructions for making a protein called insulin-like growth factor-binding protein 7 (IGFBP7). The IGFBP7 protein is active in the lining of bloo...

**Answer**:

The primary genetic cause of pulmonary veno-occlusive disease (PVOD) is mutations in the EIF2AK4 gene. Mutations in other genes may cause a small percentage of cases. The EIF2AK4 gene produces a protein that helps cells respond to changes that could damage the cell, such as low levels of amino acids, by reducing protein production to conserve amino acids. Mutations in EIF2AK4 likely eliminate functional EIF2AK4 protein, though it is unknown how the absence of this protein's function leads to pulmonary vessel abnormalities in PVOD [medquad_ans_0000850-3].

**Supported citations**: `medquad_ans_0000850-3`
**Unsupported citations**: _none_

### Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Scores**: context_relevance=0.625, groundedness=0.933, answer_relevance=0.425

**Top docs**:

1. `medquad_ans_0001006-3` (score=0.9991) - Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. In the liver, enzymes break down tyrosine in a five step process, resulting in molecules that are either excreted by t...
2. `medquad_ans_0000984-3` (score=0.9939) - As the condition name suggests, mutations in the TK2 gene cause TK2-MDS. The TK2 gene provides instructions for making an enzyme called thymidine kinase 2 that functions within cell structures called mitochondria, whi...
3. `medquad_ans_0000332-4` (score=0.1581) - This condition is generally not inherited but arises from a mutation in the body's cells that occurs after conception. This alteration is called a somatic mutation.
4. `medquad_ans_0001102-4` (score=0.1581) - This condition is generally not inherited but arises from a mutation in the body's cells that occurs after conception. This alteration is called a somatic mutation.
5. `medquad_ans_0000406-4` (score=0.0748) - This condition is generally not inherited but arises from a mutation in the body's cells that occurs after conception. This alteration is called a somatic mutation. A somatic mutation in one copy of the NRAS or BRAF g...

**Answer**:

Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively [medquad_ans_0001006-3]. These mutations lead to a decrease in the activity of specific enzymes involved in the breakdown of tyrosine, resulting in the accumulation of toxic levels of tyrosine and its byproducts [medquad_ans_0001006-3].

**Supported citations**: `medquad_ans_0001006-3`
**Unsupported citations**: _none_

### Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Scores**: context_relevance=0.143, groundedness=0.000, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000209-2` (score=0.9953) - Treatment involves restoring normal levels of thyroid hormone and may include thyroid drugs, radioactive iodine, and sometimes partial or complete surgical removal of the thyroid.
2. `medquad_ans_0000172-2__dup1` (score=0.9946) - Treatment is symptomatic and supportive, and may include drug therapy.
3. `medquad_ans_0000043-2__dup2` (score=0.9943) - In most cases of BEB the treatment of choice is botulinum toxin injections which relax the muscles and stop the spasms. Other treatment options include medications (drug therapy) or surgery--either local surgery of th...
4. `medquad_ans_0000217-2` (score=0.9942) - Treatment involves eliminating or reducing exposure to the toxic substance, followed by symptomatic and supportive therapy.
5. `medquad_ans_0000209-3` (score=0.9930) - With treatment, muscle weakness may improve or be reversed.

**Answer**:

The provided context does not contain information about the treatments for Graves' Disease. Therefore, a specific answer cannot be given based on the available context. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000239-2__dup1` (score=0.9913) - TSEs tend to progress rapidly and usually culminate in death over the course of a few months to a few years.
2. `medquad_ans_0000039-1__dup2` (score=0.2434) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, ...
3. `medquad_ans_0000203-3__dup1` (score=0.0171) - The disease tends to advance rapidly over the course of 5 to 10 years, with progressive loss of motor skills, eventual confinement to bed, and death. There is no remission from the disease. There is currently no cure.
4. `medquad_ans_0000039-3__dup1` (score=0.0037) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten dise...
5. `medquad_ans_0000192-2__dup1` (score=0.0025) - There is no cure for MLD. Bone marrow transplantation may delay progression of the disease in some infantile-onset cases. Other treatment is symptomatic and supportive. Considerable progress has been made with regard ...

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, or stumbling. Symptoms of Batten disease are linked to a buildup of substances called lipopigments in the body's tissues. Lipopigments are made up of fats and proteins. Because vision loss is often an early sign, Batten disease may be first suspected during an eye exam. Often, an eye specialist or other physician may refer the child to a neurologist. Diagnostic tests for Batten disease include blood or urine tests, skin or tissue sampling, an electroencephalogram (EEG), electrical studies of the eyes, and brain scans. Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002061-1` (score=0.5247) - Ehlers-Danlos syndrome progeroid type is a genetic disorder of the connective tissue, which is the material between the cells of the body that gives tissues form and strength. The disorder primarily affects the skin, ...
2. `medquad_ans_0003802-1` (score=0.0001) - What are the signs and symptoms of Marfanoid hypermobility syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Marfanoid hypermobility syndrome. If the information is available...
3. `medquad_ans_0003657-1` (score=0.0001) - What are the signs and symptoms of Loeys-Dietz syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Loeys-Dietz syndrome type 2. If the information is available, the tabl...
4. `medquad_ans_0004353-1` (score=0.0001) - What are the signs and symptoms of Nestor-guillermo progeria syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Nestor-guillermo progeria syndrome. If the information is avail...
5. `medquad_ans_0002049-6` (score=0.0001) - How might Ehlers-Danlos syndrome be treated? There is no specific cure for Ehlers-Danlos syndrome (EDS). The treatment and management is focused on preventing serious complications and relieving associated signs and s...

**Answer**:

Ehlers-Danlos syndrome, progeroid type is a genetic disorder of the connective tissue that primarily affects the skin, hair, and skeletal system. Symptoms usually appear by childhood or adolescence and include unusually flexible joints, loose elastic skin, and easy scarring. Unique features of this type include sparse scalp hair and eyebrows, and loose elastic skin on the face, which cause affected individuals to look older than their age. Additional symptoms may involve bone weakness, weak muscle tone, mild intellectual disability, and delayed growth in children. This condition is caused by mutations in the B4GALT7 gene and is inherited in an autosomal recessive pattern [medquad_ans_0002061-1].

**Supported citations**: `medquad_ans_0002061-1`
**Unsupported citations**: _none_

### Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.984, answer_relevance=0.480

**Top docs**:

1. `medquad_ans_0000416-3` (score=0.9991) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
2. `medquad_ans_0000066-3` (score=0.0018) - Mutations in the GATM gene cause arginine:glycine amidinotransferase deficiency. The GATM gene provides instructions for making the enzyme arginine:glycine amidinotransferase. This enzyme participates in the two-step ...
3. `medquad_ans_0001013-4` (score=0.0004) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000442-3` (score=0.0003) - Mutations in the GAMT gene cause guanidinoacetate methyltransferase deficiency. The GAMT gene provides instructions for making the enzyme guanidinoacetate methyltransferase. This enzyme participates in the two-step pr...
5. `medquad_ans_0000749-4` (score=0.0002) - This condition has an autosomal dominant inheritance pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all reported cases have resulted from new mutations in th...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the last two steps of the metabolism of the amino acid histidine and in the production of one of the forms of the vitamin folate [medquad_ans_0000416-3].

**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

### Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Scores**: context_relevance=1.000, groundedness=0.967, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001169-2` (score=0.9994) - What are the signs and symptoms of Chondrodysplasia punctata 1, X-linked recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 1, X-linked recessive. I...
2. `medquad_ans_0005838-1` (score=0.0001) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia, Aggrecan type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia, Aggrecan type. I...
3. `medquad_ans_0006051-2` (score=0.0001) - What are the signs and symptoms of Tetrasomy X? The Human Phenotype Ontology provides the following list of signs and symptoms for Tetrasomy X. If the information is available, the table below includes how often the s...
4. `medquad_ans_0000332-1__dup1` (score=0.0001) - What are the signs and symptoms of Anauxetic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Anauxetic dysplasia. If the information is available, the table below includes...
5. `medquad_ans_0004193-1` (score=0.0001) - What are the signs and symptoms of Multiple epiphyseal dysplasia 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Multiple epiphyseal dysplasia 4. If the information is available, ...

**Answer**:

The symptoms of Chondrodysplasia punctata 1, X-linked recessive include abnormality of the vertebral column, anosmia, cataract, depressed nasal bridge, epiphyseal stippling, hearing impairment, hypogonadism, ichthyosis, microcephaly, short distal phalanx of finger, short nasal septum, short nose, short stature, and X-linked recessive inheritance [medquad_ans_0001169-2].

**Supported citations**: `medquad_ans_0001169-2`
**Unsupported citations**: _none_

### Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Scores**: context_relevance=0.429, groundedness=0.872, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000750-5` (score=0.9975) - These resources address the diagnosis or management of Ollier disease: - Genetic Testing Registry: Enchondromatosis These resources from MedlinePlus offer information about the diagnosis and management of various heal...
2. `medquad_ans_0000945-5` (score=0.9972) - These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliative Care
3. `medquad_ans_0001043-5` (score=0.9963) - These resources address the diagnosis or management of Wolman disease: - Genetic Testing Registry: Lysosomal acid lipase deficiency These resources from MedlinePlus offer information about the diagnosis and management...
4. `medquad_ans_0000172-2__dup1` (score=0.9959) - Treatment is symptomatic and supportive, and may include drug therapy.
5. `medquad_ans_0000132-10` (score=0.9879) - A person with Whipple disease and malabsorption may need - a diet high in calories and protein - vitamins - nutritional supplements

**Answer**:

The treatments for Whipple disease include symptomatic and supportive care, which may involve drug therapy. Additionally, a person with Whipple disease and malabsorption may need a diet high in calories and protein, vitamins, and nutritional supplements [medquad_ans_0000172-2\_\_dup1][medquad_ans_0000132-10].

**Supported citations**: `medquad_ans_0000172-2__dup1`, `medquad_ans_0000132-10`
**Unsupported citations**: _none_

### Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Scores**: context_relevance=0.429, groundedness=0.000, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000052-2__dup1` (score=0.9966) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
2. `medquad_ans_0000016-2__dup1` (score=0.9964) - Drug therapy including verapamil may help to reduce the severity and duration of attacks of paralysis associated with the more serious form of alternating hemiplegia
3. `medquad_ans_0000190-2__dup1` (score=0.9962) - Treatment is symptomatic and may include medication therapies with nonsteroidal anti-inflammatory drugs (NSAIDs) and corticosteroids to reduce swelling, as well as antibiotics and immunosuppressants. Surgery may be re...
4. `medquad_ans_0000209-3` (score=0.9960) - With treatment, muscle weakness may improve or be reversed.
5. `medquad_ans_0000043-2__dup2` (score=0.9955) - In most cases of BEB the treatment of choice is botulinum toxin injections which relax the muscles and stop the spasms. Other treatment options include medications (drug therapy) or surgery--either local surgery of th...

**Answer**:

The provided context does not contain information about the treatments for Bell's Palsy. Therefore, based on the given context, it is not possible to answer the question. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`
