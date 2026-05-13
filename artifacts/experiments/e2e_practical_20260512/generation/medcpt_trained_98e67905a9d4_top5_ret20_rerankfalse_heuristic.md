# Generation Evaluation

## Summary

- `avg_answer_chars`: 582.6167
- `avg_answer_relevance`: 0.7476
- `avg_citation_presence_rate`: 0.8267
- `avg_context_relevance`: 0.8345
- `avg_empty_answer`: 0.0000
- `avg_end_to_end_latency_ms`: 6195.0815
- `avg_generation_latency_ms`: 6074.6439
- `avg_groundedness`: 0.7439
- `avg_insufficient_context`: 0.0233
- `avg_num_retrieved_docs`: 5.0000
- `avg_rerank_latency_ms`: 0.0000
- `avg_retrieval_latency_ms`: 120.2749
- `avg_supported_citation_rate`: 0.7333
- `avg_unsupported_citation_rate`: 0.0933
- `generation_remote_concurrency`: 4.0000
- `num_examples`: 300
- `reranker_enabled_rate`: 0.0000

## Examples

### Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.974, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006510-1` (score=0.5759) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the informatio...
2. `medquad_ans_0006511-1` (score=0.5465) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the informatio...
3. `medquad_ans_0006509-2` (score=0.5398) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is...
4. `medquad_ans_0005115-2` (score=0.5193) - What are the signs and symptoms of Proteus syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Proteus syndrome. If the information is available, the table below includes how o...
5. `medquad_ans_0005014-1` (score=0.5126) - What are the signs and symptoms of PPM-X syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for PPM-X syndrome. If the information is available, the table below includes how often...

**Answer**:

The symptoms of X-linked lymphoproliferative syndrome 1 include cellular immunodeficiency (90%), decreased antibody level in blood (50%), hepatomegaly (50%), lymphadenopathy (50%), lymphoma (50%), splenomegaly (50%), anemia (7.5%), encephalitis, fulminant hepatitis, hepatic encephalopathy, IgG deficiency, immunodeficiency, increased IgM level, meningitis, pancytopenia, recurrent pharyngitis, reduced natural killer cell activity, thrombocytopenia, and X-linked recessive inheritance [medquad_ans_0006510-1].

**Supported citations**: `medquad_ans_0006510-1`
**Unsupported citations**: _none_

### Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000342-4` (score=0.4886) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
2. `medquad_ans_0001013-4` (score=0.4873) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000785-4` (score=0.4854) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000123-4` (score=0.4850) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0001088-4` (score=0.4850) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether early-onset primary dystonia is inherited. Therefore, based on the given information, it cannot be determined. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Scores**: context_relevance=0.714, groundedness=0.899, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000397-2__dup1` (score=0.5240) - The tapeworms that cause taeniasis (Taenia saginata, T. solium, and T. asiatica) are found worldwide. Eating raw or undercooked beef or pork is the primary risk factor for acquiring taeniasis. Persons who don't eat ra...
2. `medquad_ans_0000121-7` (score=0.5025) - Some people with cysticercosis do not need to be treated. There are medications available to treat cysticercosis for those who do need treatment. Sometimes surgery may be needed. Your doctor will advise you on which t...
3. `medquad_ans_0000121-6` (score=0.4825) - If you think that you may have cysticercosis, please see your health care provider. Your health care provider will ask you about your symptoms, where you have travelled, and what kinds of foods you eat. The diagnosis ...
4. `medquad_ans_0000121-2__dup3` (score=0.4815) - Cysticercosis is an infection caused by the larvae of the parasite Taenia solium. This infection occurs after a person swallows tapeworm eggs. The larvae get into tissues such as muscle and brain, and form cysts there...
5. `medquad_ans_0000397-1__dup2` (score=0.4779) - Taeniasis in humans is a parasitic infection caused by the tapeworm species Taenia saginata (beef tapeworm), Taenia solium (pork tapeworm), and Taenia asiatica (Asian tapeworm). Humans can become infected with these t...

**Answer**:

People at risk for taeniasis include those who eat raw or undercooked beef or pork. Infections with Taenia saginata occur where contaminated raw beef is consumed, particularly in Eastern Europe, Russia, eastern Africa, and Latin America. Taenia solium infections are more common in underdeveloped communities with poor sanitation and where raw or undercooked pork is consumed, with higher rates seen in Latin America, Eastern Europe, sub-Saharan Africa, India, and Asia. Taeniasis due to T. solium is also seen in the United States, typically among Latin American immigrants. Additionally, individuals with poor hygiene who ingest tapeworm eggs from contaminated environments may be at risk for cysticercosis, which can result from taeniasis [medquad_ans_0000397-2__dup1].

**Supported citations**: `medquad_ans_0000397-2__dup1`
**Unsupported citations**: _none_

### Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000194-2__dup1` (score=0.5850) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
2. `medquad_ans_0000193-2__dup1` (score=0.5704) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 1. If the information is availab...
3. `medquad_ans_0000197-2__dup1` (score=0.5648) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 5. If the information is availab...
4. `medquad_ans_0000195-2__dup1` (score=0.5576) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 3. If the information is availab...
5. `medquad_ans_0000196-2__dup1` (score=0.5576) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 4. If the information is availab...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy [medquad_ans_0000194-2__dup1].

**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

### Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003421-1` (score=0.6209) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0005007-2` (score=0.5697) - What are the signs and symptoms of Potocki-Lupski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Potocki-Lupski syndrome. If the information is available, the table below ...
3. `medquad_ans_0003313-1` (score=0.5690) - What are the signs and symptoms of Juberg Marsidi syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Juberg Marsidi syndrome. If the information is available, the table below ...
4. `medquad_ans_0001667-1` (score=0.5594) - What are the signs and symptoms of Crisponi syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Crisponi syndrome. If the information is available, the table below includes how...
5. `medquad_ans_0003423-1` (score=0.5581) - What are the signs and symptoms of Kozlowski Celermajer Tink syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kozlowski Celermajer Tink syndrome. If the information is avail...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

### Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5426) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000150-4` (score=0.5412) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000574-4` (score=0.5412) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000908-4` (score=0.5412) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000805-4` (score=0.5412) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention hemophilia or specify how it is inherited. Therefore, based on the given information, it is not possible to determine if hemophilia is inherited. [No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.970, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.6351) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often t...
2. `medquad_ans_0006421-1` (score=0.5639) - What are the signs and symptoms of Wells-Jankovic syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Wells-Jankovic syndrome. If the information is available, the table below ...
3. `medquad_ans_0005369-1` (score=0.5416) - What are the signs and symptoms of Rhizomelic syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Rhizomelic syndrome. If the information is available, the table below includes...
4. `medquad_ans_0006355-2` (score=0.5376) - What are the signs and symptoms of Vici syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Vici syndrome. If the information is available, the table below includes how often t...
5. `medquad_ans_0002946-1` (score=0.5325) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...

**Answer**:

The symptoms of IVIC syndrome include hearing impairment (90%), limitation of joint mobility (90%), short stature (90%), strabismus (90%), abnormal dermatoglyphics (50%), aplasia/hypoplasia of the thumb (50%), radioulnar synostosis (50%), scoliosis (50%), synostosis of carpal bones (50%), triphalangeal thumb (50%), abnormality of the clavicle (7.5%), arrhythmia (7.5%), leukocytosis (7.5%), preaxial hand polydactyly (7.5%), thrombocytopenia (7.5%), and urogenital fistula (7.5%) [medquad_ans_0003284-1].

**Supported citations**: `medquad_ans_0003284-1`
**Unsupported citations**: _none_

### Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Scores**: context_relevance=1.000, groundedness=0.955, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000775-2` (score=0.4980) - Paroxysmal nocturnal hemoglobinuria is a rare disorder, estimated to affect between 1 and 5 per million people.
2. `medquad_ans_0000775-5` (score=0.4697) - These resources address the diagnosis or management of paroxysmal nocturnal hemoglobinuria: - Duke University School of Medicine: Hemostasis & Thrombosis Center - Genetic Testing Registry: Paroxysmal nocturnal hemoglo...
3. `medquad_ans_0000775-1` (score=0.4620) - Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. The disorder affects red blood cells (erythrocytes), which carry oxygen; white bloo...
4. `medquad_ans_0000775-3` (score=0.4596) - Mutations in the PIGA gene cause paroxysmal nocturnal hemoglobinuria. The PIGA gene provides instructions for making a protein called phosphatidylinositol glycan class A. This protein takes part in a series of steps t...
5. `medquad_ans_0004744-1` (score=0.4553) - What are the signs and symptoms of Paroxysmal cold hemoglobinuria? The Human Phenotype Ontology provides the following list of signs and symptoms for Paroxysmal cold hemoglobinuria. If the information is available, th...

**Answer**:

Paroxysmal nocturnal hemoglobinuria is a rare acquired disorder that leads to the premature death and impaired production of blood cells, affecting red blood cells (erythrocytes), white blood cells (leukocytes), and platelets (thrombocytes) [medquad_ans_0000775-1]. It is characterized by sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to hemoglobinuria—dark-colored urine due to hemoglobin in the urine, most noticeable in the morning [medquad_ans_0000775-1]. This results in hemolytic anemia, causing symptoms like fatigue, weakness, pallor, shortness of breath, and tachycardia. Individuals are also prone to infections due to white

**Supported citations**: `medquad_ans_0000775-1`
**Unsupported citations**: _none_

### Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.977, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000010-10` (score=0.5529) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the li...
2. `medquad_ans_0000037_4-2` (score=0.5238) - Different factors increase or decrease the risk of getting colorectal cancer. Anything that increases your chance of getting a disease is called a risk factor. Anything that decreases your chance of getting a disease ...
3. `medquad_ans_0000010-11` (score=0.5190) - Colorectal cancer accounts for almost ten percent of all cancer deaths in the United States. The risk of developing colorectal cancer rises after age 50. It is common in both men and women.
4. `medquad_ans_0000037_4-1` (score=0.5182) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...
5. `medquad_ans_0000221-1__dup2` (score=0.5158) - The colon and rectum are part of the large intestine. Colorectal cancer occurs when tumors form in the lining of the large intestine. It is common in both men and women. The risk of developing colorectal cancer rises ...

**Answer**:

Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. The colon and the rectum are parts of the body's digestive system. Cancer that begins in the colon is called colon cancer, and cancer that begins in the rectum is called rectal cancer. Cancer that begins in either of these organs may also be called colorectal cancer [medquad_ans_0000037_4-1]. Colorectal cancer occurs when malignant tumors form in the lining of the large intestine, also called the large bowel [medquad_ans_0000010-10]. It is the second leading cause of death from cancer in the United States [medquad_ans_0000037_4-1].

**Supported citations**: `medquad_ans_0000037_4-1`, `medquad_ans_0000010-10`
**Unsupported citations**: _none_

### Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Scores**: context_relevance=1.000, groundedness=0.514, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000544-2` (score=0.4783) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.
2. `medquad_ans_0000544-4` (score=0.4486) - Both types of junctional epidermolysis bullosa are inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive...
3. `medquad_ans_0003317-2` (score=0.4480) - What are the signs and symptoms of Junctional epidermolysis bullosa? The Human Phenotype Ontology provides the following list of signs and symptoms for Junctional epidermolysis bullosa. If the information is available...
4. `medquad_ans_0002122-1` (score=0.4463) - Epidermolysis bullosa acquisita (EBA) is a rare autoimmune disorder that causes the skin to blister in response to minor injury. Common areas of blistering include the hands, feet, knees, elbows, and buttocks. It can ...
5. `medquad_ans_0000544-3` (score=0.4452) - Junctional epidermolysis bullosa results from mutations in the LAMA3, LAMB3, LAMC2, and COL17A1 genes. Mutations in each of these genes can cause Herlitz JEB or non-Herlitz JEB. LAMB3 gene mutations are the most commo...

**Answer**:

Junctional epidermolysis bullosa is a rare genetic disorder that affects fewer than 1 per million people in the United States. It is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations. The parents of an individual with this condition each carry one copy of the mutated gene but typically do not show signs or symptoms of the condition. Common signs and symptoms include abnormal blistering of the skin (seen in 90% of patients), abnormality of dental enamel (90%), abnormality of the fingernails (90%), abnormality of the toenails (90%), and aplasia/hypoplasia of the skin (90%). Other symptoms include abnormality of the stomach (50%), duodenal stenosis (50%), pruritus (50%), subcutaneous hemorrhage (50%), and various other less common features such as

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Scores**: context_relevance=0.900, groundedness=0.908, answer_relevance=0.740

**Top docs**:

1. `medquad_ans_0000668-4` (score=0.5367) - Molybdenum cofactor deficiency has an autosomal recessive pattern of inheritance, which means both copies of the gene in each cell have mutations. An affected individual usually inherits one altered copy of the gene f...
2. `medquad_ans_0000668-3` (score=0.5109) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same ...
3. `medquad_ans_0000477-3` (score=0.4696) - Hereditary xanthinuria type I is caused by mutations in the XDH gene. This gene provides instructions for making an enzyme called xanthine dehydrogenase. This enzyme is involved in the normal breakdown of purines, whi...
4. `medquad_ans_0000416-3` (score=0.4468) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
5. `medquad_ans_0000342-4` (score=0.4443) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. These mutations result in reduced or eliminated function of the associated protein, which impairs the biosynthesis of the molybdenum cofactor. The three forms of the disorder—types A, B, and C—are distinguished by their genetic cause: MOCS1 gene mutations cause type A, MOCS2 gene mutations cause type B, and GPHN gene mutations cause type C [medquad_ans_0000668-3]. Additionally, in some cases, an individual may inherit two mutated copies of the MOCS1 gene through a mechanism called uniparental isodisomy, where the child receives two copies of the mutated gene from one parent due to an error in the formation of egg or sperm cells [medquad_ans_0000668-4].

**Supported citations**: `medquad_ans_0000668-3`, `medquad_ans_0000668-4`
**Unsupported citations**: _none_

### Example 12 (`0000756-1`)

**Question**: What is (are) ornithine translocase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.428, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000756-2` (score=0.5597) - Ornithine translocase deficiency is a very rare disorder. Fewer than 100 affected individuals have been reported worldwide.
2. `medquad_ans_0000756-1` (score=0.5304) - Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The...
3. `medquad_ans_0000756-3` (score=0.5109) - Mutations in the SLC25A15 gene cause ornithine translocase deficiency. Ornithine translocase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions tha...
4. `medquad_ans_0004582-1` (score=0.5077) - What are the signs and symptoms of Ornithine translocase deficiency syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Ornithine translocase deficiency syndrome. If the inform...
5. `medquad_ans_0004581-2` (score=0.4988) - What are the signs and symptoms of Ornithine transcarbamylase deficiency? Ornithine transcarbamylase (OTC) deficiency often becomes evident in the first few days of life. An infant with OTC deficiency may be lacking i...

**Answer**:

Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood due to a failure in the urea cycle. This condition results from mutations in the SLC25A15 gene, which provide instructions for making a mitochondrial ornithine transporter. This protein is essential for moving ornithine across the inner membrane of mitochondria to the mitochondrial matrix, where it participates in the urea cycle. When mutated, the transporter is unstable or misshapen and cannot transport ornithine effectively, leading to toxic buildup of ammonia, especially when protein breakdown increases. The disorder varies in severity and age of onset; infants may present with lethargy, poor feeding, breathing or temperature regulation issues, seizures, or coma, while later-onset cases are typically less severe and may involve intolerance to high-protein foods or episodes of vomiting, lethargy, ataxia, confusion, or blurred vision. Complications

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Scores**: context_relevance=0.857, groundedness=0.898, answer_relevance=0.971

**Top docs**:

1. `medquad_ans_0000586-1__dup2` (score=0.6612) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state ha...
2. `medquad_ans_0000362-1__dup2` (score=0.6334) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
3. `medquad_ans_0000431-1__dup1` (score=0.6334) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...
4. `medquad_ans_0000869-1__dup1` (score=0.6191) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can brin...
5. `medquad_ans_0000001-7__dup1` (score=0.6142) - Topics

**Answer**:

Yes, there is information about financial assistance. Financial assistance may be available if you do not have health insurance or need help with costs that are not covered. Certain government programs and nonprofit organizations can provide such assistance. You can also discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2]. Additionally, government health insurance programs like Medicaid can help low-income individuals pay their medical bills, with state-specific rules regarding income, resources, age, disability, pregnancy, and immigration status [medquad_ans_0000586-1__dup2].

**Supported citations**: `medquad_ans_0000362-1__dup2`, `medquad_ans_0000586-1__dup2`
**Unsupported citations**: _none_

### Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0004105-2` (score=0.5312) - What are the symptoms of Mollaret meningitis? The symptoms of Mollaret meningitis are the same as those found in other types of meningitis. In Mollaret meningitis, however, the symptoms are recurring and are often acc...
2. `medquad_ans_0004105-1` (score=0.5071) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About hal...
3. `medquad_ans_0000599-1__dup1` (score=0.4718) - Meningococci are a type of bacteria that cause serious infections. The most common infection is meningitis, which is an inflammation of the thin tissue that surrounds the brain and spinal cord. Meningococci can also c...
4. `medquad_ans_0000598-1__dup2` (score=0.4552) - Meningitis is inflammation of the thin tissue that surrounds the brain and spinal cord, called the meninges. There are several types of meningitis. The most common is viral meningitis, which you get when a virus enter...
5. `medquad_ans_0000115-1__dup1` (score=0.4394) - Meningitis is an infection of the meninges, the membranes that surround the brain and spinal cord. Encephalitis is inflammation of the brain itself. Causes of encephalitis and meningitis include viruses, bacteria, fun...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches, separated by weeks or months of no symptoms. About half of affected individuals may also experience long-term abnormalities of the nervous system that come and go, such as seizures, double vision, abnormal reflexes, paralysis of a cranial nerve (paresis), hallucinations, or coma. The exact cause remains unknown, although recent data suggests that herpes simplex virus (HSV-2 and, less frequently, HSV-1) may cause some, if not most cases. Other potential causes include trauma and viral infections other than herpes simplex [medquad_ans_0004105-1]. The symptoms of Mollaret meningitis are the same as those found in other types of meningitis, including high fever, severe headache, nausea, vomiting, stiff neck, photophobia

**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

### Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003319-1` (score=0.5072) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. U...
2. `medquad_ans_0003331-1` (score=0.5049) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
3. `medquad_ans_0000052-1` (score=0.5007) - Amyotrophic lateral sclerosis (ALS) is a progressive disease that affects motor neurons, which are specialized nerve cells that control muscle movement. These nerve cells are found in the spinal cord and the brain. In...
4. `medquad_ans_0003319-2` (score=0.4938) - What are the signs and symptoms of juvenile amyotrophic lateral sclerosis? Signs and symptoms of juvenile ALS vary but include slowly to very slowly progressive muscle weakness, increased muscle tone, Babinski reflex,...
5. `medquad_ans_0003331-3` (score=0.4829) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...

**Answer**:

Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be inherited in an autosomal dominant or autosomal recessive fashion [medquad_ans_0003319-1].

**Supported citations**: `medquad_ans_0003319-1`
**Unsupported citations**: _none_

### Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Scores**: context_relevance=0.889, groundedness=0.935, answer_relevance=0.711

**Top docs**:

1. `medquad_ans_0000219-11` (score=0.5591) - - Erectile dysfunction (ED) is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last...
2. `medquad_ans_0000334-1__dup2` (score=0.5359) - Erectile dysfunction (ED) is a common type of male sexual dysfunction. It is when a man has trouble getting or keeping an erection. ED becomes more common as you get older. But it's not a natural part of aging. Some p...
3. `medquad_ans_0000219-8` (score=0.5309) - Your doctor can offer you a number of treatments for ED. For many men, the answer is as simple as taking a pill. Other men have to try two or three options before they find a treatment that works for them. Dont give u...
4. `medquad_ans_0000809-1__dup2` (score=0.5301) - Summary : Many men have sexual problems. They become more common as men age. Problems can include - Erectile dysfunction - Reduced or lost interest in sex - Problems with ejaculation - Low testosterone Stress, illness...
5. `medquad_ans_0000219-7` (score=0.5287) - To find the cause of your ED, your doctor may - take your medical and sexual history - ask you questions about your mental health - give you a physical exam - test your blood - give you a nighttime erection test - per...

**Answer**:

To understand erectile dysfunction (ED), know that it is when you cannot get or keep an erection firm enough for sex. This can occur if you sometimes get an erection, but it does not last long enough, or if you are unable to get an erection at all [medquad_ans_0000219-11]. An erection happens when blood flow into the penis increases, involving hormones, blood vessels, nerves, and muscles working together [medquad_ans_0000219-11]. ED often has multiple causes, such as diseases that damage nerves, arteries, or muscles [medquad_ans_0000219-11].

To determine the cause, your doctor may take your medical and sexual history, ask about your mental health, perform a physical exam, test your blood, conduct a nighttime erection test, or perform an injection test or Doppler penile ultrasound [medquad_ans

**Supported citations**: `medquad_ans_0000219-11`
**Unsupported citations**: _none_

### Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.962, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0006267-3` (score=0.5172) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
2. `medquad_ans_0000981-5` (score=0.5029) - These resources address the diagnosis or management of tibial muscular dystrophy: - Gene Review: Gene Review: Udd Distal Myopathy - Genetic Testing Registry: Distal myopathy Markesbery-Griggs type These resources from...
3. `medquad_ans_0000576-5` (score=0.5023) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...
4. `medquad_ans_0000395-5` (score=0.4994) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
5. `medquad_ans_0000202-5` (score=0.4900) - These resources address the diagnosis or management of collagen VI-related myopathy: - Gene Review: Gene Review: Collagen Type VI-Related Disorders - Genetic Testing Registry: Bethlem myopathy - Genetic Testing Regist...

**Answer**:

Physical therapy, including early mobilization, regular stretching, and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation often become necessary in the first or second decade of life. Prevention of chest infections may be achieved with the use of antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that people with Ullrich congenital muscular dystrophy may benefit from cyclosporin A, although more studies are needed to confirm its benefits [medquad_ans_0006267-3].

**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

### Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Scores**: context_relevance=0.250, groundedness=0.129, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5660) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000043-4` (score=0.5645) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000154-4` (score=0.5645) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000957-4` (score=0.5645) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000829-4` (score=0.5645) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether microcephalic osteodysplastic primordial dwarfism type II is inherited. While several documents describe autosomal recessive inheritance patterns in general, no specific mention is made of microcephalic osteodysplastic primordial dwarfism type II. Therefore, based on the given information, the inheritance pattern for this specific condition cannot be determined.

Answer: Insufficient context to determine if microcephalic osteodysplastic primordial dwarfism type II is inherited.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.6181) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000385-4` (score=0.6148) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0001090-4` (score=0.6148) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0001074-4` (score=0.6148) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000269-4` (score=0.6148) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Ochoa syndrome or its mode of inheritance. Therefore, it is not possible to determine based on the given information whether Ochoa syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Scores**: context_relevance=0.500, groundedness=0.731, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000531-5` (score=0.5910) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management...
2. `medquad_ans_0001071-5` (score=0.5273) - These resources address the diagnosis or management of 15q24 microdeletion: - Gene Review: Gene Review: 15q24 Microdeletion - Genetic Testing Registry: 15q24 deletion syndrome These resources from MedlinePlus offer in...
3. `medquad_ans_0001070-5` (score=0.5268) - These resources address the diagnosis or management of 15q13.3 microdeletion: - Gene Review: Gene Review: 15q13.3 Microdeletion - Genetic Testing Registry: 15q13.3 microdeletion syndrome These resources from MedlinePl...
4. `medquad_ans_0000878-5` (score=0.4955) - These resources address the diagnosis or management of ring chromosome 14 syndrome: - Genetic Testing Registry: Ring chromosome 14 - MedlinePlus Encyclopedia: Chromosome These resources from MedlinePlus offer informat...
5. `medquad_ans_0001077-5` (score=0.4919) - These resources address the diagnosis or management of 18q deletion syndrome: - Gene Review: Gene Review: Leukodystrophy Overview - University of Texas Chromosome 18 Clinical Research Center These resources from Medli...

**Answer**:

The provided context does not specify treatments for isodicentric chromosome 15 syndrome. While it mentions that resources from MedlinePlus cover diagnosis and management of various health conditions, including diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care, no specific treatments are outlined for isodicentric chromosome 15 syndrome [medquad_ans_0000531-5]. Therefore, based on the given information, the exact treatments are not available.

**Supported citations**: `medquad_ans_0000531-5`
**Unsupported citations**: _none_

### Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.749, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006224-2` (score=0.5716) - What are the signs and symptoms of Troyer syndrome? The signs and symptoms of Troyer syndrome can vary, and some people are more severely affected than others. Symptoms typically begin in early childhood. Most affecte...
2. `medquad_ans_0000998-2` (score=0.5614) - About 20 cases of Troyer syndrome have been reported in the Old Order Amish population of Ohio. It has not been found outside this population.
3. `medquad_ans_0000272-2__dup1` (score=0.5486) - There are no specific treatments to prevent or slow the progressive degeneration seen in Troyer syndrome. Symptomatic therapy includes antispasmodic drugs and physical therapy to improve muscle strength and maintain r...
4. `medquad_ans_0000998-1` (score=0.5454) - Troyer syndrome is part of a group of genetic disorders known as hereditary spastic paraplegias. These disorders are characterized by progressive muscle stiffness (spasticity) and the development of paralysis of the l...
5. `medquad_ans_0000272-3__dup1` (score=0.5373) - Prognosis varies, although the disease is progressive. Some patients may have a mild form of the disease while others eventually lose the ability to walk normally. Troyer syndrome does not shorten the normal life span.

**Answer**:

The provided context does not specify the cause of Troyer syndrome. [medquad_ans_0006224-2] describes the signs and symptoms, inheritance pattern (autosomal recessive), and associated features but does not mention the underlying genetic or biological cause. Therefore, based on the given information, the cause of Troyer syndrome cannot be determined.

**Supported citations**: `medquad_ans_0006224-2`
**Unsupported citations**: _none_

### Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.927, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000983-3` (score=0.5539) - Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. This gene provides instructions for making a protein that acts as a channel across cell membranes. This channel, known as CaV1....
2. `medquad_ans_0003672-1` (score=0.4689) - Timothy syndrome is a type of long QT syndrome. It affects many parts of the body including the heart, fingers, toes, face, and the nervous system. It is characterized by long QT syndrome, although some people with Ti...
3. `medquad_ans_0000140-3` (score=0.4617) - Brugada syndrome can be caused by mutations in one of several genes. The most commonly mutated gene in this condition is SCN5A, which is altered in approximately 30 percent of affected individuals. This gene provides ...
4. `medquad_ans_0000983-4` (score=0.4535) - This condition is considered to have an autosomal dominant pattern of inheritance, which means one copy of the altered CACNA1C gene in each cell is sufficient to cause the disorder. Most cases result from new mutation...
5. `medquad_ans_0000159-3` (score=0.4528) - CPVT can result from mutations in two genes, RYR2 and CASQ2. RYR2 gene mutations cause about half of all cases, while mutations in the CASQ2 gene account for 1 percent to 2 percent of cases. In people without an ident...

**Answer**:

The genetic changes related to Timothy syndrome are mutations in the CACNA1C gene. These mutations change the structure of CaV1.2 calcium channels, causing them to stay open much longer than usual and leading to an abnormal influx of calcium ions into cells. All reported cases of Timothy syndrome are due to changes in the CACNA1C gene [medquad_ans_0000983-3]. Additionally, Timothy syndrome is inherited in an autosomal dominant manner, although most cases occur spontaneously due to a random mutation in the CACNA1C gene [medquad_ans_0003672-1].

**Supported citations**: `medquad_ans_0000983-3`, `medquad_ans_0003672-1`
**Unsupported citations**: _none_

### Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Scores**: context_relevance=1.000, groundedness=0.743, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005261-2` (score=0.5928) - What are the signs and symptoms of Reactive arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Reactive arthritis. If the information is available, the table below includes h...
2. `medquad_ans_0004326-1` (score=0.5698) - What are the signs and symptoms of Negative rheumatoid factor polyarthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Negative rheumatoid factor polyarthritis. If the informat...
3. `medquad_ans_0000061-11` (score=0.5682) - Rheumatoid arthritis can be difficult to diagnose in its early stages because the full range of symptoms develops over time, and only a few symptoms may be present in the early stages. As part of the diagnosis, your d...
4. `medquad_ans_0005651-2` (score=0.5624) - What are the signs and symptoms of Sjogren syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Sjogren syndrome. If the information is available, the table below includes how o...
5. `medquad_ans_0005361-1` (score=0.5560) - What are the signs and symptoms of Rheumatoid nodulosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Rheumatoid nodulosis. If the information is available, the table below includ...

**Answer**:

The provided context does not contain information about the symptoms of Rheumatoid Arthritis. The context discusses symptoms of Reactive arthritis and Negative rheumatoid factor polyarthritis, but not Rheumatoid Arthritis as a whole. Therefore, based on the given information, the symptoms of Rheumatoid Arthritis cannot be determined.

[medquad_ans_0005261-2] [medquad_ans_0004326-1]

**Supported citations**: `medquad_ans_0005261-2`, `medquad_ans_0004326-1`
**Unsupported citations**: _none_

### Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Scores**: context_relevance=1.000, groundedness=0.518, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0002796-1` (score=0.5371) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is no...
2. `medquad_ans_0000149-1__dup3` (score=0.5179) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, d...
3. `medquad_ans_0002796-2` (score=0.5144) - What are the signs and symptoms of Henoch-Schonlein purpura? The Human Phenotype Ontology provides the following list of signs and symptoms for Henoch-Schonlein purpura. If the information is available, the table belo...
4. `medquad_ans_0002796-4` (score=0.5056) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...
5. `medquad_ans_0000149-8` (score=0.4949) - - Henoch-Schnlein purpura (HSP) is a disease that causes small blood vessels in the body to become inflamed and leak. - HSP is caused by an abnormal immune system response in which the bodys immune system attacks the ...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). It is caused by inflammation of small blood vessels in the body, leading to leakage. The primary symptom is a rash that appears as many small raised bruises. HSP can affect the kidneys, digestive tract, and joints. It is most common in children between 2 and 6 years of age but can occur at any age. Most people recover completely, though kidney damage is the most likely long-term complication. In adults, HSP can lead to chronic kidney disease (CKD) and kidney failure, including end-stage renal disease requiring dialysis or kidney transplant. Common signs and symptoms include abdominal pain (90%), arthralgia (90%), bruising susceptibility (90%), hematuria (9

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Scores**: context_relevance=1.000, groundedness=0.967, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0003166-1` (score=0.5778) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information ...
2. `medquad_ans_0003163-1` (score=0.5470) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information ...
3. `medquad_ans_0001446-2` (score=0.5415) - What are the signs and symptoms of Common variable immunodeficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Common variable immunodeficiency. If the information is available...
4. `medquad_ans_0006509-2` (score=0.5309) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is...
5. `medquad_ans_0005560-1` (score=0.5294) - What are the signs and symptoms of Severe combined immunodeficiency, atypical? The Human Phenotype Ontology provides the following list of signs and symptoms for Severe combined immunodeficiency, atypical. If the info...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [medquad_ans_0003166-1].

**Supported citations**: `medquad_ans_0003166-1`
**Unsupported citations**: _none_

### Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0000896-3` (score=0.5129) - Senior-Lken syndrome can be caused by mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopic, finger...
2. `medquad_ans_0000606-3` (score=0.4785) - Mutations in the OCRL gene cause Lowe syndrome. The OCRL gene provides instructions for making an enzyme that helps modify fat (lipid) molecules called membrane phospholipids. By controlling the levels of specific mem...
3. `medquad_ans_0000582-3` (score=0.4718) - Mutations in the FLNB gene cause Larsen syndrome. The FLNB gene provides instructions for making a protein called filamin B. This protein helps build the network of protein filaments (cytoskeleton) that gives structur...
4. `medquad_ans_0005069-3` (score=0.4683) - What genes are related to Hutchinson-Gilford progeria syndrome? Mutations in the LMNA gene cause Hutchinson-Gilford progeria syndrome. The LMNA gene provides instructions for making a protein called lamin A. This prot...
5. `medquad_ans_0000174-3` (score=0.4666) - Mutations in the CHD7 gene cause more than half of all cases of CHARGE syndrome. The CHD7 gene provides instructions for making a protein that most likely regulates gene activity (expression) by a process known as chr...

**Answer**:

Genetic changes related to Senior-Lken syndrome include mutations in one of at least five genes. These mutations likely lead to problems with the structure and function of cilia, which are microscopic projections from cells involved in signaling pathways. The proteins produced from these genes are known or suspected to play roles in cilia, and defects in cilia are believed to disrupt important chemical signaling pathways within cells. However, it remains unclear how these defects specifically lead to nephronophthisis and Leber congenital amaurosis. Some individuals with Senior-Lken syndrome do not have identified mutations in these five genes, and in such cases, the genetic cause is unknown [medquad_ans_0000896-3].

**Supported citations**: `medquad_ans_0000896-3`
**Unsupported citations**: _none_

### Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Scores**: context_relevance=0.125, groundedness=0.719, answer_relevance=0.525

**Top docs**:

1. `medquad_ans_0000359-2` (score=0.6114) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
2. `medquad_ans_0000989-2` (score=0.5960) - This condition affects an estimated 1 in 50,000 people.
3. `medquad_ans_0000650-2` (score=0.5949) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
4. `medquad_ans_0000790-2` (score=0.5773) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
5. `medquad_ans_0000874-2` (score=0.5765) - This condition affects an estimated 1 in 8,500 females.

**Answer**:

The prevalence of lymphedema-distichiasis syndrome varies by source: it affects about 1 per million people worldwide [medquad_ans_0000359-2], an estimated 1 in 50,000 people [medquad_ans_0000989-2], or 1 in 50,000 to 100,000 people [medquad_ans_0000650-2]. One estimate ranges from 1 in 25,000 to 300,000 individuals [medquad_ans_0000790-2], and another specifically notes it affects an estimated 1 in 8,500 females [medquad_ans_0000874-2]. Due to the range of estimates, the exact number of people affected is uncertain. [

**Supported citations**: `medquad_ans_0000359-2`, `medquad_ans_0000989-2`, `medquad_ans_0000650-2`, `medquad_ans_0000790-2`, `medquad_ans_0000874-2`
**Unsupported citations**: _none_

### Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Scores**: context_relevance=0.857, groundedness=0.877, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000013-25` (score=0.5743) - Genetic testing may be helpful whether the test identifies a mutation or not. Test results can - serve as a relief, eliminating some of the uncertainty about a health condition - help doctors make recommendations for ...
2. `medquad_ans_0000397-1__dup1` (score=0.5457) - Summary : Genetic counseling provides information and support to people who have, or may be at risk for, genetic disorders. A genetic counselor meets with you to discuss genetic risks. The counseling may be for yourse...
3. `medquad_ans_0000013-26` (score=0.5431) - Finding out your test results can affect you emotionally. Learning that you are someone in your family has or is at risk for a disease can be scary. Some people can also feel guilty, angry, anxious, or depressed when ...
4. `medquad_ans_0000399-1__dup1` (score=0.5388) - Summary : Genetic tests are tests on blood and other tissue to find genetic disorders. Over 2000 tests are available. Doctors use genetic tests for several reasons. These include - Finding genetic diseases in unborn b...
5. `medquad_ans_0001717-5` (score=0.5341) - Is genetic testing available for Danon disease? Yes. GeneTests lists laboratories offering clinical genetic testing for Danon disease. Clinical genetic tests are ordered to help diagnose a person or family and to aid ...

**Answer**:

Yes, there is information about genetic testing. Genetic testing may be helpful whether the test identifies a mutation or not. It can serve as a relief by eliminating uncertainty about a health condition, help doctors make treatment or monitoring recommendations, provide information for individuals and families to make health-related decisions, assist in taking steps to lower the chance of developing a disease (such as through earlier screening or lifestyle changes), and support informed choices about the future, such as whether to have a baby [medquad_ans_0000013-25]. However, genetic testing has emotional impacts, including fear, guilt, anger, anxiety, or depression, and costs can range from less than $100 to more than $2,000, with insurance potentially covering part or all of the cost [medquad_ans_0000013-26]. Additionally, genetic testing cannot predict with certainty whether a person will develop a disease

**Supported citations**: `medquad_ans_0000013-25`, `medquad_ans_0000013-26`
**Unsupported citations**: _none_

### Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.993, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000654-1` (score=0.5111) - Microcephaly-capillary malformation syndrome is an inherited disorder characterized by an abnormally small head size (microcephaly) and abnormalities of small blood vessels in the skin called capillaries (capillary ma...
2. `medquad_ans_0000149-1` (score=0.5061) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxy...
3. `medquad_ans_0003726-1` (score=0.4973) - What are the signs and symptoms of Macrocephaly-capillary malformation? The Human Phenotype Ontology provides the following list of signs and symptoms for Macrocephaly-capillary malformation. If the information is ava...
4. `medquad_ans_0000056-1__dup3` (score=0.4733) - Arteriovenous malformations (AVMs) are defects in your vascular system. The vascular system includes arteries, veins, and capillaries. Arteries carry blood away from the heart to other organs; veins carry blood back t...
5. `medquad_ans_0003238-1` (score=0.4711) - Intracranial arteriovenous malformations (AVMs) are abnormal connections between the arteries and veins in the brain. Most people with brain or spinal AVMs experience few, if any, major symptoms. About 12 percent of p...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. It is characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin. These malformations appear as multiple small, round, pink or red spots on the skin and may be visible from birth or develop during childhood. In most affected individuals, capillary malformations occur on the face, arms, and legs. By themselves, capillary malformations usually do not cause any health problems, and in some people with CM-AVM, they are the only sign of the disorder [medquad_ans_0000149-1].

**Supported citations**: `medquad_ans_0000149-1`
**Unsupported citations**: _none_

### Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Scores**: context_relevance=1.000, groundedness=0.654, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001986-1` (score=0.5863) - What are the signs and symptoms of Dystonia 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 1. If the information is available, the table below includes how often the sym...
2. `medquad_ans_0001994-1` (score=0.5819) - What are the signs and symptoms of Dystonia 18? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 18. If the information is available, the table below includes how often the s...
3. `medquad_ans_0001988-1` (score=0.5745) - What are the signs and symptoms of Dystonia 11? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 11. If the information is available, the table below includes how often the s...
4. `medquad_ans_0001992-1` (score=0.5697) - What are the signs and symptoms of Dystonia 16? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 16. If the information is available, the table below includes how often the s...
5. `medquad_ans_0001995-1` (score=0.5686) - What are the signs and symptoms of Dystonia 19? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 19. If the information is available, the table below includes how often the s...

**Answer**:

The provided context does not contain information about the symptoms of Dystonia 16. [medquad_ans_0001986-1] and [medquad_ans_0001994-1] discuss symptoms of Dystonia 1 and Dystonia 18, respectively, but there is no mention of Dystonia 16. Therefore, the symptoms of Dystonia 16 cannot be determined from the given context.

**Supported citations**: `medquad_ans_0001986-1`, `medquad_ans_0001994-1`
**Unsupported citations**: _none_

### Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Scores**: context_relevance=0.714, groundedness=0.980, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000899-5` (score=0.5794) - These resources address the diagnosis or management of septo-optic dysplasia: - Genetic Testing Registry: Septo-optic dysplasia sequence - MedlinePlus Encyclopedia: Growth Hormone Deficiency - MedlinePlus Encyclopedia...
2. `medquad_ans_0005548-5` (score=0.5612) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not tr...
3. `medquad_ans_0000745-5` (score=0.5377) - These resources address the diagnosis or management of oculodentodigital dysplasia: - Genetic Testing Registry: Oculodentodigital dysplasia - MedlinePlus Encyclopedia: Webbing of the fingers or toes - UC Davis Childre...
4. `medquad_ans_0000010-5` (score=0.5251) - These resources address the diagnosis or management of acromicric dysplasia: - Genetic Testing Registry: Acromicric dysplasia These resources from MedlinePlus offer information about the diagnosis and management of va...
5. `medquad_ans_0000150-2__dup1` (score=0.5149) - There is no standard course of treatment for holoprosencephaly. Treatment is symptomatic and supportive.

**Answer**:

Treatment for septo-optic dysplasia is symptomatic and supportive. Hormone deficiencies may be treated with hormone replacement therapy. Optical problems are generally not treatable, and vision, physical, and occupational therapies may be required [medquad_ans_0005548-5].

**Supported citations**: `medquad_ans_0005548-5`
**Unsupported citations**: _none_

### Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Scores**: context_relevance=0.900, groundedness=0.979, answer_relevance=0.660

**Top docs**:

1. `medquad_ans_0000208-14` (score=0.5045) - - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. - The prostate is a walnut-shaped gland that is part of the male reproductive system. - The...
2. `medquad_ans_0000208-11` (score=0.4921) - The complications of benign prostatic hyperplasia treatment depend on the type of treatment. Medications Medications used to treat benign prostatic hyperplasia may have side effects that sometimes can be serious. Men ...
3. `medquad_ans_0000213-3__dup2` (score=0.4834) - Urinary retention can result from - obstruction of the urethra - nerve problems - medications - weakened bladder muscles Obstruction of the Urethra Obstruction of the urethra causes urinary retention by blocking the n...
4. `medquad_ans_0000208-7` (score=0.4732) - The complications of benign prostatic hyperplasia may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney da...
5. `medquad_ans_0000211-4__dup2` (score=0.4601) - Urinary incontinence in men results when the brain does not properly signal the bladder, the sphincters do not squeeze strongly enough, or both. The bladder muscle may contract too much or not enough because of a prob...

**Answer**:

The complications of benign prostatic hyperplasia may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney damage - bladder stones [medquad_ans_0000208-14].

**Supported citations**: `medquad_ans_0000208-14`
**Unsupported citations**: _none_

### Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Scores**: context_relevance=0.444, groundedness=0.000, answer_relevance=0.711

**Top docs**:

1. `medquad_ans_0000150-2__dup1` (score=0.4958) - There is no standard course of treatment for holoprosencephaly. Treatment is symptomatic and supportive.
2. `medquad_ans_0000019-2__dup1` (score=0.4802) - There is no cure or standard treatment for anencephaly. Treatment is supportive.
3. `medquad_ans_0000004-2__dup1` (score=0.4645) - Treatment for LKS usually consists of medications, such as anticonvulsants and corticosteroids, and speech therapy, which should be started early. A controversial treatment option involves a surgical technique called ...
4. `medquad_ans_0000899-5` (score=0.4520) - These resources address the diagnosis or management of septo-optic dysplasia: - Genetic Testing Registry: Septo-optic dysplasia sequence - MedlinePlus Encyclopedia: Growth Hormone Deficiency - MedlinePlus Encyclopedia...
5. `medquad_ans_0000189-2__dup1` (score=0.4504) - There is no standard treatment for megalencephaly. Treatment will depend upon the disorder with which the megalencephaly is associated and will address individual symptoms and disabilities.

**Answer**:

The provided context does not contain information about the treatments for the absence of the septum pellucidum. Therefore, a specific answer cannot be given based on the available information. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Scores**: context_relevance=1.000, groundedness=0.512, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.5244) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (...
2. `medquad_ans_0000114-2__dup1` (score=0.5147) - What are the signs and symptoms of Acute febrile neutrophilic dermatosis? The most obvious signs of acute febrile neutrophilic dermatosis are distinctive skin lesions that usually develop according to a specific patte...
3. `medquad_ans_0003395-2` (score=0.5101) - What are the signs and symptoms of Klatskin tumor? The symptoms associated with Klatskin tumors are usually due to blocked bile ducts. Symptoms may include: Jaundice Itching Light colored stools and/or dark urine Abdo...
4. `medquad_ans_0004428-2` (score=0.5066) - What are the signs and symptoms of Nodular nonsuppurative panniculitis? The Human Phenotype Ontology provides the following list of signs and symptoms for Nodular nonsuppurative panniculitis. If the information is ava...
5. `medquad_ans_0005261-2` (score=0.5061) - What are the signs and symptoms of Reactive arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Reactive arthritis. If the information is available, the table below includes h...

**Answer**:

The symptoms of Febrile Ulceronecrotic Mucha-Habermann disease (FUMHD) include initial red scaly skin lesions (papules) that ulcerate, break down, form open sores, and then develop a red-brown crust (PLEVA). These lesions can progress rapidly to large, destructive ulcers with extensive, painful loss of skin tissue. Skin lesions may become infected, leading to pus and a putrid odor. FUMHD is often associated with high fever (up to 104°F), which may be persistent or intermittent. Other symptoms may include feeling ill, sore throat, congestion, muscle soreness or pain, joint pain, diarrhea, central nervous system symptoms, abdominal pain, enlarged spleen, arthritis, megaloblastic anemia, interstitial pneumonitis, lymphocytic (viral) myocarditis, and sepsis. FUMHD can become life threatening [

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0002833-4` (score=0.5456) - How is hereditary leiomyomatosis and renal cell cancer inherited? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the ...
2. `medquad_ans_0000785-4` (score=0.5171) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000817-4` (score=0.5129) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000695-4` (score=0.5129) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000642-4` (score=0.5129) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

Hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the gene in each cell is enough to cause symptoms of the condition [medquad_ans_0002833-4].

**Supported citations**: `medquad_ans_0002833-4`
**Unsupported citations**: _none_

### Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002445-1` (score=0.6152) - What are the signs and symptoms of Frontometaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontometaphyseal dysplasia. If the information is available, the tabl...
2. `medquad_ans_0001626-1` (score=0.5966) - What are the signs and symptoms of Craniodiaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Craniodiaphyseal dysplasia. If the information is available, the table ...
3. `medquad_ans_0005135-1` (score=0.5812) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
4. `medquad_ans_0002145-1` (score=0.5806) - What are the signs and symptoms of Epiphyseal dysplasia hearing loss dysmorphism? The Human Phenotype Ontology provides the following list of signs and symptoms for Epiphyseal dysplasia hearing loss dysmorphism. If th...
5. `medquad_ans_0005857-1` (score=0.5802) - What are the signs and symptoms of Spondylometaphyseal dysplasia X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia X-linked. If the information ...

**Answer**:

The symptoms of Frontometaphyseal dysplasia include abnormality of dental morphology, abnormality of frontal sinus, abnormality of the metaphyses, bowing of the long bones, camptodactyly of finger, craniofacial hyperostosis, hypertelorism, limitation of joint mobility, prominent supraorbital ridges, abnormal form of the vertebral bodies, abnormality of the palate, accelerated skeletal maturation, aplasia/hypoplasia of the thumb, arachnodactyly, conductive hearing impairment, elbow dislocation, scoliosis, sensorineural hearing impairment, skeletal muscle atrophy, synostosis of carpal bones, ulnar deviation of finger, abnormality of the larynx, abnormality of the urethra, complete atrioventricular canal defect, craniosynostosis, tracheal stenosis, ureteral stenosis, ankle

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Scores**: context_relevance=0.818, groundedness=0.928, answer_relevance=0.527

**Top docs**:

1. `medquad_ans_0000128-7` (score=0.5665) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding,...
2. `medquad_ans_0000097-8` (score=0.5286) - A health care provider may treat the symptoms of diverticulosis with a high-fiber diet or fiber supplements, medications, and possibly probiotics. Treatment for diverticular disease varies, depending on whether a pers...
3. `medquad_ans_0000128-9` (score=0.5109) - A doctor may prescribe antibiotics and recommend following a liquid diet. Most people get better with this treatment. Some people may need surgery and other treatments. - Surgery. Serious problems from diverticulitis ...
4. `medquad_ans_0000128-11` (score=0.4909) - - Diverticular disease is more common in people as they grow older. - A low-fiber diet is the most likely cause of the disease. - Most people are treated with a high-fiber diet and pain medication. - Add whole grain f...
5. `medquad_ans_0000293-1__dup1` (score=0.4889) - Diverticula are small pouches that bulge outward through the colon, or large intestine. If you have these pouches, you have a condition called diverticulosis. It becomes more common as people age. About half of all pe...

**Answer**:

Treatment for diverticular disease depends on whether the condition is diverticulosis or diverticulitis, and the severity of symptoms. For diverticulosis, a high-fiber diet is recommended, as studies show it can help prevent disease progression and manage symptoms. A health care provider may suggest a slow increase in dietary fiber to minimize gas and abdominal discomfort [medquad_ans_0000097-8]. Fiber supplements such as methylcellulose (Citrucel) or psyllium (Metamucil) may be prescribed, with doses of 0.5 to 3.5 grams per serving, and should be taken with at least 8 ounces of water [medquad_ans_0000097-8]. Medications like mesalazine (Asacol), either continuously or in cycles, may reduce abdominal pain and gastrointestinal symptoms; combining mesalazine with rifaximin (

**Supported citations**: `medquad_ans_0000097-8`
**Unsupported citations**: _none_

### Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.711, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0006066-1` (score=0.6245) - What are the signs and symptoms of Thompson Baraitser syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thompson Baraitser syndrome. If the information is available, the tabl...
2. `medquad_ans_0000621-1__dup1` (score=0.5993) - What are the signs and symptoms of Baraitser-Winter syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser-Winter syndrome. If the information is available, the table be...
3. `medquad_ans_0000619-1__dup1` (score=0.5941) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is avail...
4. `medquad_ans_0004686-1` (score=0.5685) - What are the signs and symptoms of Pallister W syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pallister W syndrome. If the information is available, the table below includ...
5. `medquad_ans_0000623-2__dup1` (score=0.5662) - What are the signs and symptoms of Barber Say syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Barber Say syndrome. If the information is available, the table below includes...

**Answer**:

The provided context does not mention "Baraitser Brett Piesowicz syndrome" or any condition with a similar name. Therefore, based on the given information, the symptoms of Baraitser Brett Piesowicz syndrome cannot be determined.

Note: The context discusses "Thompson Baraitser syndrome," not "Baraitser Brett Piesowicz syndrome." There is no evidence in the provided text to support the existence or symptoms of the latter condition. [medquad_ans_0006066-1]

**Supported citations**: `medquad_ans_0006066-1`
**Unsupported citations**: _none_

### Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003911-2` (score=0.5530) - How might meningiomas be treated? The treatment varies depending on the location of the meningioma and the symptoms caused by the tumor. Careful observation is sometimes the best course of action for people with a men...
2. `medquad_ans_0000052-2__dup1` (score=0.5061) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
3. `medquad_ans_0000052-4__dup1` (score=0.4804) - Scientists continue to investigate ways to better understand, diagnose, and treat CNS tumors. Experimental treatment options may include new drugs, gene therapy, surgery , radiation, biologic modulators that enhance t...
4. `medquad_ans_0000060-2__dup1` (score=0.4773) - The primary treatment option for a CCM is surgical removal. Radiation therapy has not been shown to be effective. The decision to operate is made based upon the risk of approaching the lesion. For example, symptomatic...
5. `medquad_ans_0002753-4` (score=0.4732) - What treatment is available for meningeal hemangiopericytoma? Radical surgical resection with removal of all meningeal attachments is typically the preferred treatment. However this treatment option is generally possi...

**Answer**:

The treatment for meningiomas varies depending on the location of the meningioma and the symptoms it causes. Careful observation may be the best course of action for some individuals. When treatment is necessary, surgery and radiation are the most common forms of treatment. Radiation may be used if the meningioma cannot be operated on or if it is only partially removed by surgery. Radiation is also used in cases of malignant, atypical, or recurrent tumors. Other treatments that have been tried or are being explored include hydroxyurea, epidermal growth factor receptor inhibitors, platelet-derived growth factor receptor inhibitors, vascular endothelial growth factor inhibitors, immunotherapy to stimulate the immune system, and somatostatin analogs which prevent the release of growth hormones [medquad_ans_0003911-2]. Additionally, chemotherapy is among the three most commonly used treatments, although its use is less emphasized compared to surgery and radiation [med

**Supported citations**: `medquad_ans_0003911-2`
**Unsupported citations**: _none_

### Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Scores**: context_relevance=1.000, groundedness=0.880, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000006_6-8` (score=0.5309) - Key Points - There are different types of treatment for children who have central nervous system (CNS) embryonal tumors. - Children who have CNS embryonal tumors should have their treatment planned by a team of health...
2. `medquad_ans_0000006_7-8` (score=0.5306) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a te...
3. `medquad_ans_0000006_7-7` (score=0.5114) - Check the list of NCI-supported cancer clinical trials that are now accepting patients with childhood central nervous system germ cell tumor. For more specific results, refine the search by using other search features...
4. `medquad_ans_0000006_8-5` (score=0.5078) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The size of the tumor. - Where the tumor is in the brain...
5. `medquad_ans_0000006_3-9` (score=0.5038) - Key Points - There are different types of treatment for patients with childhood astrocytoma. - Children with astrocytomas should have their treatment planned by a team of health care providers who are experts in treat...

**Answer**:

The provided context discusses treatments for children who have central nervous system (CNS) embryonal tumors, but it does not specifically mention "Childhood Central Nervous System Germ Cell Tumors." Therefore, the treatments for Childhood Central Nervous System Germ Cell Tumors cannot be determined from the given information.

[medquad_ans_0000006_6-8]

**Supported citations**: `medquad_ans_0000006_6-8`
**Unsupported citations**: _none_

### Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Scores**: context_relevance=1.000, groundedness=0.500, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0004401-4` (score=0.5038) - How is neutral lipid storage disease with myopathy inherited? This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individua...
2. `medquad_ans_0004401-1` (score=0.4994) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (my...
3. `medquad_ans_0000729-2` (score=0.4924) - Neutral lipid storage disease with myopathy is a rare condition; its incidence is unknown.
4. `medquad_ans_0004401-3` (score=0.4909) - What causes neutral lipid storage disease with myopathy? Neutral lipid storage disease with myopathy is caused by mutations in the PNPLA2 gene. This gene provides instructions for making an enzyme called adipose trigl...
5. `medquad_ans_0004401-2` (score=0.4871) - What are the signs and symptoms of Neutral lipid storage disease with myopathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Neutral lipid storage disease with myopathy. If the in...

**Answer**:

Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (myopathy). This condition is caused by mutations in the PNPLA2 gene, which provides instructions for making an enzyme called adipose triglyceride lipase (ATGL). The ATGL enzyme plays a role in breaking down fats called triglycerides into simpler molecules called fatty acids, which can then be used for energy. Mutations in the PNPLA2 gene impair the ATGL enzyme's ability to break down triglycerides, leading to their accumulation in muscle and tissues. This results in the signs and symptoms of the condition, including muscle weakness, elevated serum creatine phosphokinase, exercise intolerance, and other related manifestations. The condition is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Scores**: context_relevance=0.800, groundedness=0.941, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0006546-1` (score=0.5186) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. ...
2. `medquad_ans_0000981-1__dup1` (score=0.5141) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been repo...
3. `medquad_ans_0006546-2` (score=0.4952) - How might a Zika virus infection be treated? There is no vaccine to prevent Zika virus infections, nor is there a specific medicine to treat Zika. Individuals infected with the Zika virus should get plenty of rest, dr...
4. `medquad_ans_0000174-1__dup3` (score=0.4651) - Chikungunya is a virus that spread by the same kinds of mosquitoes that spread dengue and Zika virus. Rarely, it can spread from mother to newborn around the time of birth. It may also possibly spread through infected...
5. `medquad_ans_0000970-1__dup1` (score=0.4225) - West Nile virus (WNV) is an infectious disease that first appeared in the United States in 1999. Infected mosquitoes spread the virus that causes it. People who get WNV usually have no symptoms or mild symptoms. The s...

**Answer**:

Zika virus is a virus spread primarily through the bite of an infected mosquito. It can also be transmitted from a pregnant mother to her child during pregnancy or around the time of birth, through sexual contact with an affected male partner, and through blood transfusions. The illness associated with Zika virus infection is usually mild, with symptoms such as fever, rash, joint pain, and conjunctivitis, which typically last for several days to a week. In rare cases, Zika virus infection has been linked to serious birth defects, including microcephaly and other brain anomalies, as well as Guillain-Barre syndrome. There is currently no vaccine or medicine to treat Zika virus infection, and prevention measures include avoiding mosquito bites, postponing travel to affected areas, and taking precautions for sexual transmission [medquad_ans_0006546-1][medquad_ans_0000981-1__dup1].

**Supported citations**: `medquad_ans_0006546-1`, `medquad_ans_0000981-1__dup1`
**Unsupported citations**: _none_

### Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.989, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000019-3__dup2` (score=0.5746) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack...
2. `medquad_ans_0000019-4__dup2` (score=0.5664) - Broken heart syndrome affects women more often than men. Often, people who experience broken heart syndrome have previously been healthy. Research shows that the traditional risk factors for heart disease may not appl...
3. `medquad_ans_0000019-1__dup2` (score=0.5569) - Broken heart syndrome is a condition in which extreme stress can lead to heart muscle failure. The failure is severe, but often short-term. Most people who experience broken heart syndrome think they may be having a h...
4. `medquad_ans_0000019-7` (score=0.5542) - Even though broken heart syndrome may feel like a heart attack, its a very different problem that needs a different type of treatment. The good news is that broken heart syndrome is usually treatable, and most people ...
5. `medquad_ans_0000443-1__dup1` (score=0.5251) - In the United States, 1 in 4 women dies from heart disease. The most common cause of heart disease in both men and women is narrowing or blockage of the coronary arteries, the blood vessels that supply blood to the he...

**Answer**:

Broken heart syndrome is believed to be caused by extreme emotional or physical stress, which triggers a surge of fight or flight hormones such as adrenaline and noradrenaline. This hormone surge overwhelms and "stuns" the heart muscle, leading to temporary changes in heart muscle cells or coronary blood vessels. Potential causes include a hormone surge, coronary artery spasm, or microvascular dysfunction [medquad_ans_0000019-3__dup2].

**Supported citations**: `medquad_ans_0000019-3__dup2`
**Unsupported citations**: _none_

### Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Scores**: context_relevance=0.800, groundedness=0.581, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000005-9__dup1` (score=0.5878) - - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. - Hashimotos disease often leads to reduced thyroid function, or hypothyroidism. Hypothyroidism is...
2. `medquad_ans_0000448-2` (score=0.5726) - Hashimoto thyroiditis affects 1 to 2 percent of people in the United States. It occurs more often in women than in men, which may be related to hormonal factors. The condition is the most common cause of thyroid under...
3. `medquad_ans_0002735-1` (score=0.5724) - Hashimoto's encephalitis (HE) is a condition characterized by onset of confusion with altered level of consciousness; seizures; and jerking of muscles (myoclonus). Psychosis, including visual hallucinations and parano...
4. `medquad_ans_0000448-3` (score=0.5693) - Hashimoto thyroiditis is thought to result from a combination of genetic and environmental factors. Some of these factors have been identified, but many remain unknown. Hashimoto thyroiditis is classified as an autoim...
5. `medquad_ans_0000005-1__dup5` (score=0.5663) - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. An autoimmune disease is a disorder in which the bodys immune system attacks the bodys own cells and...

**Answer**:

Hashimoto thyroiditis is an autoimmune disease characterized by the immune system attacking the thyroid gland, leading to chronic inflammation. It is the most common cause of hypothyroidism in the United States and affects 1 to 2 percent of people in the United States. The condition is more common in women than in men, and its development is thought to result from a combination of genetic and environmental factors [medquad_ans_0000005-9__dup1, medquad_ans_0000448-2, medquad_ans_0000448-3]. Hashimoto thyroiditis may progress slowly, with some individuals showing no symptoms initially, and may lead to an enlarged thyroid (goiter), which can cause a feeling of fullness in the throat [medquad_ans_0000005-9__dup1]. It is classified as an autoimmune disorder [medquad_ans

**Supported citations**: `medquad_ans_0000005-9__dup1`
**Unsupported citations**: `medquad_ans_0000005-9__dup1, medquad_ans_0000448-2, medquad_ans_0000448-3`

### Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0001043-2` (score=0.5953) - Wolman disease is estimated to occur in 1 in 350,000 newborns.
2. `medquad_ans_0000002-3__dup1` (score=0.5886) - Wolmans disease is usually fatal by age 1. The onset and course of cholesteryl ester storage disease varies, and individuals may live into adulthood.
3. `medquad_ans_0006463-2` (score=0.5886) - What are the signs and symptoms of Wolman disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Wolman disease. If the information is available, the table below includes how often...
4. `medquad_ans_0006463-1` (score=0.5850) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include...
5. `medquad_ans_0001043-5` (score=0.5787) - These resources address the diagnosis or management of Wolman disease: - Genetic Testing Registry: Lysosomal acid lipase deficiency These resources from MedlinePlus offer information about the diagnosis and management...

**Answer**:

Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands [medquad_ans_0006463-1]. It is estimated to occur in 1 in 350,000 newborns [medquad_ans_0001043-2]. The disease is usually fatal by age 1 [medquad_ans_0000002-3__dup1]. Common symptoms in infants include enlarged liver and spleen, poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor absorption of nutrients from food [medquad_ans_0006463-1]. Signs and symptoms such as abdominal pain, cognitive impairment, hepatic failure, hepatomegaly, hyperker

**Supported citations**: `medquad_ans_0006463-1`, `medquad_ans_0001043-2`, `medquad_ans_0000002-3__dup1`
**Unsupported citations**: _none_

### Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002753-1` (score=0.5157) - Hemangiopericytoma is a term used to described a group of tumors that are derived from pericytes, the cells normally arranged along specific types of blood vessels called capillaries and venules. These types of tumors...
2. `medquad_ans_0002753-3` (score=0.5096) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma a...
3. `medquad_ans_0002753-2` (score=0.4983) - What are the signs and symptoms of Hemangiopericytoma? The Human Phenotype Ontology provides the following list of signs and symptoms for Hemangiopericytoma. If the information is available, the table below includes h...
4. `medquad_ans_0002753-4` (score=0.4924) - What treatment is available for meningeal hemangiopericytoma? Radical surgical resection with removal of all meningeal attachments is typically the preferred treatment. However this treatment option is generally possi...
5. `medquad_ans_0002750-1` (score=0.4452) - The term hemangioendothelioma describes several types of vascular neosplasms and includes both non-cancerous (benign) and cancerous (malignant) growths. The term has also been applied to those that show "borderline" b...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

### Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Scores**: context_relevance=0.875, groundedness=0.459, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000655-3` (score=0.5037) - Microphthalmia may be caused by changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or...
2. `medquad_ans_0000655-4` (score=0.4679) - Isolated microphthalmia is sometimes inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition eac...
3. `medquad_ans_0000593-3` (score=0.4599) - Mutations in at least two genes on the X chromosome are thought to be responsible for Lenz microphthalmia syndrome. Only one of these genes, BCOR, has been identified. The BCOR gene provides instructions for making a ...
4. `medquad_ans_0000743-3` (score=0.4527) - Ocular albinism type 1 results from mutations in the GPR143 gene. This gene provides instructions for making a protein that plays a role in pigmentation of the eyes and skin. It helps control the growth of melanosomes...
5. `medquad_ans_0000924-3` (score=0.4465) - Mutations in the SOX2 gene cause SOX2 anophthalmia syndrome. This gene provides instructions for making a protein that plays a critical role in the formation of many different tissues and organs during embryonic devel...

**Answer**:

Genetic changes related to microphthalmia include mutations in many genes involved in the early development of the eye, most of which have not been identified. These mutations may occur in a chromosomal abnormality affecting one or more genes. In some cases, microphthalmia is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations, with parents each carrying one mutated copy but typically not showing symptoms. Specific gene mutations have been identified in certain syndromes, such as mutations in the BCOR gene on the X chromosome associated with Lenz microphthalmia syndrome, which disrupts the development of the eyes and other organs. Additionally, ocular albinism type 1, which involves mutations in the GPR143 gene, can also be related to microphthalmia due to its impact on eye pigmentation and development [medquad_ans_0000655-3, medquad_ans

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Scores**: context_relevance=1.000, groundedness=0.855, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0003550-2` (score=0.5653) - What are the signs and symptoms of Lesch Nyhan syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lesch Nyhan syndrome. If the information is available, the table below includ...
2. `medquad_ans_0005298-1` (score=0.5636) - What are the signs and symptoms of Renal hypouricemia? The Human Phenotype Ontology provides the following list of signs and symptoms for Renal hypouricemia. If the information is available, the table below includes h...
3. `medquad_ans_0002277-2` (score=0.5597) - What are the signs and symptoms of Familial juvenile hyperuricaemic nephropathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial juvenile hyperuricaemic nephropathy. If the ...
4. `medquad_ans_0000909-1__dup1` (score=0.5460) - What are the signs and symptoms of Camptodactyly taurinuria? The Human Phenotype Ontology provides the following list of signs and symptoms for Camptodactyly taurinuria. If the information is available, the table belo...
5. `medquad_ans_0003833-1` (score=0.5423) - What are the signs and symptoms of Maturity-onset diabetes of the young, type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Maturity-onset diabetes of the young, type 5. If the ...

**Answer**:

Gout is listed as a symptom of Lesch Nyhan syndrome [medquad_ans_0003550-2]. The frequency of gout in patients with Lesch Nyhan syndrome is not specified in the provided context.

**Supported citations**: `medquad_ans_0003550-2`
**Unsupported citations**: _none_

### Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Scores**: context_relevance=1.000, groundedness=0.716, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000309-2` (score=0.5167) - The prevalence of dystonia 6 is unknown. Studies indicate that it likely accounts for between 1 and 3 percent of all cases of dystonia. For reasons that are unclear, the disorder appears to be slightly more prevalent ...
2. `medquad_ans_0001986-1` (score=0.5068) - What are the signs and symptoms of Dystonia 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 1. If the information is available, the table below includes how often the sym...
3. `medquad_ans_0000307-1__dup1` (score=0.5057) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...
4. `medquad_ans_0000112-1__dup1` (score=0.5022) - The dystonias are movement disorders in which sustained muscle contractions cause twisting and repetitive movements or abnormal postures. The movements, which are involuntary and sometimes painful, may affect a single...
5. `medquad_ans_0000309-1` (score=0.4921) - Dystonia 6 is one of many forms of dystonia, which is a group of conditions characterized by involuntary movements, twisting (torsion) and tensing of various muscles, and unusual positioning of affected body parts. Dy...

**Answer**:

The provided context does not contain information about the causes of Dystonia 8. [medquad_ans_0000309-2], [medquad_ans_0001986-1], and [medquad_ans_0000307-1__dup1] discuss general dystonia and Dystonia 1, but there is no mention of Dystonia 8 or its causes. Therefore, the cause of Dystonia 8 cannot be determined from the given context.

**Supported citations**: `medquad_ans_0000309-2`, `medquad_ans_0001986-1`, `medquad_ans_0000307-1__dup1`
**Unsupported citations**: _none_

### Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Scores**: context_relevance=0.444, groundedness=0.920, answer_relevance=0.444

**Top docs**:

1. `medquad_ans_0000612-2` (score=0.5991) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unk...
2. `medquad_ans_0000650-2` (score=0.5938) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
3. `medquad_ans_0000989-2` (score=0.5909) - This condition affects an estimated 1 in 50,000 people.
4. `medquad_ans_0000359-2` (score=0.5882) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
5. `medquad_ans_0000934-2` (score=0.5656) - This condition affects fewer than 1 in 150,000 males and is very rare in females.

**Answer**:

Lysinuric protein intolerance affects an estimated 1 in 50,000 to 1 in 100,000 people worldwide [medquad_ans_0000650-2]. In Finland, it occurs in 1 in 60,000 newborns, and in Japan, in 1 in 57,000 newborns [medquad_ans_0000612-2]. It is also estimated to affect about 1 per million people worldwide [medquad_ans_0000359-2], and is more common in certain areas of Quebec, Canada [medquad_ans_0000359-2]. The condition affects fewer than 1 in 150,000 males and is very rare in females [medquad_ans_0000934-2].

**Supported citations**: `medquad_ans_0000650-2`, `medquad_ans_0000612-2`, `medquad_ans_0000359-2`, `medquad_ans_0000934-2`
**Unsupported citations**: _none_

### Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Scores**: context_relevance=0.875, groundedness=0.771, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0001051-2__dup1` (score=0.5089) - What are the signs and symptoms of Cerebral autosomal recessive arteriopathy with subcortical infarcts and leukoencephalopathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Cerebr...
2. `medquad_ans_0002200-2` (score=0.5070) - What are the signs and symptoms of Fabry disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Fabry disease. If the information is available, the table below includes how often t...
3. `medquad_ans_0004939-2` (score=0.5038) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their abilit...
4. `medquad_ans_0002812-2` (score=0.5027) - What are the signs and symptoms of Hereditary angiopathy with nephropathy, aneurysms, and muscle cramps syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Hereditary angiopath...
5. `medquad_ans_0004307-2` (score=0.4896) - What are the signs and symptoms of Nail-patella syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Nail-patella syndrome. If the information is available, the table below incl...

**Answer**:

The provided context does not contain information about the symptoms of Renal Artery Stenosis. [medquad_ans_0001051-2__dup1] and [medquad_ans_0002200-2] discuss symptoms of Cerebral autosomal recessive arteriopathy with subcortical infarcts and leukoencephalopathy and Fabry disease, respectively, but neither addresses Renal Artery Stenosis. Therefore, the requested information is not available in the context.

**Supported citations**: `medquad_ans_0001051-2__dup1`, `medquad_ans_0002200-2`
**Unsupported citations**: _none_

### Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Scores**: context_relevance=0.800, groundedness=0.759, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000712-5` (score=0.4947) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis -...
2. `medquad_ans_0000302-5` (score=0.4805) - These resources address the diagnosis or management of Dowling-Degos disease: - Cleveland Clinic: Skin Care Concerns - Genetic Testing Registry: Reticulate acropigmentation of Kitamura These resources from MedlinePlus...
3. `medquad_ans_0000718-5` (score=0.4767) - These resources address the diagnosis or management of NOMID: - Genetic Testing Registry: Chronic infantile neurological, cutaneous and articular syndrome These resources from MedlinePlus offer information about the d...
4. `medquad_ans_0000009-5` (score=0.4747) - These resources address the diagnosis or management of acral peeling skin syndrome: - Birmingham Children's Hospital, National Health Service (UK) - Genetic Testing Registry: Peeling skin syndrome, acral type These re...
5. `medquad_ans_0000059-15` (score=0.4662) - The goals of psoriasis treatment are to change the course of the disease by interfering with the increased production of skin cells, and to remove scales and smooth rough skin.

**Answer**:

The provided context does not specify treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis. While it mentions that the Genetic Testing Registry includes resources for these conditions and that MedlinePlus offers information on diagnosis and management of various health conditions—including diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care—no specific treatments are detailed for these syndromes. Therefore, based on the given information, the treatments cannot be determined. [medquad_ans_0000712-5]

**Supported citations**: `medquad_ans_0000712-5`
**Unsupported citations**: _none_

### Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.5428) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0004851-1` (score=0.4799) - What are the signs and symptoms of Pheochromocytoma-islet cell tumor syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pheochromocytoma-islet cell tumor syndrome. If the info...
3. `medquad_ans_0002593-2` (score=0.4794) - What are the signs and symptoms of Glucocorticoid-remediable aldosteronism? The Human Phenotype Ontology provides the following list of signs and symptoms for Glucocorticoid-remediable aldosteronism. If the informatio...
4. `medquad_ans_0000648-1__dup1` (score=0.4769) - What are the signs and symptoms of Bartter syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Bartter syndrome type 3. If the information is available, the table below ...
5. `medquad_ans_0004850-1` (score=0.4721) - What are the signs and symptoms of Pheochromocytoma, childhood? The Human Phenotype Ontology provides the following list of signs and symptoms for Pheochromocytoma, childhood. If the information is available, the tabl...

**Answer**:

The symptoms of Familial hyperaldosteronism type III include hypercalciuria (5%), metabolic acidosis (5%), polydipsia (5%), polyuria (5%), adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

**Supported citations**: `medquad_ans_0002259-1`
**Unsupported citations**: _none_

### Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5762) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000642-4` (score=0.5748) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000914-4` (score=0.5748) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000474-4` (score=0.5748) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000598-4` (score=0.5748) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Spinocerebellar ataxia 11 or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Spinocerebellar ataxia 11 is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Scores**: context_relevance=1.000, groundedness=0.870, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0003040-1` (score=0.6128) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload....
2. `medquad_ans_0002778-1` (score=0.5591) - What are the signs and symptoms of Hemolytic anemia lethal congenital nonspherocytic with genital and other abnormalities? The Human Phenotype Ontology provides the following list of signs and symptoms for Hemolytic a...
3. `medquad_ans_0000340-1__dup1` (score=0.5472) - What are the signs and symptoms of Anemia sideroblastic and spinocerebellar ataxia? The Human Phenotype Ontology provides the following list of signs and symptoms for Anemia sideroblastic and spinocerebellar ataxia. I...
4. `medquad_ans_0000258-2__dup1` (score=0.5447) - What are the signs and symptoms of Alpha-thalassemia? The Human Phenotype Ontology provides the following list of signs and symptoms for Alpha-thalassemia. If the information is available, the table below includes how...
5. `medquad_ans_0002766-2` (score=0.5344) - What are the signs and symptoms of Hemochromatosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hemochromatosis. If the information is available, the table below includes how oft...

**Answer**:

The signs and symptoms of Hypochromic microcytic anemia with iron overload include abnormality of metabolism/homeostasis, abnormality of the liver, anemia, and autosomal recessive inheritance. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which collects information from sources such as Orphanet, a European rare disease database. Frequencies are typically listed as rough estimates of the percentage of patients who have a particular feature, often presented as fractions (e.g., 25/25 meaning all patients in a study had the symptom). However, specific symptom frequencies for this condition are not detailed in the provided context [medquad_ans_0003040-1].

**Supported citations**: `medquad_ans_0003040-1`
**Unsupported citations**: _none_

### Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000441-4` (score=0.5773) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has a parent and other family ...
2. `medquad_ans_0000785-4` (score=0.5772) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000749-4` (score=0.5735) - This condition has an autosomal dominant inheritance pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all reported cases have resulted from new mutations in th...
4. `medquad_ans_0000403-4` (score=0.5727) - This condition has an autosomal dominant inheritance pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. All reported cases have resulted from new mutations in the gene ...
5. `medquad_ans_0000284-4` (score=0.5722) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention benign chronic pemphigus or its mode of inheritance. Therefore, it is not possible to determine based on the given information whether benign chronic pemphigus is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000926-4` (score=0.6142) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000286-4` (score=0.6142) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000473-4` (score=0.6142) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0001037-4` (score=0.6142) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000301-4` (score=0.6142) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Camurati-Engelmann disease or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Camurati-Engelmann disease is inherited. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Scores**: context_relevance=0.900, groundedness=0.914, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000580-3` (score=0.4890) - Somatic mutations in the BRAF gene have been identified in the Langerhans cells of about half of individuals with Langerhans cell histiocytosis. Somatic gene mutations are acquired during a person's lifetime and are p...
2. `medquad_ans_0000483-3` (score=0.4775) - Histiocytosis-lymphadenopathy plus syndrome is caused by mutations in the SLC29A3 gene, which provides instructions for making a protein called equilibrative nucleoside transporter 3 (ENT3). ENT3 belongs to a family o...
3. `medquad_ans_0000324-4` (score=0.4755) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
4. `medquad_ans_0000174-3` (score=0.4717) - Mutations in the CHD7 gene cause more than half of all cases of CHARGE syndrome. The CHD7 gene provides instructions for making a protein that most likely regulates gene activity (expression) by a process known as chr...
5. `medquad_ans_0001697-3` (score=0.4512) - What causes cutaneous mastocytosis? Most cases of cutaneous mastocytosis are caused by changes (mutations) in the KIT gene. This gene encodes a protein that helps control many important cellular processes such as cell...

**Answer**:

The genetic changes related to histiocytosis-lymphadenopathy plus syndrome are mutations in the SLC29A3 gene, which provide instructions for making a protein called equilibrative nucleoside transporter 3 (ENT3). These mutations reduce or eliminate the activity of the ENT3 protein, leading to impaired nucleoside transport. This results in a buildup of nucleosides in lysosomes, which may damage cell function, and a reduction in nucleosides in mitochondria, potentially impairing cellular energy production [medquad_ans_0000483-3].

**Supported citations**: `medquad_ans_0000483-3`
**Unsupported citations**: _none_

### Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5317) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000647-4` (score=0.5285) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000266-4` (score=0.5285) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000231-4` (score=0.5285) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000528-4` (score=0.5285) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Noonan syndrome or its mode of inheritance. Therefore, it is not possible to determine from the given information whether Noonan syndrome is inherited. [no_source]

**Supported citations**: _none_
**Unsupported citations**: `no_source`

### Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Scores**: context_relevance=0.500, groundedness=0.677, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001089-5` (score=0.6144) - These resources address the diagnosis or management of HMG-CoA lyase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of hydroxymethylglutaryl-CoA lyase These resources from MedlinePlus offer inf...
2. `medquad_ans_0001093-5` (score=0.6099) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Aci...
3. `medquad_ans_0001082-5` (score=0.5977) - These resources address the diagnosis or management of 2-methylbutyryl-CoA dehydrogenase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of 2-methylbutyryl-CoA dehydrogenase These resources from...
4. `medquad_ans_0000120-5` (score=0.5975) - These resources address the diagnosis or management of beta-ketothiolase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of acetyl-CoA acetyltransferase These resources from MedlinePlus offer in...
5. `medquad_ans_0000621-5` (score=0.5955) - These resources address the diagnosis or management of malonyl-CoA decarboxylase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of malonyl-CoA decarboxylase These resources from MedlinePlus off...

**Answer**:

The provided context does not specify the treatments for 3-methylglutaconyl-CoA hydratase deficiency. It lists diagnostic resources such as "Baby's First Test" and references to the Genetic Testing Registry and MedlinePlus Encyclopedia, but does not detail any drug therapy or specific treatment approaches [medquad_ans_0001093-5]. Therefore, based on the given information, the treatments for 3-methylglutaconyl-CoA hydratase deficiency cannot be determined.

**Supported citations**: `medquad_ans_0001093-5`
**Unsupported citations**: _none_

### Example 61 (`0003000-1`)

**Question**: What is (are) Hyperkalemic periodic paralysis ?

**Scores**: context_relevance=1.000, groundedness=0.936, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.5051) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0003000-2` (score=0.4600) - What are the signs and symptoms of Hyperkalemic periodic paralysis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hyperkalemic periodic paralysis. If the information is available, ...
3. `medquad_ans_0003059-4` (score=0.4551) - How is hypokalemic periodic paralysis inherited? This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, ...
4. `medquad_ans_0000126-1__dup1` (score=0.4503) - Familial periodic paralyses are a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. They are characterized by episodes in which the affect...
5. `medquad_ans_0003059-5` (score=0.4472) - How is hypokalemic periodic paralysis diagnosed? The diagnosis of hypokalemic periodic paralysis is based on a history of episodes of paralysis and low levels of potassium in the blood during attacks (less than 0.9 to...

**Answer**:

Hyperkalemic periodic paralysis is a rare neurological disorder characterized by episodic flaccid weakness and periodic hyperkalemic paralysis. It affects an estimated 1 in 200,000 people. The condition is associated with hyperkalemia (elevated potassium levels in the blood) and is part of a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. Signs and symptoms include muscle weakness, myalgia, myotonia, gait disturbance, and involuntary movements, with hyperkalemia and EMG abnormality occurring in approximately 50% of patients. The Human Phenotype Ontology provides a list of signs and symptoms, with frequencies based on studies of patients [medquad_ans_0000494-2][medquad_ans_0003000-2].

**Supported citations**: `medquad_ans_0000494-2`, `medquad_ans_0003000-2`
**Unsupported citations**: _none_

### Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Scores**: context_relevance=0.600, groundedness=0.691, answer_relevance=0.760

**Top docs**:

1. `medquad_ans_0000225-5` (score=0.5171) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of...
2. `medquad_ans_0000228-5` (score=0.4307) - These resources address the diagnosis or management of congenital hepatic fibrosis: - Gene Review: Gene Review: Congenital Hepatic Fibrosis Overview - Genetic Testing Registry: Congenital hepatic fibrosis These resour...
3. `medquad_ans_0000130-5` (score=0.4287) - These resources address the diagnosis or management of BPES: - Gene Review: Gene Review: Blepharophimosis, Ptosis, and Epicanthus Inversus - Genetic Testing Registry: Blepharophimosis, ptosis, and epicanthus inversus ...
4. `medquad_ans_0000745-5` (score=0.4272) - These resources address the diagnosis or management of oculodentodigital dysplasia: - Genetic Testing Registry: Oculodentodigital dysplasia - MedlinePlus Encyclopedia: Webbing of the fingers or toes - UC Davis Childre...
5. `medquad_ans_0000873-5` (score=0.4214) - These resources address the diagnosis or management of retroperitoneal fibrosis: - Johns Hopkins Medicine These resources from MedlinePlus offer information about the diagnosis and management of various health conditi...

**Answer**:

The provided context does not specify the treatments for congenital fibrosis of the extraocular muscles. While it mentions resources related to diagnosis and management, such as Gene Review, Genetic Testing Registry entries, and MedlinePlus Encyclopedia entries on extraocular muscle function testing and strabismus, no explicit information on treatments is available [medquad_ans_0000225-5]. Therefore, based on the given context, the treatments cannot be determined.

**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

### Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Scores**: context_relevance=1.000, groundedness=0.954, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000491-1__dup2` (score=0.5335) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms...
2. `medquad_ans_0000152-3__dup1` (score=0.5246) - Huntingtons disease causes disability that gets worse over time. People with this disease usually die within 15 to 20 years following diagnosis. At this time, no treatment is available to slow, stop or reverse the cou...
3. `medquad_ans_0002944-6` (score=0.5180) - How might Huntington disease be treated? Unfortunately, there is currently no cure for Huntington disease (HD). The current goal of treatment is to slow down the course of the disease and help affected people function...
4. `medquad_ans_0000489-1` (score=0.5158) - Huntington disease is a progressive brain disorder that causes uncontrolled movements, emotional problems, and loss of thinking ability (cognition). Adult-onset Huntington disease, the most common form of this disorde...
5. `medquad_ans_0002944-2` (score=0.5088) - What are the signs and symptoms of Huntington disease? Huntington disease (HD) is a progressive disorder that causes motor, cognitive, and psychiatric signs and symptoms. On average, most people begin developing featu...

**Answer**:

Huntington's disease is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms may include uncontrolled movements, clumsiness, and balance problems. As the disease progresses, it can lead to loss of the ability to walk, talk, and swallow, and some individuals may stop recognizing family members. The disease is progressive, causing worsening disability over time, with affected individuals typically living 15 to 20 years after diagnosis. There is currently no cure, and while medications can help manage symptoms such as abnormal movements and depression, they cannot slow down or stop the disease [medquad_ans_0000491-1__dup2]. Adult-onset Huntington's disease, the most common form, usually appears in a person's thirties or forties and is associated with emotional problems, loss of thinking ability

**Supported citations**: `medquad_ans_0000491-1__dup2`
**Unsupported citations**: _none_

### Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=0.778, groundedness=0.950, answer_relevance=0.511

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.6526) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
2. `medquad_ans_0000359-2` (score=0.6138) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
3. `medquad_ans_0000402-2` (score=0.6015) - The prevalence of GACI has been estimated to be about 1 in 391,000. At least 200 affected individuals have been described in the medical literature.
4. `medquad_ans_0000419-2` (score=0.5991) - Glutathione synthetase deficiency is very rare. This disorder has been described in more than 70 people worldwide.
5. `medquad_ans_0000650-2` (score=0.5951) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Approximately 20 affected individuals have been identified with glutamate formiminotransferase deficiency [medquad_ans_0000416-2].

**Supported citations**: `medquad_ans_0000416-2`
**Unsupported citations**: _none_

### Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Scores**: context_relevance=0.375, groundedness=0.931, answer_relevance=0.375

**Top docs**:

1. `medquad_ans_0000738-2` (score=0.6219) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
2. `medquad_ans_0000359-2` (score=0.6143) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
3. `medquad_ans_0000989-2` (score=0.6040) - This condition affects an estimated 1 in 50,000 people.
4. `medquad_ans_0000146-2` (score=0.6033) - While this condition occurs in people of all ethnic backgrounds, it is most common in people of Ashkenazi (eastern and central European) Jewish heritage. Studies suggest that this disorder affects 1 in 6,400 to 13,500...
5. `medquad_ans_0000934-2` (score=0.6013) - This condition affects fewer than 1 in 150,000 males and is very rare in females.

**Answer**:

Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people [medquad_ans_0000738-2].

**Supported citations**: `medquad_ans_0000738-2`
**Unsupported citations**: _none_

### Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Scores**: context_relevance=0.571, groundedness=0.989, answer_relevance=0.571

**Top docs**:

1. `medquad_ans_0000339-5__dup1` (score=0.5960) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on t...
2. `medquad_ans_0000339-1__dup3` (score=0.5873) - Also called crab lice or "crabs," pubic lice are parasitic insects found primarily in the pubic or genital area of humans. Pubic lice infestation is found worldwide and occurs in all races, ethnic groups, and levels o...
3. `medquad_ans_0000557-1__dup2` (score=0.5733) - Summary : Lice are parasitic insects that can be found on people's heads and bodies. They survive by feeding on human blood. Lice found on each area of the body are different from each other. The three types of lice t...
4. `medquad_ans_0000339-2__dup1` (score=0.5581) - Pubic ("crab") lice infestation is found worldwide and occurs in all races and ethnic groups and in all levels of society. Pubic lice usually are spread through sexual contact and are most common in adults. Occasional...
5. `medquad_ans_0000214-5__dup1` (score=0.5537) - Misdiagnosis of head lice infestation is common. The diagnosis of head lice infestation is best made by finding a live nymph or adult louse on the scalp or hair of a person. Because adult and nymph lice are very small...

**Answer**:

Pubic lice infestation is diagnosed by finding a “crab” louse or eggs (nits) on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp) [medquad_ans_0000339-5__dup1]. Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

### Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.869, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0005571-3` (score=0.6569) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
2. `medquad_ans_0000803-5` (score=0.5984) - These resources address the diagnosis or management of Poland syndrome: - Children's Medical Center of Dallas - Great Ormond Street Hospital (UK): Treatment Options for Symbrachydactyly - St. Louis Children's Hospital...
3. `medquad_ans_0000901-5` (score=0.5820) - These resources address the diagnosis or management of Sheldon-Hall syndrome: - Gillette Children's Hospital - NYU Langone Medical Center These resources from MedlinePlus offer information about the diagnosis and mana...
4. `medquad_ans_0000789-5` (score=0.5688) - These resources address the diagnosis or management of Peters plus syndrome: - Gene Review: Gene Review: Peters Plus Syndrome - Genetic Testing Registry: Peters plus syndrome These resources from MedlinePlus offer inf...
5. `medquad_ans_0000540-5` (score=0.5614) - These resources address the diagnosis or management of Jackson-Weiss syndrome: - Gene Review: Gene Review: FGFR-Related Craniosynostosis Syndromes - Genetic Testing Registry: Jackson-Weiss syndrome - MedlinePlus Encyc...

**Answer**:

Treatment options attempted for Shapiro syndrome include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, or sympathectomy. However, responses to these treatments have varied, and due to the limited number of diagnosed cases and the periodic nature of the disease, evaluating effective treatment options remains challenging. It is recommended that treatment decisions be made in consultation with a healthcare provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

### Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.930, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0002359-2` (score=0.5774) - What are the signs and symptoms of FG syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for FG syndrome. If the information is available, the table below includes how often the s...
2. `medquad_ans_0000373-1` (score=0.5749) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects ...
3. `medquad_ans_0002362-1` (score=0.5585) - What are the signs and symptoms of FG syndrome 4? The Human Phenotype Ontology provides the following list of signs and symptoms for FG syndrome 4. If the information is available, the table below includes how often t...
4. `medquad_ans_0002359-1` (score=0.5306) - FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. People wi...
5. `medquad_ans_0002359-3` (score=0.5300) - How might FG syndrome be treated? Treatment is aimed at addressing the individual symptoms present in each case. This often involves care by a team of providers which may include pediatricians, neurologists, cardiolog...

**Answer**:

FG syndrome is a genetic disorder characterized by a range of signs and symptoms, including aplasia/hypoplasia of the corpus callosum, behavioral abnormalities, broad forehead, cognitive impairment, high forehead, low-set posteriorly rotated ears, muscular hypotonia, abnormality of the palate, brachydactyly, clinodactyly of the 5th finger, cryptorchidism, EEG abnormality, epicanthus, fine hair, mask-like facies, open mouth, seizures, strabismus, abnormality of the intestine, hernia of the abdominal wall, hypertonia, ptosis, single transverse palmar crease, and sensorineural hearing impairment [medquad_ans_0002359-2]. The frequency of these symptoms varies, with many occurring in approximately 90% of patients, while others are seen in lower percentages or are reported in smaller numbers. The

**Supported citations**: `medquad_ans_0002359-2`
**Unsupported citations**: _none_

### Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006242-1` (score=0.6278) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0006572-2` (score=0.5539) - What are the signs and symptoms of 22q13.3 deletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 22q13.3 deletion syndrome. If the information is available, the table be...
3. `medquad_ans_0003554-1` (score=0.5525) - What are the signs and symptoms of Lethal congenital contracture syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Lethal congenital contracture syndrome 1. If the informat...
4. `medquad_ans_0003857-1` (score=0.5519) - What are the signs and symptoms of Meckel syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Meckel syndrome type 2. If the information is available, the table below in...
5. `medquad_ans_0006175-1` (score=0.5511) - What are the signs and symptoms of Tricho-dento-osseous syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Tricho-dento-osseous syndrome 1. If the information is available, ...

**Answer**:

The signs and symptoms of Tukel syndrome include carpal bone aplasia, carpal synostosis, compensatory chin elevation, congenital fibrosis of extraocular muscles, nonprogressive restrictive external ophthalmoplegia, postaxial oligodactyly, ptosis, restrictive external ophthalmoplegia, and syndactyly [medquad_ans_0006242-1].

**Supported citations**: `medquad_ans_0006242-1`
**Unsupported citations**: _none_

### Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Scores**: context_relevance=0.333, groundedness=0.025, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000676-5` (score=0.5626) - These resources address the diagnosis or management of mucolipidosis III gamma: - Gene Review: Gene Review: Mucolipidosis III Gamma - Genetic Testing Registry: Mucolipidosis III Gamma - MedlinePlus Encyclopedia: Cloud...
2. `medquad_ans_0000677-5` (score=0.5553) - These resources address the diagnosis or management of mucolipidosis type IV: - Gene Review: Gene Review: Mucolipidosis IV - Genetic Testing Registry: Ganglioside sialidase deficiency - MedlinePlus Encyclopedia: Gastr...
3. `medquad_ans_0000675-5` (score=0.5450) - These resources address the diagnosis or management of mucolipidosis III alpha/beta: - Gene Review: Gene Review: Mucolipidosis III Alpha/Beta - Genetic Testing Registry: Pseudo-Hurler polydystrophy - MedlinePlus Encyc...
4. `medquad_ans_0000674-5` (score=0.5409) - These resources address the diagnosis or management of mucolipidosis II alpha/beta: - Gene Review: Gene Review: Mucolipidosis II - Genetic Testing Registry: I cell disease - MedlinePlus Encyclopedia: Clubfoot - Medlin...
5. `medquad_ans_0000682-5` (score=0.5201) - These resources address the diagnosis or management of mucopolysaccharidosis type VI: - Emory University Lysosomal Storage Disease Center - Genetic Testing Registry: Mucopolysaccharidosis type VI - MedlinePlus Encyclo...

**Answer**:

The provided context does not specify the treatments for mucolipidoses. While it lists resources that address diagnosis and management, including diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care, it does not explicitly detail treatments for mucolipidoses [medquad_ans_0000676-5, medquad_ans_0000677-5, medquad_ans_0000675-5, medquad_ans_0000674-5]. Therefore, based on the given information, the specific treatments for mucolipidoses cannot be determined.

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000676-5, medquad_ans_0000677-5, medquad_ans_0000675-5, medquad_ans_0000674-5`

### Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.970, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0005084-2` (score=0.5530) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis...
2. `medquad_ans_0000689-1__dup1` (score=0.5084) - What are the signs and symptoms of Benign recurrent intrahepatic cholestasis? The Human Phenotype Ontology provides the following list of signs and symptoms for Benign recurrent intrahepatic cholestasis. If the inform...
3. `medquad_ans_0003240-2` (score=0.5063) - What are the signs and symptoms of Intrahepatic cholestasis of pregnancy? The Human Phenotype Ontology provides the following list of signs and symptoms for Intrahepatic cholestasis of pregnancy. If the information is...
4. `medquad_ans_0001157-1` (score=0.5027) - What are the signs and symptoms of Cholestasis, progressive familial intrahepatic 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Cholestasis, progressive familial intrahepatic 4....
5. `medquad_ans_0001548-2` (score=0.4758) - What are the signs and symptoms of Congenital porphyria? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital porphyria. If the information is available, the table below includ...

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 include cirrhosis, conjugated hyperbilirubinemia, death in childhood, diarrhea, elevated alkaline phosphatase, failure to thrive, fat malabsorption, hepatocellular carcinoma, hepatomegaly, infantile onset, intermittent jaundice, intrahepatic cholestasis, pruritus, short stature, and splenomegaly [medquad_ans_0005084-2].

**Supported citations**: `medquad_ans_0005084-2`
**Unsupported citations**: _none_

### Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Scores**: context_relevance=0.600, groundedness=0.826, answer_relevance=0.440

**Top docs**:

1. `medquad_ans_0000211-6` (score=0.5818) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he i...
2. `medquad_ans_0000359-2` (score=0.5533) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
3. `medquad_ans_0000989-2` (score=0.5518) - This condition affects an estimated 1 in 50,000 people.
4. `medquad_ans_0000219-4__dup3` (score=0.5486) - Erectile dysfunction affects men of all races and in all parts of the world. Men are more likely to have ED as they get older. For example, ED occurs in - about 12 percent of men younger than 60 - 22 percent of men ag...
5. `medquad_ans_0000650-2` (score=0.5480) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Urinary incontinence occurs in 11 to 34 percent of older men [medquad_ans_0000211-6]. Two to 11 percent of older men report daily urinary incontinence [medquad_ans_0000211-6]. Therefore, the prevalence of urinary incontinence in men ranges from 11% to 34% among older men.

**Supported citations**: `medquad_ans_0000211-6`
**Unsupported citations**: _none_

### Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.520, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0005415-2` (score=0.4992) - What are the signs and symptoms of Robinow syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Robinow syndrome. If the information is available, the table below includes how o...
2. `medquad_ans_0005414-1` (score=0.4876) - What are the signs and symptoms of Robinow Sorauf syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Robinow Sorauf syndrome. If the information is available, the table below ...
3. `medquad_ans_0000882-1` (score=0.4753) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
4. `medquad_ans_0005382-1` (score=0.4560) - What are the signs and symptoms of Rienhoff syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Rienhoff syndrome. If the information is available, the table below includes how...
5. `medquad_ans_0005415-1` (score=0.4379) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...

**Answer**:

Robinow syndrome is a rare genetic disorder characterized by a variety of physical signs and symptoms, including short stature, anteverted nares, short nose, abnormality of dental morphology, brachydactyly, clinodactyly of the 5th finger, dental malocclusion, downturned corners of mouth, hypertelorism, hypoplasia of penis, malar flattening, micromelia, short distal phalanx of finger, vertebral segmentation defect, wide mouth, small hand, depressed nasal bridge, bifid tongue, long eyelashes, high palate, abnormality of female external genitalia, abnormality of the eyelashes, abnormality of the fingernails, abnormality of the ribs, abnormality of thumb phalanx, cryptorchidism, elbow dislocation, epicanthus, frontal bossing, gingival overgrowth, hearing impairment, kyphosis, long palpebral fiss

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 74 (`0000551-1`)

**Question**: What is (are) juvenile primary lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.957, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000551-2` (score=0.4994) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.
2. `medquad_ans_0003331-1` (score=0.4991) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
3. `medquad_ans_0000551-1` (score=0.4917) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. The features of this disorder are caused by damage to moto...
4. `medquad_ans_0003331-4` (score=0.4654) - How is juvenile primary lateral sclerosis inherited? Juvenile primary lateral sclerosis is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of a...
5. `medquad_ans_0005053-2` (score=0.4622) - What are the signs and symptoms of Primary lateral sclerosis? Primary lateral sclerosis (PLS) causes weakness in the voluntary muscles, such as those used to control the legs, arms and tongue. PLS can happen at any ag...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. It is caused by damage to motor neurons, which are specialized nerve cells in the brain and spinal cord that control muscle movement. Symptoms begin in early childhood and progress slowly over many years. Early symptoms include clumsiness, muscle weakness and spasticity in the legs, and difficulty with balance. As symptoms progress, spasticity spreads to the arms and hands, and individuals may develop slurred speech, drooling, difficulty swallowing, and an inability to walk. Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene and is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations, and each parent carries one copy of the mutated gene without showing signs or symptoms [medquad_ans_0003331-1]med

**Supported citations**: `medquad_ans_0003331-1`
**Unsupported citations**: _none_

### Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Scores**: context_relevance=0.750, groundedness=0.886, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000040-5__dup1` (score=0.5554) - Unlike other types of cancer, leukemia isn't a tumor that your doctor can surgically remove. Leukemia cells are produced in the bone marrow and travel throughout the body. The Goal of Treatment The goal of treatment f...
2. `medquad_ans_0000040-18` (score=0.5549) - Treatment depends on a number of factors, including the type of leukemia, the patient's age and general health, where leukemia cells have collected in the body, and whether the leukemia has been treated before. Certai...
3. `medquad_ans_0000554-1__dup2` (score=0.5402) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells c...
4. `medquad_ans_0000040-11` (score=0.5353) - There are four common types of leukemia. They are chronic lymphocytic leukemia, chronic myeloid leukemia, acute myeloid leukemia, and acute lymphocytic leukemia. Chronic lymphocytic leukemia, chronic myeloid leukemia,...
5. `medquad_ans_0000040-10` (score=0.5351) - Acute leukemia gets worse quickly. In chronic leukemia, symptoms develop gradually and are generally not as severe as in acute leukemia.

**Answer**:

Leukemia is a type of cancer characterized by the uncontrolled production of abnormal white blood cells in the bone marrow, which then travel throughout the body. Unlike other cancers, leukemia is not a tumor that can be surgically removed. The goal of treatment is to destroy leukemia cells and allow normal blood cells to be produced in the bone marrow. Treatment options include chemotherapy, biological therapy, radiation therapy, or stem cell transplantation, depending on the type and extent of the disease, the patient's age and health, and other factors such as the presence of leukemia cells in the cerebrospinal fluid [medquad_ans_0000040-5__dup1]. Leukemia can be classified as acute or chronic, with acute leukemia requiring immediate treatment to stop the rapid growth of cells and achieve remission, while chronic leukemia may not require treatment until symptoms appear [medquad_ans_0000040-5\_\_dup

**Supported citations**: `medquad_ans_0000040-5__dup1`
**Unsupported citations**: _none_

### Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Scores**: context_relevance=1.000, groundedness=0.993, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000324-1__dup1` (score=0.4729) - Anaplastic astrocytoma is a rare, cancerous (malignant) type of brain tumor that arises from star-shaped brain cells called astrocytes. These cells surround and protect nerve cells in the brain and spinal cord. An ana...
2. `medquad_ans_0002574-1` (score=0.4587) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...
3. `medquad_ans_0002485-1` (score=0.4502) - Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. Gangliocytomas may occur in all age groups but most often occur in people between the ages of 10 and 30. The most common si...
4. `medquad_ans_0001809-4` (score=0.4499) - What treatment is available for desmoplastic infantile gangliomas? Surgical resection (removal of the area of the brain with the tumor) has been the standard treatment reported in the medical literature. The size of t...
5. `medquad_ans_0002485-2` (score=0.4480) - What are the signs and symptoms of gangliocytomas? Signs and symptoms caused by the presence of a gangliocytoma can vary depending on the tumor's location. Seizures are the most common symptom. Other symptoms may incl...

**Answer**:

Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. It may occur in all age groups but most often occurs in people between the ages of 10 and 30. The most common site is the temporal lobe of the brain, but it can arise anywhere in the CNS, including the cerebellum, brainstem, floor of the third ventricle, and spinal cord. It is among the most frequent tumors associated with epilepsy. Signs and symptoms depend on the tumor's location and may include seizures (most commonly), increased brain pressure, endocrine disorders, and focal symptoms. Gangliocytomas are generally slow-growing and usually do not become malignant. Treatment involves surgical removal of the tumor [medquad_ans_0002485-1].

**Supported citations**: `medquad_ans_0002485-1`
**Unsupported citations**: _none_

### Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.944, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000130-3__dup1` (score=0.5337) - The prognosis for most individuals with Miller Fisher syndrome is good. In most cases, recovery begins within 2 to 4 weeks of the onset of symptoms, and may be almost complete within 6 months. Some individuals are lef...
2. `medquad_ans_0000130-4__dup1` (score=0.5210) - The NINDS supports research aimed at discovering new ways to diagnose, treat, and, ultimately, cure neuropathies such as Miller Fisher syndrome.
3. `medquad_ans_0000130-1__dup1` (score=0.5172) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of ...
4. `medquad_ans_0000658-2` (score=0.4889) - Miller syndrome is a rare disorder; it is estimated to affect fewer than 1 in 1 million newborns. At least 30 cases have been reported in the medical literature.
5. `medquad_ans_0000130-2__dup1` (score=0.4851) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportiv...

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of tendon reflexes. Like Guillain-Barr syndrome, symptoms may be preceded by a viral illness. Additional symptoms include generalized muscle weakness and respiratory failure. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder [medquad_ans_0000130-1__dup1]. The prognosis for most individuals is good, with recovery beginning within 2 to 4 weeks of symptom onset and potentially being almost complete within 6 months, although some individuals may have residual deficits and rare relapses may occur [medquad_ans_0000130-3__dup1]. Treatment is identical to that for Guillain-Barr syndrome, involving intravenous immunoglobulin (IVIg) or

**Supported citations**: `medquad_ans_0000130-1__dup1`, `medquad_ans_0000130-3__dup1`
**Unsupported citations**: _none_

### Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.768, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006570-2` (score=0.5274) - What are the signs and symptoms of 22q11.2 deletion syndrome? Signs and symptoms of 22q11.2 deletion syndrome vary greatly from person to person, even among affected people in the same family. Symptoms may include: He...
2. `medquad_ans_0001084-3` (score=0.5194) - Most people with 22q11.2 deletion syndrome are missing a sequence of about 3 million DNA building blocks (base pairs) on one copy of chromosome 22 in each cell. This region contains 30 to 40 genes, many of which have ...
3. `medquad_ans_0006570-3` (score=0.5092) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location desig...
4. `medquad_ans_0006570-4` (score=0.5002) - Is 22q11.2 deletion syndrome inherited? Most cases of 22q11.2 deletion syndrome are not inherited from a parent and are caused by a random error during the formation of egg or sperm cells, or during early fetal develo...
5. `medquad_ans_0006570-1` (score=0.4945) - 22q11.2 deletion syndrome is a spectrum disorder that includes conditions formerly called DiGeorge syndrome; velocardiofacial syndrome; conotruncal anomaly face syndrome; cases of Opitz G/BBB syndrome; and Cayler card...

**Answer**:

The provided context does not specify the cause of 22q11.2 deletion syndrome. Therefore, based on the given information, the cause cannot be determined. [medquad_ans_0006570-2]

**Supported citations**: `medquad_ans_0006570-2`
**Unsupported citations**: _none_

### Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Scores**: context_relevance=1.000, groundedness=0.934, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000812-1__dup2` (score=0.5960) - Shingles is a disease caused by the varicella-zoster virus - the same virus that causes chickenpox. After you have chickenpox, the virus stays in your body. It may not cause problems for many years. As you get older, ...
2. `medquad_ans_0000148-1__dup2` (score=0.5842) - Shingles (herpes zoster) is an outbreak of rash or blisters on the skin that is caused by the same virus that causes chickenpox the varicella-zoster virus. The first sign of shingles is often burning or tingling pain ...
3. `medquad_ans_0000062-8` (score=0.5821) - Shingles is very common. Fifty percent of all Americans will have had shingles by the time they are 80. While shingles occurs in people of all ages, it is most common in 60-to 80-year-olds. In fact, one out of every t...
4. `medquad_ans_0000062-12` (score=0.5728) - Shingles is caused by a virus called the varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores and other symptoms healed, but the virus remained....
5. `medquad_ans_0000062-2__dup3` (score=0.5530) - Caused By A Virus Shingles is caused by a virus called varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores healed and the other symptoms went a...

**Answer**:

Shingles is caused by the varicella-zoster virus, the same virus that causes chickenpox. After a person has chickenpox, the virus remains in the body and can reactivate later in life, leading to shingles [medquad_ans_0000812-1__dup2].

**Supported citations**: `medquad_ans_0000812-1__dup2`
**Unsupported citations**: _none_

### Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.784, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0002456-1` (score=0.5822) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the ta...
2. `medquad_ans_0002200-2` (score=0.5468) - What are the signs and symptoms of Fabry disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Fabry disease. If the information is available, the table below includes how often t...
3. `medquad_ans_0002945-1` (score=0.5205) - What are the signs and symptoms of Hurler syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Hurler syndrome. If the information is available, the table below includes how oft...
4. `medquad_ans_0003528-1` (score=0.5159) - What are the signs and symptoms of Legius syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Legius syndrome. If the information is available, the table below includes how oft...
5. `medquad_ans_0002458-2` (score=0.5050) - What are the signs and symptoms of Fryns syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns syndrome. If the information is available, the table below includes how often...

**Answer**:

The symptoms of Fryns Hofkens Fabry syndrome include ulnar deviation of the finger (90%), distal ulnar hypoplasia, dysplastic radii, hypoplasia of the radius, mesomelic arm shortening, radial bowing, and ulnar deviation of the hand [medquad_ans_0002456-1].

Note: There appears to be a confusion in the name "Fryns Hofkens Fabry syndrome" as Fabry disease is a distinct condition, and "Fryns Hofkens" is not a recognized syndrome in the provided context. The listed symptoms are more consistent with a skeletal dysplasia or a condition involving ulnar deviation and radiological abnormalities, not Fabry disease. The provided context does not support the existence of "Fryns Hofkens Fabry syndrome" as a valid medical entity. Therefore, the symptoms listed may be mis

**Supported citations**: `medquad_ans_0002456-1`
**Unsupported citations**: _none_

### Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Scores**: context_relevance=1.000, groundedness=0.867, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000050-8` (score=0.6125) - Scientists are pursuing a wide range of basic and clinical studies on osteoporosis. Significant advances in preventing and treating osteoporosis continue to be made. Such advances are the direct result of research foc...
2. `medquad_ans_0000207-1__dup2` (score=0.5742) - Summary : Clinical trials are research studies that test how well new medical approaches work in people. Each study answers scientific questions and tries to find better ways to prevent, screen for, diagnose, or treat...
3. `medquad_ans_0000004-31` (score=0.5455) - Clinical research is medical research involving people. It includes clinical studies, which observe and gather information about large groups of people. It also includes clinical trials, which test a medicine, therapy...
4. `medquad_ans_0000019_3-7` (score=0.5381) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, ta...
5. `medquad_ans_0000027_4-4` (score=0.5313) - Cancer prevention clinical trials are used to study ways to prevent cancer. Cancer prevention clinical trials are used to study ways to lower the risk of developing certain types of cancer. Some cancer prevention tria...

**Answer**:

Scientists are pursuing a wide range of basic and clinical studies on osteoporosis, focusing on determining the causes and consequences of bone loss at the cellular and tissue levels, assessing risk factors, developing new strategies to maintain and even enhance bone density and reduce fracture risk, and exploring the roles of genetics, hormones, calcium, vitamin D, drugs, and exercise on bone mass [medquad_ans_0000050-8]. Clinical trials are also involved in testing new medical approaches to prevent, screen for, diagnose, or treat osteoporosis, including evaluating the safety and effectiveness of new treatments compared to existing ones [medquad_ans_0000207-1__dup2]. These trials follow a protocol that outlines the study design, procedures, and participant eligibility, and are reviewed and approved by an Institutional Review Board (IRB) to ensure ethical standards and participant welfare [medquad_ans_000020

**Supported citations**: `medquad_ans_0000050-8`, `medquad_ans_0000207-1__dup2`
**Unsupported citations**: _none_

### Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Scores**: context_relevance=0.857, groundedness=0.902, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000430-1__dup1` (score=0.5564) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. ...
2. `medquad_ans_0000869-1__dup1` (score=0.5525) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can brin...
3. `medquad_ans_0000001-7__dup1` (score=0.5495) - Topics
4. `medquad_ans_0000338-1__dup1` (score=0.5470) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the s...
5. `medquad_ans_0000694-1__dup1` (score=0.5421) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of me...

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. These scams often promise quick fixes, miracle cures, secret ingredients, or scientific breakthroughs, especially for common health issues like weight loss, memory loss, sexual performance, and joint pain. They target individuals with serious conditions such as cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. Red flags include claims that sound too good to be true. To protect yourself, it is advised to consult a doctor or healthcare professional before taking unproven treatments, especially when on prescription drugs. The Food and Drug Administration (FDA) is mentioned as a relevant authority [medquad_ans_0000430-1__dup1].

**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

### Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Scores**: context_relevance=0.833, groundedness=0.234, answer_relevance=0.433

**Top docs**:

1. `medquad_ans_0000055-3__dup2` (score=0.5568) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision -...
2. `medquad_ans_0000043-2__dup4` (score=0.5248) - Many people have no visible signs or symptoms of diabetes. Symptoms can also be so mild that you might not notice them. More than 5 million people in the United States have type 2 diabetes and do not know it. - increa...
3. `medquad_ans_0000009-3__dup3` (score=0.5237) - There are several things you can do to lower your risk for cataract. They include - having regular eye exams - quitting smoking - wearing sunglasses - taking care of other health problems - maintaining a healthy weigh...
4. `medquad_ans_0000009-13` (score=0.5213) - Here are several things you can do to lower your risk for cataract. Have regular eye exams. Eye exams can help detect cataracts and other age-related eye problems at their earliest stages. If you are age 60 or older, ...
5. `medquad_ans_0000022-4__dup3` (score=0.5208) - The signs and symptoms of type 2 diabetes can be so mild that you might not even notice them. Nearly 7 million people in the United States have type 2 diabetes and dont know they have the disease. Many have no signs o...

**Answer**:

The provided context does not contain information about symptoms of preventing diabetes problems or specifically about keeping eyes healthy in the context of preventing diabetes-related issues. While the context mentions symptoms of diabetes retinal problems and general eye health practices (like regular eye exams and diet), it does not link these to symptoms of preventing diabetes problems. Therefore, based on the given information, the question cannot be answered.

Answer: Insufficient context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.6304) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000424-4` (score=0.6297) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000891-4` (score=0.6297) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000305-4` (score=0.6297) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0001066-4` (score=0.6297) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Aarskog-Scott syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Aarskog-Scott syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Scores**: context_relevance=0.800, groundedness=0.934, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001063-4` (score=0.4883) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
2. `medquad_ans_0001062-4` (score=0.4820) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
3. `medquad_ans_0001063-3` (score=0.4666) - Mutations in the ABCB7 gene cause X-linked sideroblastic anemia and ataxia. The ABCB7 gene provides instructions for making a protein that is critical for heme production. Heme is a component of the hemoglobin protein...
4. `medquad_ans_0001056-3` (score=0.4499) - X-linked intellectual disability, Siderius type is caused by mutations in the PHF8 gene. This gene provides instructions for making a protein that is found in the nucleus of cells, particularly in brain cells before a...
5. `medquad_ans_0006525-1` (score=0.4480) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin. People with X-linked sideroblastic anemia have mature red blood cells that...

**Answer**:

Mutations in the ABCB7 gene cause X-linked sideroblastic anemia and ataxia [medquad_ans_0001063-4]. The ABCB7 gene provides instructions for making a protein that is critical for heme production and plays a role in the formation of certain proteins containing clusters of iron and sulfur atoms, helping maintain iron homeostasis [medquad_ans_0001063-3]. These genetic changes are inherited in an X-linked recessive pattern, meaning the gene is located on the X chromosome, and males are more frequently affected due to having only one X chromosome [medquad_ans_0001063-4].

**Supported citations**: `medquad_ans_0001063-4`, `medquad_ans_0001063-3`
**Unsupported citations**: _none_

### Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000602-1` (score=0.5041) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifi...
2. `medquad_ans_0003601-1` (score=0.4794) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
3. `medquad_ans_0003608-2` (score=0.4703) - What are the signs and symptoms of Limb-girdle muscular dystrophy type 2I? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy type 2I. If the information ...
4. `medquad_ans_0003610-1` (score=0.4636) - Limb-girdle muscular dystrophy type 2C (LGMD2C) is a condition that affects the muscles and is caused by mutations in the gamma-sarcoglycan gene. This condition belongs to a group of muscle disorders called limb-girdl...
5. `medquad_ans_0003607-1` (score=0.4542) - What are the signs and symptoms of Limb-girdle muscular dystrophy type 2H? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy type 2H. If the information ...

**Answer**:

Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs, particularly affecting the proximal muscles closest to the body, such as those in the shoulders, upper arms, pelvic area, and thighs [medquad_ans_0000602-1]. It is a group of disorders that affect the voluntary muscles around the hips and shoulders, characterized by progressive loss of muscle strength and bulk over time [medquad_ans_0003601-1]. The condition may onset in childhood, adolescence, young adulthood, or later in life, and affects males and females equally [medquad_ans_0003601-1]. Most forms are inherited in an autosomal recessive manner, while several rare forms are inherited in an autosomal dominant pattern [medquad_ans_0003601-1]. There

**Supported citations**: `medquad_ans_0000602-1`, `medquad_ans_0003601-1`
**Unsupported citations**: _none_

### Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Scores**: context_relevance=0.833, groundedness=0.909, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000055-3__dup2` (score=0.5147) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision -...
2. `medquad_ans_0000343-1__dup1` (score=0.5119) - Some eye problems are minor and don't last long. But some can lead to a permanent loss of vision. Common eye problems include - Refractive errors - Cataracts - clouded lenses - Glaucoma - a disorder caused by damage t...
3. `medquad_ans_0000275-1__dup1` (score=0.5073) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
4. `medquad_ans_0000055-7` (score=0.5012) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need othe...
5. `medquad_ans_0000054-1__dup4` (score=0.4805) - Diabetes problems are health problems that can happen when you have diabetes. If your diabetes is not under control, you will have too much glucose\*, also called sugar, in your blood. Having too much glucose in your b...

**Answer**:

Diabetic eye problems refer to eye conditions that can occur as a result of high blood glucose levels in people with diabetes. The most common problem is diabetic retinopathy, which damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye [medquad_ans_0000275-1__dup1]. Other eye problems that can affect people with diabetes include cataracts (a clouding of the lens) and glaucoma (a condition caused by increased pressure in the eye that damages the optic nerve) [medquad_ans_0000275-1__dup1]. These conditions may not show symptoms in the early stages, but as they progress, symptoms can include blurry or double vision, rings or flashing lights, dark or floating spots, pain or pressure in the eyes, and trouble seeing out of the corners of the eyes [medquad_ans_00000

**Supported citations**: `medquad_ans_0000275-1__dup1`
**Unsupported citations**: _none_

### Example 88 (`0006449-1`)

**Question**: What is (are) Wilson disease ?

**Scores**: context_relevance=0.800, groundedness=0.495, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0001037-2` (score=0.5964) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
2. `medquad_ans_0000973-1__dup1` (score=0.5717) - Wilson disease is a rare inherited disorder that prevents your body from getting rid of extra copper. You need a small amount of copper from food to stay healthy. Too much copper is poisonous. Normally, your liver rel...
3. `medquad_ans_0000133-1__dup3` (score=0.5708) - Wilson disease is a genetic disease that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filt...
4. `medquad_ans_0000276-3__dup1` (score=0.5699) - Early onset of the disease may foretell a worse prognosis than later onset. If the disorder is detected early and treated appropriately, an individual with WD can usually enjoy normal health and a normal lifespan. If ...
5. `medquad_ans_0001037-1` (score=0.5622) - Wilson disease is an inherited disorder in which excessive amounts of copper accumulate in the body, particularly in the liver, brain, and eyes. The signs and symptoms of Wilson disease usually first appear between th...

**Answer**:

Wilson disease is a rare inherited disorder that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filters extra copper and releases it into bile, a digestive fluid that carries toxins and wastes out of the body. In Wilson disease, the liver does not filter copper correctly, leading to copper buildup in the liver, brain, eyes, and other organs. This accumulation can cause damage to the liver, central nervous system, and other organs, with symptoms typically appearing between ages 5 and 35, most often during the teenage years. The most characteristic sign is a rusty brown ring around the cornea of the eye. If detected early and treated appropriately, individuals can enjoy good health and a normal lifespan; however, untreated Wilson disease can lead to brain damage, liver failure, and death. Treatment involves lifelong medication to remove excess copper and a low-c

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 89 (`0001503-2`)

**Question**: What are the symptoms of Congenital dyserythropoietic anemia type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001504-1` (score=0.5759) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital dyserythropoietic anemia type 3. If the info...
2. `medquad_ans_0001503-2` (score=0.5716) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to abs...
3. `medquad_ans_0002778-1` (score=0.5668) - What are the signs and symptoms of Hemolytic anemia lethal congenital nonspherocytic with genital and other abnormalities? The Human Phenotype Ontology provides the following list of signs and symptoms for Hemolytic a...
4. `medquad_ans_0000706-2__dup1` (score=0.5598) - What are the signs and symptoms of Beta-thalassemia? The Human Phenotype Ontology provides the following list of signs and symptoms for Beta-thalassemia. If the information is available, the table below includes how o...
5. `medquad_ans_0000259-2__dup1` (score=0.5501) - What are the signs and symptoms of Alpha-thalassemia x-linked intellectual disability syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Alpha-thalassemia x-linked intellectua...

**Answer**:

The symptoms of Congenital dyserythropoietic anemia type 2 include jaundice, gallstones, an enlarged liver and spleen, and iron overload, which can lead to complications such as an abnormal heart rhythm (arrhythmia), congestive heart failure, diabetes, and chronic liver disease (cirrhosis). Rarely, mediastinal tumors may occur. Blood transfusions may be necessary during pregnancy or other special circumstances like anemic crisis, major surgery, or infections [medquad_ans_0001503-2].

**Supported citations**: `medquad_ans_0001503-2`
**Unsupported citations**: _none_

### Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.520

**Top docs**:

1. `medquad_ans_0000155-3__dup3` (score=0.5772) - Chronic kidney disease means you have damaged kidneys that cannot filter blood normally. Wastes then build up in your blood, harming your body. Kidney disease usually does not get better and may lead to kidney failure...
2. `medquad_ans_0000038-4__dup3` (score=0.5533) - Different Treatments for Different Stages There are several types of treatments related to kidney disease. Some are used in earlier stages of kidney disease to protect your kidneys. These medications and lifestyle cha...
3. `medquad_ans_0000033-1__dup4` (score=0.5490) - Diabetic kidney disease, also called diabetic nephropathy, is kidney disease caused by diabetes. Even when well controlled, diabetes can lead to chronic kidney disease (CKD) and kidney failure, described as end-stage ...
4. `medquad_ans_0000140-2__dup2` (score=0.5474) - Chronic kidney disease is kidney damage that occurs slowly over many years, often due to diabetes or high blood pressure. Once damaged, the kidneys cant filter blood as they should. This damage can cause wastes to bui...
5. `medquad_ans_0000165-7` (score=0.5422) - - Kidney disease can affect children in various ways, ranging from treatable disorders without long-term consequences to life-threatening conditions. Acute kidney disease develops suddenly, lasts a short time, and can...

**Answer**:

Kidney disease means you have damaged kidneys that cannot filter blood normally, leading to the buildup of wastes in the blood, which can harm the body. Kidney disease usually does not get better and may progress to kidney failure [medquad_ans_0000155-3__dup3].

**Supported citations**: `medquad_ans_0000155-3__dup3`
**Unsupported citations**: _none_

### Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Scores**: context_relevance=0.857, groundedness=0.980, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000058-23` (score=0.5360) - There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But ...
2. `medquad_ans_0000836-5` (score=0.5257) - These resources address the diagnosis or management of prostate cancer: - American College of Radiology: Prostate Cancer Radiation Treatment - Genetic Testing Registry: Familial prostate cancer - Genetic Testing Regis...
3. `medquad_ans_0000058-6__dup1` (score=0.5174) - Choosing Treatment There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade...
4. `medquad_ans_0000036_1-6` (score=0.5144) - Key Points - There are different types of treatment for patients with prostate cancer. - Seven types of standard treatment are used: - Watchful waiting or active surveillance - Surgery - Radiation therapy and radiopha...
5. `medquad_ans_0000006_1-8` (score=0.4991) - New types of treatment are being tested in clinical trials. This summary section refers to new treatments being studied in clinical trials, but it may not mention every new treatment being studied. Information about c...

**Answer**:

The treatments for prostate cancer may involve watchful waiting, surgery, radiation therapy, or hormonal therapy. Some men receive a combination of these therapies [medquad_ans_0000058-23]. A cure is probable for men whose prostate cancer is diagnosed early [medquad_ans_0000058-23].

**Supported citations**: `medquad_ans_0000058-23`
**Unsupported citations**: _none_

### Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Scores**: context_relevance=0.857, groundedness=0.858, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000431-1__dup1` (score=0.6171) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...
2. `medquad_ans_0000196-1__dup3` (score=0.6168) - We all want high-quality health care, but it's hard to know how to choose. There are many things to consider, including - What your insurance covers - Whether a health care provider or service is accredited - The loca...
3. `medquad_ans_0000869-1__dup1` (score=0.6158) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can brin...
4. `medquad_ans_0000586-1__dup2` (score=0.6118) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state ha...
5. `medquad_ans_0000362-1__dup2` (score=0.5946) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...

**Answer**:

Yes, there is information about health insurance. Health insurance helps protect individuals from high medical care costs by being a contract between the individual and an insurance company, where the company agrees to pay part of the expenses when medical care is needed [medquad_ans_0000431-1__dup1]. Many people in the United States obtain health insurance through their employers, with employers often helping to pay for it. These employer-sponsored plans are typically managed care plans that contract with healthcare providers and facilities to offer care at reduced costs [medquad_ans_0000431-1__dup1]. Individuals can also purchase health insurance independently, and certain groups may qualify for government health insurance such as Medicare and Medicaid [medquad_ans_0000431-1__dup1]. Medicaid is specifically a government health insurance program designed to assist low-income individuals in the U.S., with eligibility based on factors like age

**Supported citations**: `medquad_ans_0000431-1__dup1`
**Unsupported citations**: _none_

### Example 93 (`0003860-1`)

**Question**: What is (are) MECP2 duplication syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.928, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003860-3` (score=0.4900) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
2. `medquad_ans_0000635-3` (score=0.4817) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
3. `medquad_ans_0003860-2` (score=0.4741) - What are the signs and symptoms of MECP2 duplication syndrome? MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Infa...
4. `medquad_ans_0003860-4` (score=0.4644) - Is MECP2 duplication syndrome inherited? MECP2 duplication syndrome is inherited in an X-linked manner. A condition is considered X-linked if the genetic change that causes the condition is located on the X chromosome...
5. `medquad_ans_0003860-1` (score=0.4568) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...

**Answer**:

MECP2 duplication syndrome is a genetic disorder caused by an extra copy (duplication) of the MECP2 gene in each cell, which results from a duplication of genetic material on the long (q) arm of the X chromosome [medquad_ans_0003860-3]. The MECP2 gene encodes a protein called MeCP2 that is critical for normal brain function, primarily by regulating the expression of other genes in the brain [medquad_ans_0000635-3]. An extra copy of the MECP2 gene leads to excess MeCP2 protein, which disrupts the proper regulation of gene expression, causing abnormal neuronal function and irregular brain activity [medquad_ans_0003860-3]. This condition occurs almost exclusively in males and is characterized by moderate to severe intellectual disability, severe hypotonia in infancy, and feeding difficulties [medquad_ans_0

**Supported citations**: `medquad_ans_0003860-3`, `medquad_ans_0000635-3`
**Unsupported citations**: _none_

### Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Scores**: context_relevance=0.667, groundedness=0.859, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000028_3-5` (score=0.5855) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer (the size of the tumor and wheth...
2. `medquad_ans_0000017_1-5` (score=0.5752) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - How the melanoma cells look under a microscope. - The si...
3. `medquad_ans_0000034_1-5` (score=0.5638) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The location and size of the ...
4. `medquad_ans_0000028_1-4` (score=0.5622) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) depends mostly on the stage of the cancer and the type of treatment used to remove the cancer. Treatment ...
5. `medquad_ans_0000043_1-4` (score=0.5563) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The patient's age and general...

**Answer**:

The outlook for melanoma depends on several factors, including the stage of the cancer (such as tumor size and whether it has spread to lymph nodes or other parts of the body), the location of the cancer in the body, whether the cancer has just been diagnosed or has recurred, the patient's age and general health, and how deeply the tumor has grown into the skin [medquad_ans_0000028_3-5]. Additionally, for intraocular melanoma specifically, the outlook is influenced by how the melanoma cells look under a microscope, the size and thickness of the tumor, the part of the eye affected, whether the tumor has spread within the eye or to other parts of the body, and certain genetic changes linked to the cancer [medquad_ans_0000017_1-5]. The stage of the cancer, tumor location and size, and recurrence status also impact prognosis [medquad

**Supported citations**: `medquad_ans_0000028_3-5`, `medquad_ans_0000017_1-5`
**Unsupported citations**: _none_

### Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Scores**: context_relevance=0.500, groundedness=0.095, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5880) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000131-4` (score=0.5850) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000945-4` (score=0.5850) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000657-4` (score=0.5850) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0001074-4` (score=0.5850) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention "boomerang dysplasia" or specify its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether boomerang dysplasia is inherited.

Answer: Insufficient context to determine if boomerang dysplasia is inherited.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Scores**: context_relevance=0.900, groundedness=0.950, answer_relevance=0.660

**Top docs**:

1. `medquad_ans_0002838-3` (score=0.4845) - How is hereditary multiple osteochondromas inherited? HMO is caused by mutations in the EXT1 and EXT2 genes. It is inherited in an autosomal dominant pattern, which means that one copy of the altered gene in each cell...
2. `medquad_ans_0002838-1` (score=0.4739) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called...
3. `medquad_ans_0002838-4` (score=0.4671) - Is genetic testing available for hereditary multiple osteochondromas? GeneTests lists the names of laboratories that are performing genetic testing for hereditary multiple osteochondromas. To view the contact informat...
4. `medquad_ans_0000466-3` (score=0.4635) - Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and the EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. The two exosto...
5. `medquad_ans_0000342-4` (score=0.4143) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

Hereditary multiple osteochondromas (HMO) are caused by mutations in the EXT1 and EXT2 genes [medquad_ans_0002838-3][medquad_ans_0002838-1]. These mutations lead to the development of multiple benign bone tumors covered by cartilage, known as osteochondromas [medquad_ans_0002838-1]. The condition is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the condition [medquad_ans_0002838-3].

**Supported citations**: `medquad_ans_0002838-3`, `medquad_ans_0002838-1`
**Unsupported citations**: _none_

### Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Scores**: context_relevance=0.125, groundedness=0.742, answer_relevance=0.325

**Top docs**:

1. `medquad_ans_0000359-2` (score=0.6113) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
2. `medquad_ans_0000989-2` (score=0.6003) - This condition affects an estimated 1 in 50,000 people.
3. `medquad_ans_0000650-2` (score=0.5933) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
4. `medquad_ans_0000790-2` (score=0.5864) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
5. `medquad_ans_0000484-2` (score=0.5847) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.

**Answer**:

The exact incidence of oculofaciocardiodental syndrome is unknown, but it is estimated to affect 1 in 87,000 people [medquad_ans_0000484-2]. Additionally, other estimates range from 1 in 25,000 to 300,000 individuals [medquad_ans_0000790-2], and one source estimates 1 in 50,000 to 100,000 people [medquad_ans_0000650-2]. However, the most specific estimate provided is 1 in 87,000.

**Supported citations**: `medquad_ans_0000484-2`, `medquad_ans_0000790-2`, `medquad_ans_0000650-2`
**Unsupported citations**: _none_

### Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Scores**: context_relevance=1.000, groundedness=0.973, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001350-5` (score=0.5335) - What happens when citrullinemia type I is not treated? Untreated individuals with the severe form of citrullinemia type I have hyperammonemia (plasma ammonia concentration 1000-3000 mol/L). Without prompt intervention...
2. `medquad_ans_0000189-1` (score=0.4972) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused b...
3. `medquad_ans_0001350-2` (score=0.4967) - What are the signs and symptoms of Citrullinemia type I? Citrullinemia type I presents as a clinical spectrum that includes an acute neonatal form, a milder late-onset form, a form without symptoms and/or hyperammonem...
4. `medquad_ans_0001350-4` (score=0.4883) - How is citrullinemia type I inherited? Citrullinemia type I is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosom...
5. `medquad_ans_0000189-2` (score=0.4833) - Type I citrullinemia is the most common form of the disorder, affecting about 1 in 57,000 people worldwide. Type II citrullinemia is found primarily in the Japanese population, where it occurs in an estimated 1 in 100...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused by mutations in different genes [medquad_ans_0000189-1]. Type I citrullinemia (also known as classic citrullinemia) usually becomes evident in the first few days of life, with affected infants appearing normal at birth but later experiencing progressive lack of energy (lethargy), poor feeding, vomiting, seizures, and loss of consciousness [medquad_ans_0000189-1]. A milder form of type I citrullinemia can develop later in childhood or adulthood, associated with intense headaches, partial loss of vision, problems with balance and muscle coordination (ataxia), and lethargy [medquad_ans_00001

**Supported citations**: `medquad_ans_0000189-1`
**Unsupported citations**: _none_

### Example 99 (`0000718-1__dup1`)

**Question**: What is (are) Bilateral frontal polymicrogyria ?

**Scores**: context_relevance=0.833, groundedness=0.725, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000807-1` (score=0.4902) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. The surface of the brain normally has many ridges or folds, called gyri. In people with polymicrogyria, the brain develops...
2. `medquad_ans_0000722-1__dup1` (score=0.4827) - Bilateral perisylvian polymicrogyria (BPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). Signs and symptoms include partial paralysis of muscles on both sides of th...
3. `medquad_ans_0000718-1__dup1` (score=0.4722) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
4. `medquad_ans_0000720-1__dup1` (score=0.4703) - Bilateral generalized polymicrogyria is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). This is the most widespread form of polymicrogyria and typically affects the enti...
5. `medquad_ans_0004956-1` (score=0.4656) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. Specifically, the surface of the brain develops too many folds which are unusually small. The signs and symptoms associate...

**Answer**:

The provided context does not mention bilateral frontal polymicrogyria. Therefore, based on the available information, it is not possible to answer the question. [medquad_ans_0000807-1] discusses bilateral polymicrogyria in general and specific forms like bilateral perisylvian polymicrogyria (BPP), but does not address bilateral frontal polymicrogyria.

**Supported citations**: `medquad_ans_0000807-1`
**Unsupported citations**: _none_

### Example 100 (`0000944-5`)

**Question**: What are the treatments for spondylocostal dysostosis ?

**Scores**: context_relevance=0.429, groundedness=0.672, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000949-5` (score=0.5731) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from Medli...
2. `medquad_ans_0000943-5` (score=0.5408) - These resources address the diagnosis or management of spondylocarpotarsal synostosis syndrome: - Gene Review: Gene Review: FLNB-Related Disorders - Genetic Testing Registry: Spondylocarpotarsal synostosis syndrome Th...
3. `medquad_ans_0000944-5` (score=0.5200) - These resources address the diagnosis or management of spondylocostal dysostosis: - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive - Genetic Testing Registry: Jarcho-Levin syndrome - Genetic ...
4. `medquad_ans_0000187-5` (score=0.5048) - These resources address the diagnosis or management of CHST3-related skeletal dysplasia: - Gene Review: Gene Review: CHST3-Related Skeletal Dysplasia - Genetic Testing Registry: Spondyloepiphyseal dysplasia with conge...
5. `medquad_ans_0000800-5` (score=0.4980) - These resources address the diagnosis or management of platyspondylic lethal skeletal dysplasia, Torrance type: - Genetic Testing Registry: Platyspondylic lethal skeletal dysplasia Torrance type - MedlinePlus Encyclop...

**Answer**:

The provided context does not specify treatments for spondylocostal dysostosis. While it mentions resources such as "Diagnostic Tests," "Drug Therapy," "Surgery and Rehabilitation," "Genetic Counseling," and "Palliative Care" as general categories available for diagnosis and management of various conditions, including spondylocostal dysostosis, no specific treatments are detailed for this condition [medquad_ans_0000944-5]. Therefore, based on the given information, the exact treatments for spondylocostal dysostosis cannot be determined.

**Supported citations**: `medquad_ans_0000944-5`
**Unsupported citations**: _none_

### Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.956, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004825-1` (score=0.6085) - What are the signs and symptoms of Perry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Perry syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0006409-2` (score=0.5613) - What are the signs and symptoms of Weaver syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weaver syndrome. If the information is available, the table below includes how oft...
3. `medquad_ans_0003679-1` (score=0.5597) - What are the signs and symptoms of Lowry Wood syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lowry Wood syndrome. If the information is available, the table below includes...
4. `medquad_ans_0005067-2` (score=0.5488) - What are the signs and symptoms of Primrose syndrome? Signs and symptoms of primrose syndrome that have been reported in the literature include: Severe learning disabilities Boney ear cartilage Cystic changes in to to...
5. `medquad_ans_0004202-2` (score=0.5463) - What are the signs and symptoms of Multiple pterygium syndrome lethal type? The Human Phenotype Ontology provides the following list of signs and symptoms for Multiple pterygium syndrome lethal type. If the informatio...

**Answer**:

The symptoms of Perry syndrome include abnormality of extrapyramidal motor function, respiratory insufficiency, sleep disturbance, tremor, weight loss, developmental regression, hallucinations, hypotension, abnormality of metabolism/homeostasis, apathy, autosomal dominant inheritance, bradykinesia, central hypoventilation, dysarthria, hypoventilation, inappropriate behavior, insomnia, mask-like facies, parkinsonism, rapidly progressive disease, rigidity, short stepped shuffling gait, vertical supranuclear gaze palsy, and weak voice. These symptoms occur in approximately 90% of patients, except for developmental regression, hallucinations, and hypotension, which occur in about 7.5% of patients [medquad_ans_0004825-1].

**Supported citations**: `medquad_ans_0004825-1`
**Unsupported citations**: _none_

### Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Scores**: context_relevance=1.000, groundedness=0.880, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0006398-2` (score=0.6085) - What are the signs and symptoms of Warfarin sensitivity? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin sensitivity. If the information is available, the table below includ...
2. `medquad_ans_0006399-1` (score=0.5260) - What are the signs and symptoms of Warfarin syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin syndrome. If the information is available, the table below includes how...
3. `medquad_ans_0001028-2` (score=0.5063) - The prevalence of warfarin sensitivity is unknown. However, it appears to be more common in people who are older, those with lower body weights, and individuals of Asian ancestry. Of the approximately 2 million people...
4. `medquad_ans_0005207-1` (score=0.4795) - What are the signs and symptoms of Purpura simplex? The Human Phenotype Ontology provides the following list of signs and symptoms for Purpura simplex. If the information is available, the table below includes how oft...
5. `medquad_ans_0005231-1` (score=0.4762) - What are the signs and symptoms of Quebec platelet disorder? The Human Phenotype Ontology provides the following list of signs and symptoms for Quebec platelet disorder. If the information is available, the table belo...

**Answer**:

The signs and symptoms of Warfarin sensitivity include abnormality of blood and blood-forming tissues, abnormality of metabolism/homeostasis, and autosomal dominant inheritance. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which collects information from sources such as Orphanet. However, specific symptom frequencies are not detailed in the provided context [medquad_ans_0006398-2].

**Supported citations**: `medquad_ans_0006398-2`
**Unsupported citations**: _none_

### Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.502, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0004262-2` (score=0.5945) - What are the signs and symptoms of Myhre syndrome? Myhre syndrome is a condition with features affecting many systems and functions of the body. Associated findings might include: Delayed development of language and m...
2. `medquad_ans_0004262-3` (score=0.5837) - What causes Myhre syndrome? Myhre syndrome is caused by mutations in the SMAD4 gene. This gene provides instructions for making a protein involved in transmitting chemical signals from the cell surface to the nucleus....
3. `medquad_ans_0000701-1` (score=0.5788) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most...
4. `medquad_ans_0000701-3` (score=0.5699) - Mutations in the SMAD4 gene cause Myhre syndrome. The SMAD4 gene provides instructions for making a protein involved in transmitting chemical signals from the cell surface to the nucleus. This signaling pathway, calle...
5. `medquad_ans_0000701-2` (score=0.5686) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.

**Answer**:

Myhre syndrome is a condition affecting multiple systems and functions of the body. It is associated with a range of signs and symptoms, including delayed development of language and motor skills (such as crawling and walking), intellectual disability ranging from mild to moderate, behavioral issues such as autism-related developmental disorders, hearing loss (sensorineural, conductive, or mixed), reduced growth both before and after birth (short stature), skin stiffness resulting in a muscular appearance, skeletal abnormalities like thickened skull bones, flattened spine bones (platyspondyly), broad ribs, underdeveloped pelvic wing structures (hypoplastic iliac wings), and short fingers and toes (brachydactyly), joint problems (arthropathy) with stiffness and limited mobility, typical facial features such as narrow eyelid openings (short palpebral fissures), short distance between nose and upper lip (short philtrum), sunken midface (midface hypoplas

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006057-2` (score=0.6018) - What are the signs and symptoms of Thanatophoric dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Thanatophoric dysplasia. If the information is available, the table below ...
2. `medquad_ans_0005135-1` (score=0.5838) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
3. `medquad_ans_0005690-2` (score=0.5730) - What are the signs and symptoms of Sotos syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Sotos syndrome. If the information is available, the table below includes how often...
4. `medquad_ans_0004622-1` (score=0.5668) - What are the signs and symptoms of Osteoglophonic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteoglophonic dysplasia. If the information is available, the table belo...
5. `medquad_ans_0006045-2` (score=0.5633) - What are the signs and symptoms of Tetralogy of Fallot? The Human Phenotype Ontology provides the following list of signs and symptoms for Tetralogy of Fallot. If the information is available, the table below includes...

**Answer**:

The symptoms of Thanatophoric dysplasia include abnormality of the femur, abnormality of the metaphyses, abnormality of the sacroiliac joint, aplasia/hypoplasia of the lungs, bowing of the long bones, brachydactyly syndrome, cognitive impairment, cutis laxa, depressed nasal bridge, increased nuchal translucency, macrocephaly, malar flattening, micromelia, muscular hypotonia, narrow chest, platyspondyly, respiratory insufficiency, short stature, short thorax, skeletal dysplasia, small face, split hand, abnormality of neuronal migration, frontal bossing, hearing impairment, intrauterine growth retardation, kyphosis, polyhydramnios, proptosis, ventriculomegaly, abnormality of the kidney, acanthosis nigricans, atria septal defect,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002417-2` (score=0.5645) - What are the signs and symptoms of Fragile X syndrome? Fragile X syndrome is characterized by developmental problems including intellectual disability and delayed speech and language development. Males are usually mor...
2. `medquad_ans_0003942-1` (score=0.5175) - What are the signs and symptoms of Mental retardation X-linked, South African type? The Human Phenotype Ontology provides the following list of signs and symptoms for Mental retardation X-linked, South African type. I...
3. `medquad_ans_0005014-1` (score=0.5161) - What are the signs and symptoms of PPM-X syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for PPM-X syndrome. If the information is available, the table below includes how often...
4. `medquad_ans_0000259-2__dup1` (score=0.5119) - What are the signs and symptoms of Alpha-thalassemia x-linked intellectual disability syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Alpha-thalassemia x-linked intellectua...
5. `medquad_ans_0002458-2` (score=0.5010) - What are the signs and symptoms of Fryns syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns syndrome. If the information is available, the table below includes how often...

**Answer**:

The symptoms of Fragile X syndrome include cognitive impairment, joint hypermobility, macroorchidism, neurological speech impairment, otitis media, pes planus, abnormality of the pinna, attention deficit hyperactivity disorder, frontal bossing, intellectual disability, moderate, long face, macrocephaly, mandibular prognathia, muscular hypotonia, narrow face, sinusitis, abnormality of the mitral valve, autism, cerebral cortical atrophy, dilatation of the ascending aorta, seizures, self-injurious behavior, and strabismus [medquad_ans_0002417-2]. Additionally, males may have enlarged testicles (macroorchidism) after puberty, and physical features such as a long and narrow face, large ears, prominent jaw and forehead, unusually flexible fingers, and flat feet may become more apparent with age [medquad_ans_000241

**Supported citations**: `medquad_ans_0002417-2`
**Unsupported citations**: _none_

### Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Scores**: context_relevance=1.000, groundedness=0.471, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000166-2` (score=0.5158) - CADASIL is likely a rare condition; however, its prevalence is unknown.
2. `medquad_ans_0000889-2__dup1` (score=0.5094) - What are the signs and symptoms of CADASIL? Strokes are the main feature of CADASIL and often occur repeatedly. Strokes may lead to severe disability such as an inability to walk and urinary incontinence. The average ...
3. `medquad_ans_0000889-5__dup1` (score=0.4963) - How might CADASIL be treated? There is currently no treatment for CADASIL that is proven to be effective. While antiplatelet treatment is often used, it is also not proven to be useful. Migraine should be treated both...
4. `medquad_ans_0000889-1__dup1` (score=0.4859) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited disease of the blood vessels that occurs when the thickening of blood vessel walls blocks the flow ...
5. `medquad_ans_0000056-1__dup1` (score=0.4856) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease that occurs when the thickening of blood vessel walls blocks the fl...

**Answer**:

CADASIL is a rare condition characterized by recurrent strokes, often beginning around age 46, and a slow decline in cognitive function, with over half of affected individuals experiencing cognitive deficits that may start as early as 35 years of age. Approximately 75% of people eventually develop dementia involving significant difficulties with reasoning and memory. Other common features include migraines with aura (in about 35% of cases, first occurring around age 26), psychiatric issues in 30% of cases (such as personality changes or severe depression), epilepsy in 10% of cases (typically presenting in middle age), and various neurological symptoms like gait disturbance, memory impairment, visual impairment, and urinary incontinence. Abnormalities in temperature regulation, retinal vasculature, and neurological speech impairment are present in about 90% of patients. The condition is associated with autosomal dominant inheritance and features such as cerebral cortical at

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000688-1` (score=0.5416) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate...
2. `medquad_ans_0004185-2` (score=0.5298) - What are the signs and symptoms of Multiple endocrine neoplasia type 1? Multiple endocrine neoplasia, type 1 (MEN1) is characterized primarily by several different types of endocrine tumors. People affected by MEN1 ty...
3. `medquad_ans_0004185-6` (score=0.5212) - How might multiple endocrine neoplasia, type 1 be treated? People with multiple endocrine neoplasia, type 1 (MEN1) are usually managed with regular screening to allow for early diagnosis and treatment of endocrine tum...
4. `medquad_ans_0000010-8` (score=0.5085) - - Multiple endocrine neoplasia type 1 (MEN1) is an inherited disorder that causes hormone-secreting tumors in the duodenum and the endocrine glands-most often the parathyroid, pancreas, and pituitary. - Overactive par...
5. `medquad_ans_0000688-5` (score=0.4903) - These resources address the diagnosis or management of multiple endocrine neoplasia: - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 1 - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 2 - Ge...

**Answer**:

Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cells and tissues throughout the body. Multiple endocrine neoplasia typically involves tumors (neoplasia) in at least two endocrine glands; tumors can also develop in other organs and tissues. These growths can be noncancerous (benign) or cancerous (malignant). If the tumors become cancerous, the condition can be life-threatening. The major forms of multiple endocrine neoplasia are called type 1, type 2, and type 4. These types are distinguished by the genes involved, the types of hormones made, and the characteristic signs and symptoms. Many different types of tumors are associated with multiple endocrine neoplasia. Type 1 frequently involves tumors of the

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Scores**: context_relevance=1.000, groundedness=0.768, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004163-2` (score=0.5294) - What are the signs and symptoms of Mucopolysaccharidosis type IVA? The Human Phenotype Ontology provides the following list of signs and symptoms for Mucopolysaccharidosis type IVA. If the information is available, th...
2. `medquad_ans_0004162-1` (score=0.5248) - What are the signs and symptoms of Mucopolysaccharidosis type IV? The Human Phenotype Ontology provides the following list of signs and symptoms for Mucopolysaccharidosis type IV. If the information is available, the ...
3. `medquad_ans_0000681-1` (score=0.5235) - Mucopolysaccharidosis type IV (MPS IV), also known as Morquio syndrome, is a progressive condition that mainly affects the skeleton. The rate at which symptoms worsen varies among affected individuals. The first signs...
4. `medquad_ans_0004155-2` (score=0.5166) - What are the signs and symptoms of Mucopolysaccharidosis type I? The signs and symptoms of MPS I are not present at birth, but they begin to appear during childhood. People with severe MPS I develop the features of th...
5. `medquad_ans_0000682-1` (score=0.5156) - Mucopolysaccharidosis type VI (MPS VI), also known as Maroteaux-Lamy syndrome, is a progressive condition that causes many tissues and organs to enlarge and become inflamed or scarred. Skeletal abnormalities are also ...

**Answer**:

The provided context does not specify the cause of Mucopolysaccharidosis type I. Therefore, based on the given information, the cause cannot be determined. [medquad_ans_0004163-2]

**Supported citations**: `medquad_ans_0004163-2`
**Unsupported citations**: _none_

### Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Scores**: context_relevance=0.800, groundedness=0.300, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000825-3` (score=0.4768) - Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition. The FLCN gene provides instructions for making a protein called follicu...
2. `medquad_ans_0000825-4` (score=0.4575) - When this condition is caused by mutations in the FLCN gene, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, ...
3. `medquad_ans_0005065-1` (score=0.4234) - Primary spontaneous pneumothorax is an abnormal accumulation of air in the pleural space (the space between the lungs and the chest cavity) that can result in the partial or complete collapse of a lung. It is called p...
4. `medquad_ans_0000126-3` (score=0.3990) - Mutations in the FLCN gene cause Birt-Hogg-Dub syndrome. This gene provides instructions for making a protein called folliculin. The normal function of this protein is unknown, but researchers believe that it may act ...
5. `medquad_ans_0000963-3` (score=0.3823) - Surfactant dysfunction is caused by mutations in one of several genes, including SFTPB, SFTPC, and ABCA3. Each of these genes is involved in the production of surfactant. The production and release of surfactant is a ...

**Answer**:

Genetic changes related to primary spontaneous pneumothorax include mutations in the FLCN gene. These mutations can cause primary spontaneous pneumothorax, although they are considered a very rare cause of the condition. The FLCN gene provides instructions for making a protein called folliculin, which is found in connective tissue cells and in cells that line the alveoli. Researchers believe folliculin may help control cell growth and division and may play a role in repairing and re-forming lung tissue. Mutations in the FLCN gene are inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder. However, only about 40 percent of individuals with these mutations go on to develop primary spontaneous pneumothorax [medquad_ans_0000825-3, medquad_ans_0000825-4].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000825-3, medquad_ans_0000825-4`

### Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000066-1__dup4` (score=0.5422) - Stroke -- A Serious Event A stroke is serious, just like a heart attack. Each year in the United States, approximately 795,000 people have a stroke. About 610,000 of these are first or new strokes. On average, one Ame...
2. `medquad_ans_0000066-13` (score=0.5334) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blo...
3. `medquad_ans_0000066-26` (score=0.5232) - With stroke, treatment depends on the stage of the disease. There are three treatment stages for stroke: prevention, therapy immediately after stroke, and rehabilitation after stroke. Stroke treatments include medicat...
4. `medquad_ans_0000124-8` (score=0.5198) - Treatment for a stroke depends on whether it is ischemic or hemorrhagic. Treatment for a transient ischemic attack (TIA) depends on its cause, how much time has passed since symptoms began, and whether you have other ...
5. `medquad_ans_0000066-5__dup2` (score=0.5165) - Stroke is preventable and treatable. A better understanding of the causes of stroke has helped people make lifestyle changes that have cut the stroke death rate nearly in half in the last two decades. Preventing Strok...

**Answer**:

A stroke is sometimes called a "brain attack." Most often, stroke occurs when blood flow to the brain stops because it is blocked by a clot. When this happens, the brain cells in the immediate area begin to die. Some brain cells die because they stop getting the oxygen and nutrients they need to function. Other brain cells die because they are damaged by sudden bleeding into or around the brain. The brain cells that don't die immediately remain at risk for death. These cells can linger in a compromised or weakened state for several hours. With timely treatment, these cells can be saved. New treatments are available that greatly reduce the damage caused by a stroke. But you need to arrive at the hospital as soon as possible after symptoms start to prevent disability and to greatly improve your chances for recovery. Knowing stroke symptoms, calling 911 immediately, and getting to a hospital as quickly as possible are critical [medquad_ans_0000066

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.798, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006276-2` (score=0.5763) - What are the signs and symptoms of Uncombable hair syndrome? Uncombable hair syndrome (UHS) may first become apparent any time between the ages of 3 months and 12 years. It only affects the scalp hair. The quantity of...
2. `medquad_ans_0006276-5` (score=0.5483) - How is uncombable hair syndrome diagnosed? A diagnosis of uncombable hair syndrome (UHS) is made by observing the characteristic symptoms of the condition, as well observing the hair shaft under a special microscope. ...
3. `medquad_ans_0006276-1` (score=0.5360) - Uncombable hair syndrome (UHS) is a rare disorder of the hair shaft of the scalp. It usually is characterized by silvery-blond or straw-colored hair that is disorderly; stands out from the scalp; and cannot be combed ...
4. `medquad_ans_0006276-3` (score=0.5359) - What causes uncombable hair syndrome? The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been s...
5. `medquad_ans_0006276-4` (score=0.5353) - Is uncombable hair syndrome inherited? Uncombable hair syndrome (UHS) is thought to be inherited in an autosomal dominant manner with reduced penetrance. Autosomal dominant means that having a change (mutation) in onl...

**Answer**:

The provided context does not specify the cause of Uncombable hair syndrome. [medquad_ans_0006276-1] mentions it is a rare disorder of the hair, but no underlying cause is detailed in the available information. Therefore, the cause cannot be determined from the given context.

**Supported citations**: `medquad_ans_0006276-1`
**Unsupported citations**: _none_

### Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.829, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5525) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000887-4` (score=0.5520) - Most cases of Russell-Silver syndrome are sporadic, which means they occur in people with no history of the disorder in their family. Less commonly, Russell-Silver syndrome can run in families. In some affected famili...
3. `medquad_ans_0000441-4` (score=0.5518) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has a parent and other family ...
4. `medquad_ans_0000024-4` (score=0.5499) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000959-4` (score=0.5499) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

Russell-Silver syndrome can be inherited in different patterns depending on the family. Most cases are sporadic, meaning they occur without a family history of the disorder. However, in some families, it follows an autosomal dominant pattern of inheritance, where one copy of a mutated gene is sufficient to cause the disorder. In other families, it follows an autosomal recessive pattern of inheritance, where both copies of the gene in each cell are altered [medquad_ans_0000887-4]. Therefore, Russell-Silver syndrome may or may not be inherited, and when inherited, it can follow either an autosomal dominant or autosomal recessive pattern.

**Supported citations**: `medquad_ans_0000887-4`
**Unsupported citations**: _none_

### Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.5834) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0000095-7` (score=0.5420) - Successful weight-loss treatments include setting goals and making lifestyle changes, such as eating fewer calories and being physically active. Medicines and weight-loss surgery also are options for some people if li...
3. `medquad_ans_0000067-29` (score=0.5403) - Certain kinds of chemotherapy and medicines contribute to weight gain. Unfortunately, the usual ways people try to lose weight may not work. Ask your doctor about talking with a nutritionist who can help you plan a he...
4. `medquad_ans_0000022-11` (score=0.5311) - Your eating, diet, and nutrition choices play an important role in preventing or delaying diabetes. Follow the suggestions below to reach and maintain a reasonable weight and make wise food choices most of the time. R...
5. `medquad_ans_0000095-1__dup2` (score=0.5209) - Espaol The terms "overweight" and "obesity" refer to body weight thats greater than what is considered healthy for a certain height. The most useful measure of overweight and obesity is body mass index (BMI). BMI is c...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are different types of weight loss surgery, which often limit the amount of food you can take in. Some types also affect how you digest food and absorb nutrients. All types have risks and complications, such as infections, hernias, and blood clots. Many people who have the surgery lose weight quickly, but some regain weight later. Following diet and exercise recommendations can help keep most of the weight off, and medical follow-up is required for the rest of your life [medquad_ans_0000969-1__dup2].

**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

### Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.762, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000889-5` (score=0.5757) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Sae...
2. `medquad_ans_0000172-2__dup1` (score=0.4797) - Treatment is symptomatic and supportive, and may include drug therapy.
3. `medquad_ans_0000583-5` (score=0.4770) - These resources address the diagnosis or management of laryngo-onycho-cutaneous syndrome: - Genetic Testing Registry: Laryngoonychocutaneous syndrome These resources from MedlinePlus offer information about the diagno...
4. `medquad_ans_0000803-5` (score=0.4765) - These resources address the diagnosis or management of Poland syndrome: - Children's Medical Center of Dallas - Great Ormond Street Hospital (UK): Treatment Options for Symbrachydactyly - St. Louis Children's Hospital...
5. `medquad_ans_0000667-5` (score=0.4724) - These resources address the diagnosis or management of Moebius syndrome: - Boston Children's Hospital - Cleveland Clinic - Genetic Testing Registry: Oromandibular-limb hypogenesis spectrum - Swedish Information Centre...

**Answer**:

The treatments for Saethre-Chotzen syndrome are symptomatic and supportive and may include drug therapy [medquad_ans_0000172-2__dup1]. Additionally, management may involve surgical interventions, genetic counseling, and rehabilitation as part of the broader approach to care [medquad_ans_0000889-5].

**Supported citations**: `medquad_ans_0000172-2__dup1`, `medquad_ans_0000889-5`
**Unsupported citations**: _none_

### Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=1.000, groundedness=0.912, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000758-1__dup2` (score=0.5112) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
2. `medquad_ans_0000078-6` (score=0.5015) - Idiopathic pulmonary fibrosis (IPF) causes the same kind of scarring and symptoms as some other lung diseases. This makes it hard to diagnose. Seeking medical help as soon as you have symptoms is important. If possibl...
3. `medquad_ans_0000512-1` (score=0.4868) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effect...
4. `medquad_ans_0000078-5__dup1` (score=0.4777) - The signs and symptoms of idiopathic pulmonary fibrosis (IPF) develop over time. They may not even begin to appear until the disease has done serious damage to your lungs. Once they occur, they're likely to get worse ...
5. `medquad_ans_0000078-1__dup2` (score=0.4677) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...

**Answer**:

Idiopathic pulmonary fibrosis (IPF) is a type of pulmonary fibrosis in which the lungs become scarred over time, leading to thickening and stiffening of the lung tissue. This scarring makes it difficult to catch one's breath and can result in low oxygen levels in the blood. IPF causes the same kind of scarring and symptoms as some other lung diseases, which makes it challenging to diagnose. In most cases, the cause of IPF cannot be found, which is why it is termed "idiopathic." Symptoms include shortness of breath, a dry hacking cough, fatigue, weight loss, muscle and joint aches, and clubbing of the fingers or toes [medquad_ans_0000758-1__dup2]. Diagnosis is based on medical history, physical exam, and tests such as chest X-ray and high-resolution computed tomography (HRCT) scan, which can show scar tissue and

**Supported citations**: `medquad_ans_0000758-1__dup2`
**Unsupported citations**: _none_

### Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Scores**: context_relevance=0.583, groundedness=0.852, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000213-4__dup1` (score=0.6193) - The NINDS supports research on neurological disorders such as neuroleptic malignant syndrome. Much of this research focuses on finding ways to prevent and treat the disorder.
2. `medquad_ans_0000207-4__dup1` (score=0.5562) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research relating to myoclonus in its laboratories at the National Institutes of Health (NIH) and also supports additional research through ...
3. `medquad_ans_0000213-3__dup1` (score=0.5491) - Early identification of and treatment for individuals with neuroleptic malignant syndrome improves outcome. If clinically indicated, a low potency neuroleptic can be reintroduced very slowly when the individual recove...
4. `medquad_ans_0000264-4__dup1` (score=0.5354) - The NINDS conducts and supports a broad range of research on movement disorders including tardive dyskinesia. The goals of this research are to improve understanding of these disorders and to discover ways to treat, p...
5. `medquad_ans_0000248-4__dup1` (score=0.5188) - Much of the research on RS focuses on answering fundamental questions about the disorder such as how problems in the body's metabolism may trigger the nervous system damage characteristic of RS and what role aspirin p...

**Answer**:

The National Institute of Neurological Disorders and Stroke (NINDS) supports research on neurological disorders such as neuroleptic malignant syndrome, focusing on finding ways to prevent and treat the disorder [medquad_ans_0000213-4__dup1]. Early identification and treatment of individuals with neuroleptic malignant syndrome improve outcomes. In some cases, if clinically indicated, a low potency neuroleptic can be reintroduced very slowly after recovery, though there is a risk of recurrence; alternatively, another class of drugs may be used to substitute for the neuroleptic. Anesthesia may also pose a risk to individuals who have experienced the syndrome [medquad_ans_0000213-3__dup1]. However, specific details about ongoing clinical trials are not provided in the context.

**Supported citations**: `medquad_ans_0000213-4__dup1`, `medquad_ans_0000213-3__dup1`
**Unsupported citations**: _none_

### Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Scores**: context_relevance=0.714, groundedness=0.907, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000221-8` (score=0.5316) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other he...
2. `medquad_ans_0000203-7` (score=0.5311) - People with kidney stones may have pain while urinating, see blood in the urine, or feel a sharp pain in the back or lower abdomen. The pain may last for a short or long time. People may experience nausea and vomiting...
3. `medquad_ans_0000221-6` (score=0.5237) - You may have a kidney stone if you - have pain while urinating - see blood in your urine - feel a sharp pain in your back or lower abdomenthe area between your chest and hips The pain may last for a short or long time...
4. `medquad_ans_0000204-7` (score=0.5220) - The process of diagnosing any illness begins with consideration of the symptoms. Pain or bloody urine may be the first symptom. Urine, blood, and imaging tests will help determine whether symptoms are caused by a ston...
5. `medquad_ans_0000203-11` (score=0.5188) - - A kidney stone is a solid piece of material that forms in a kidney when substances that are normally found in the urine become highly concentrated. - Kidney stones are one of the most common disorders of the urinary...

**Answer**:

To diagnose kidney stones in adults, a doctor will perform a physical exam and ask about medical history, including family history of kidney stones, diet, digestive problems, and other health conditions. The doctor may order urine, blood, and imaging tests to complete the diagnosis. Urine tests can detect infection or substances in the urine that form stones. Blood tests can identify biochemical problems that lead to kidney stones. Imaging tests, such as ultrasound or computerized tomography (CT) scans, are used to locate kidney stones in the body and may also reveal underlying issues that caused the stones to form. Ultrasound uses sound waves to create images of the urinary tract and is considered a good screening test, especially for children, as it does not expose patients to radiation. CT scans use x-rays and computer technology to detect stones, though they may be more useful for detecting small or lower ureter stones [medquad_ans_0000221-8]

**Supported citations**: `medquad_ans_0000221-8`
**Unsupported citations**: _none_

### Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.972, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000041_1-4` (score=0.5098) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - Where the cancer formed in the urethra. - Whether the ca...
2. `medquad_ans_0000041_1-1` (score=0.5052) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of ...
3. `medquad_ans_0000041_1-5` (score=0.4712) - Key Points - After urethral cancer has been diagnosed, tests are done to find out if cancer cells have spread within the urethra or to other parts of the body. - There are three ways that cancer spreads in the body. -...
4. `medquad_ans_0000205-7` (score=0.4550) - Urine retention may occur either because the bladder wall muscles cannot contract or because the sphincter muscles cannot relax. Catheter. A catheter is a thin tube that can be inserted through the urethra into the bl...
5. `medquad_ans_0000041_1-2` (score=0.4510) - Signs of urethral cancer include bleeding or trouble with urination. These and other signs and symptoms may be caused by urethral cancer or by other conditions. There may be no signs or symptoms in the early stages. C...

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra [medquad_ans_0000041_1-1]. The urethra is the tube that carries urine from the bladder to outside the body. In women, the urethra is about 1 inch long and is just above the vagina, while in men, it is about 8 inches long and goes through the prostate gland and the penis to the outside of the body; in men, the urethra also carries semen [medquad_ans_0000041_1-1]. There are different types of urethral cancer that begin in cells that line the urethra, named for the types of cells that become malignant: squamous cell carcinoma (most common), transitional cell carcinoma, and adenocarcinoma [medquad_ans_00000

**Supported citations**: `medquad_ans_0000041_1-1`
**Unsupported citations**: _none_

### Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5602) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000906-4` (score=0.5579) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000305-4` (score=0.5579) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000177-4` (score=0.5579) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0001026-4` (score=0.5579) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention paramyotonia congenita or its mode of inheritance. Therefore, it is not possible to determine from the given information whether paramyotonia congenita is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.530, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000735-2__dup1` (score=0.5714) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound bio...
2. `medquad_ans_0000736-2__dup1` (score=0.5269) - What are the signs and symptoms of Biotin-thiamine-responsive basal ganglia disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Biotin-thiamine-responsive basal ganglia disease....
3. `medquad_ans_0003986-2` (score=0.5169) - What are the signs and symptoms of Methylcobalamin deficiency cbl G type? The Human Phenotype Ontology provides the following list of signs and symptoms for Methylcobalamin deficiency cbl G type. If the information is...
4. `medquad_ans_0004302-1` (score=0.5053) - What are the signs and symptoms of NADH cytochrome B5 reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for NADH cytochrome B5 reductase deficiency. If the informatio...
5. `medquad_ans_0006567-2` (score=0.5049) - What are the signs and symptoms of 2,4-Dienoyl-CoA reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for 2,4-Dienoyl-CoA reductase deficiency. If the information is a...

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), reduced consciousness/confusion (7.5%), respiratory insufficiency (7.5%), skin ulcer (7.5%), and visual field defect (7.5%). Other associated symptoms may include feeding difficulties in infancy, recurrent skin infections, seborrheic dermatitis, sensorineural hearing impairment, hepatomegaly, hyperammonemia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Scores**: context_relevance=0.429, groundedness=0.885, answer_relevance=0.543

**Top docs**:

1. `medquad_ans_0000298-1__dup1` (score=0.5748) - Summary : We all need to drink water. How much you need depends on your size, activity level, and the weather where you live. The water you drink is a combination of surface water and groundwater. Surface water includ...
2. `medquad_ans_0000001-7__dup1` (score=0.5590) - Topics
3. `medquad_ans_0000966-1__dup2` (score=0.5411) - Summary : We all need clean water. People need it to grow crops and to operate factories, and for drinking and recreation. Fish and wildlife depend on it to survive. Many different pollutants can harm our rivers, stre...
4. `medquad_ans_0000269-1__dup4` (score=0.5346) - Frequently Asked Questions (FAQs) Vector Information
5. `medquad_ans_0000331-1__dup1` (score=0.5315) - Summary : Our environment affects our health. If parts of the environment, like the air, water, or soil become polluted, it can lead to health problems. For example, asthma pollutants and chemicals in the air or in th...

**Answer**:

Yes, there is information about water pollution. The context states that many different pollutants can harm rivers, streams, lakes, and oceans. The three most common pollutants are soil, nutrients, and bacteria. Rain washes soil into streams and rivers, which can kill tiny animals and fish eggs, clog fish gills, and block light, causing plants to die. Nutrients, often from fertilizers, cause algae growth in lakes, ponds, and reservoirs, leading to green water. Bacteria, often from sewage spills, can pollute both fresh and salt water. Additionally, actions such as not pouring household products down the drain and properly disposing of pet waste can help protect water supplies [medquad_ans_0000966-1__dup2].

**Supported citations**: `medquad_ans_0000966-1__dup2`
**Unsupported citations**: _none_

### Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Scores**: context_relevance=0.714, groundedness=0.973, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000132-3__dup1` (score=0.5629) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is...
2. `medquad_ans_0000222-3__dup1` (score=0.4984) - The prognosis for those with paresthesia depends on the severity of the sensations and the associated disorders.
3. `medquad_ans_0000106-3__dup1` (score=0.4879) - The prognosis for diabetic neuropathy depends largely on how well the underlying condition of diabetes is handled. Treating diabetes may halt progression and improve symptoms of the neuropathy, but recovery is slow. T...
4. `medquad_ans_0000257-3__dup1` (score=0.4864) - Recovery depends upon how quickly treatment is received and how severely the body is compromised. Paralysis may persist for many weeks or be permanent. Most individuals have a good chance of recovery.
5. `medquad_ans_0000043_1-4` (score=0.4830) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The patient's age and general...

**Answer**:

The outlook for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1].

**Supported citations**: `medquad_ans_0000132-3__dup1`
**Unsupported citations**: _none_

### Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000915-1__dup1` (score=0.6420) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
2. `medquad_ans_0000916-1__dup1` (score=0.6420) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
3. `medquad_ans_0000916-2__dup1` (score=0.6252) - What are the signs and symptoms of Camurati-Engelmann disease? People with Camurati-Engelmann disease have increased bone density, particularly affecting the long bones of the arms and legs (tibia, femur, humerus, uln...
4. `medquad_ans_0000145-1` (score=0.6193) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and h...
5. `medquad_ans_0000916-5__dup1` (score=0.5934) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene...

**Answer**:

Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs (tibia, femur, humerus, ulna, radius). In some cases, the skull and hip bones are also affected. The thickened bones can lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. An increase in the density of the skull results in increased pressure on the brain. The age at which affected individuals first experience symptoms varies greatly; however, most people with this condition develop pain or weakness by adolescence. Camurati-Engelmann disease is caused by a mutation in the TGFB1 gene, which is inherited in an autosomal dominant fashion. In some instances, people have the gene mutation but never develop the characteristic features of the condition, or features are present but a mutation cannot

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Scores**: context_relevance=1.000, groundedness=0.978, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002080-2` (score=0.6118) - What are the signs and symptoms of Emery-Dreifuss muscular dystrophy, X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Emery-Dreifuss muscular dystrophy, X-linked. If the in...
2. `medquad_ans_0003612-1` (score=0.4952) - What are the signs and symptoms of Limb-girdle muscular dystrophy, type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy, type 2G. If the informatio...
3. `medquad_ans_0001985-1` (score=0.4935) - What are the signs and symptoms of Dystelephalangy? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystelephalangy. If the information is available, the table below includes how oft...
4. `medquad_ans_0003604-2` (score=0.4933) - What are the signs and symptoms of Limb-girdle muscular dystrophy type 2A? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy type 2A. If the information ...
5. `medquad_ans_0003608-2` (score=0.4883) - What are the signs and symptoms of Limb-girdle muscular dystrophy type 2I? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy type 2I. If the information ...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked include Achilles tendon contracture, atrioventricular block, childhood onset, decreased cervical spine flexion due to contractures of posterior cervical muscles, elbow flexion contracture, elevated serum creatine phosphokinase, juvenile onset, pectus excavatum, primary atrial arrhythmia, slow progression, sudden cardiac death, type 1 muscle fiber atrophy, waddling gait, and X-linked recessive inheritance [medquad_ans_0002080-2].

**Supported citations**: `medquad_ans_0002080-2`
**Unsupported citations**: _none_

### Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.812, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000072-5` (score=0.5593) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins Mc...
2. `medquad_ans_0001030-5` (score=0.4421) - These resources address the diagnosis or management of Weaver syndrome: - Genetic Testing Registry: Weaver syndrome These resources from MedlinePlus offer information about the diagnosis and management of various heal...
3. `medquad_ans_0000871-3__dup1` (score=0.4308) - How might Budd-Chiari syndrome be treated? The treatment of Budd-Chiari syndrome varies, depending on the cause of the blockage. Medical treatments may include: Blood-thinning (anticoagulation) medications Clot-bustin...
4. `medquad_ans_0000969-5` (score=0.4303) - These resources address the diagnosis or management of Tangier disease: - Genetic Testing Registry: Tangier disease These resources from MedlinePlus offer information about the diagnosis and management of various heal...
5. `medquad_ans_0000193-5` (score=0.4285) - These resources address the diagnosis or management of Coats plus syndrome: - Genetic Testing Registry: Cerebroretinal microangiopathy with calcifications and cysts These resources from MedlinePlus offer information a...

**Answer**:

The provided context does not specify the treatments for arterial tortuosity syndrome. While resources for diagnosis and management are listed, including genetic testing and information from MedlinePlus on diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care, no specific treatments for arterial tortuosity syndrome are detailed [medquad_ans_0000072-5]. Therefore, based on the given information, the treatments for arterial tortuosity syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

### Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Scores**: context_relevance=1.000, groundedness=0.970, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001092-1__dup1` (score=0.5884) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
2. `medquad_ans_0006492-1` (score=0.5852) - What are the signs and symptoms of X-linked Charcot-Marie-Tooth disease type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked Charcot-Marie-Tooth disease type 1. If the in...
3. `medquad_ans_0001094-2__dup1` (score=0.5844) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1E? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1E. If the information is ava...
4. `medquad_ans_0001091-1__dup1` (score=0.5810) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1B? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1B. If the information is ava...
5. `medquad_ans_0001093-1__dup1` (score=0.5729) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is ava...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 1C include decreased motor nerve conduction velocity, distal amyotrophy, distal muscle weakness, distal sensory impairment, hypertrophic nerve changes, hyporeflexia, juvenile onset, onion bulb formation, pes cavus, segmental peripheral demyelination/remyelination, and autosomal dominant inheritance [medquad_ans_0001092-1__dup1].

**Supported citations**: `medquad_ans_0001092-1__dup1`
**Unsupported citations**: _none_

### Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.890, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005321-1` (score=0.5787) - What are the signs and symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal arterial macroaneurysm...
2. `medquad_ans_0000870-1` (score=0.4809) - Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) is a disorder that affects blood vessels in the eyes and heart. The condition generally becomes apparent in infancy or childhood. RAMSVPS d...
3. `medquad_ans_0006001-2` (score=0.4809) - What are the signs and symptoms of Takayasu arteritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Takayasu arteritis. If the information is available, the table below includes h...
4. `medquad_ans_0005160-2` (score=0.4800) - What are the signs and symptoms of Pseudoxanthoma elasticum? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudoxanthoma elasticum. If the information is available, the table belo...
5. `medquad_ans_0005916-1` (score=0.4750) - What are the signs and symptoms of Subaortic stenosis short stature syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Subaortic stenosis short stature syndrome. If the inform...

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) include exudative retinal detachment and pulmonic stenosis. Retinal arterial macroaneurysms can rupture, leading to bleeding within the eye that may cause vision loss. Additionally, supravalvular pulmonic stenosis results in narrowing of the pulmonary artery above the pulmonic valve, impairing blood flow to the lungs and reducing oxygenation of the blood. This leads to symptoms such as shortness of breath. [medquad_ans_0005321-1][medquad_ans_0000870-1]

**Supported citations**: `medquad_ans_0005321-1`, `medquad_ans_0000870-1`
**Unsupported citations**: _none_

### Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001838-2` (score=0.5400) - What are the signs and symptoms of Dicarboxylic aminoaciduria? There are no common signs or symptoms of dicarboxylic aminoaciduria. Hypoglycemia, developmental and neurological abnormalities, and obsessive compulsive ...
2. `medquad_ans_0001838-1` (score=0.5397) - Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. Symptoms have varied greatly among the few reported cases. Dicarboxylic aminoaciduria is...
3. `medquad_ans_0003443-2` (score=0.5101) - What are the signs and symptoms of L-2-hydroxyglutaric aciduria? The Human Phenotype Ontology provides the following list of signs and symptoms for L-2-hydroxyglutaric aciduria. If the information is available, the ta...
4. `medquad_ans_0006585-1` (score=0.5031) - 3-Hydroxyisobutyric aciduria is a rare metabolic condition in which the body is unable to breakdown certain amino acids (the building blocks of protein). This leads to a toxic buildup of particular acids known as orga...
5. `medquad_ans_0001082-1` (score=0.4940) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as...

**Answer**:

Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. It is caused by mutations in the SLC1A1 gene and is inherited in an autosomal recessive fashion. There are no common signs or symptoms; reported cases have shown variability, with some experiencing fasting hypoglycemia, intellectual disability, or developmental and neurological abnormalities, while others have virtually no signs or symptoms [medquad_ans_0001838-1][medquad_ans_0001838-2].

**Supported citations**: `medquad_ans_0001838-1`, `medquad_ans_0001838-2`
**Unsupported citations**: _none_

### Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Scores**: context_relevance=0.500, groundedness=0.102, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5633) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0001082-4` (score=0.5618) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000198-4` (score=0.5618) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000925-4` (score=0.5618) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000936-4` (score=0.5618) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention spondylothoracic dysostosis or its mode of inheritance. Therefore, it is not possible to determine based on the given information whether spondylothoracic dysostosis is inherited.

Answer: Insufficient context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.937, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003296-2` (score=0.4636) - What are the signs and symptoms of Jervell Lange-Nielsen syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Jervell Lange-Nielsen syndrome. If the information is available, th...
2. `medquad_ans_0003295-1` (score=0.4617) - What are the signs and symptoms of Jervell and Lange-Nielsen syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Jervell and Lange-Nielsen syndrome 2. If the information is a...
3. `medquad_ans_0000542-5` (score=0.4584) - These resources address the diagnosis or management of Jervell and Lange-Nielsen syndrome: - Gene Review: Gene Review: Jervell and Lange-Nielsen Syndrome - Genetic Testing Registry: Jervell and Lange-Nielsen syndrome ...
4. `medquad_ans_0003296-1` (score=0.4538) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. I...
5. `medquad_ans_0000542-3` (score=0.4499) - Mutations in the KCNE1 and KCNQ1 genes cause Jervell and Lange-Nielsen syndrome. The KCNE1 and KCNQ1 genes provide instructions for making proteins that work together to form a channel across cell membranes. These cha...

**Answer**:

Jervell Lange-Nielsen syndrome is an autosomal recessive disorder characterized by congenital sensorineural hearing impairment, a prolonged QT interval, sudden cardiac death, syncope, and torsade de pointes [medquad_ans_0003296-2].

**Supported citations**: `medquad_ans_0003296-2`
**Unsupported citations**: _none_

### Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Scores**: context_relevance=0.250, groundedness=0.000, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000359-2` (score=0.5735) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
2. `medquad_ans_0000650-2` (score=0.5696) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
3. `medquad_ans_0000934-2` (score=0.5677) - This condition affects fewer than 1 in 150,000 males and is very rare in females.
4. `medquad_ans_0000989-2` (score=0.5668) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000955-2` (score=0.5590) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.

**Answer**:

The provided context does not contain specific information about the prevalence of congenital cataracts, facial dysmorphism, and neuropathy. Therefore, it is not possible to determine how many people are affected by this condition based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Scores**: context_relevance=0.846, groundedness=0.980, answer_relevance=0.908

**Top docs**:

1. `medquad_ans_0000040-3` (score=0.5078) - Alpha thalassemia X-linked intellectual disability syndrome results from mutations in the ATRX gene. This gene provides instructions for making a protein that plays an essential role in normal development. Although th...
2. `medquad_ans_0000040-4` (score=0.4873) - This condition is inherited in an X-linked recessive pattern. The ATRX gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of the g...
3. `medquad_ans_0001056-3` (score=0.4461) - X-linked intellectual disability, Siderius type is caused by mutations in the PHF8 gene. This gene provides instructions for making a protein that is found in the nucleus of cells, particularly in brain cells before a...
4. `medquad_ans_0000259-1__dup1` (score=0.4430) - Alpha-thalassemia x-linked intellectual disability (ATRX) syndrome is a genetic condition that causes intellectual disability, muscle weakness (hypotonia), short height, a particular facial appearance, genital abnorma...
5. `medquad_ans_0000948-4` (score=0.4359) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The genetic changes related to alpha thalassemia X-linked intellectual disability syndrome are mutations in the ATRX gene. These mutations change the structure of the ATRX protein, which likely prevents it from effectively regulating gene expression. Reduced activity of the HBA1 and HBA2 genes, which are necessary for normal hemoglobin production, causes alpha thalassemia. Abnormal expression of other genes, which have not been identified, probably causes developmental delay, distinctive facial features, and the other signs and symptoms of the syndrome [medquad_ans_0000040-3].

**Supported citations**: `medquad_ans_0000040-3`
**Unsupported citations**: _none_

### Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002622-2` (score=0.6006) - What are the signs and symptoms of Glycogen storage disease type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 6. If the information is available, ...
2. `medquad_ans_0002614-1` (score=0.5565) - What are the signs and symptoms of Glycogen storage disease type 12? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 12. If the information is available...
3. `medquad_ans_0002623-2` (score=0.5492) - What are the signs and symptoms of Glycogen storage disease type 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 7. If the information is available, ...
4. `medquad_ans_0002617-2` (score=0.5468) - What are the signs and symptoms of Glycogen storage disease type 1B? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 1B. If the information is available...
5. `medquad_ans_0002615-2` (score=0.5399) - What are the signs and symptoms of Glycogen storage disease type 13? Glycogen storage disease type 13 causes muscle pain (myalgia). Individuals with GSD13 also experience exercise intolerance, which means they have di...

**Answer**:

The symptoms of Glycogen storage disease type 6 include hypoglycemia (90%), short stature (90%), hepatomegaly, increased hepatic glycogen content, and postnatal growth retardation [medquad_ans_0002622-2].

**Supported citations**: `medquad_ans_0002622-2`
**Unsupported citations**: _none_

### Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Scores**: context_relevance=0.750, groundedness=0.941, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000015-17` (score=0.5738) - Diabetes is a very serious disease. Over time, diabetes that is not well managed causes serious damage to the eyes, kidneys, nerves, and heart, gums and teeth. If you have diabetes, you are more likely than someone wh...
2. `medquad_ans_0000070-7__dup1` (score=0.5669) - - Diabetes is a complex group of diseases with a variety of causes. Scientists believe genes and environmental factors interact to cause diabetes in most cases. - People with diabetes have high blood glucose, also cal...
3. `medquad_ans_0000027-7__dup1` (score=0.5669) - - Diabetes is a complex group of diseases with a variety of causes. Scientists believe genes and environmental factors interact to cause diabetes in most cases. - People with diabetes have high blood glucose, also cal...
4. `medquad_ans_0000071-7__dup2` (score=0.5669) - - Diabetes is a complex group of diseases with a variety of causes. Scientists believe genes and environmental factors interact to cause diabetes in most cases. - People with diabetes have high blood glucose, also cal...
5. `medquad_ans_0000037-7` (score=0.5669) - - Diabetes is a complex group of diseases with a variety of causes. Scientists believe genes and environmental factors interact to cause diabetes in most cases. - People with diabetes have high blood glucose, also cal...

**Answer**:

Diabetes is caused by a combination of genetic and environmental factors. Scientists believe that genes and environmental factors interact to cause diabetes in most cases. Specifically, type 1 diabetes is caused by a lack of insulin due to the destruction of insulin-producing beta cells, which occurs in an autoimmune disease where the body's immune system attacks and destroys these cells. Type 2 diabetes, the most common form, is caused by a combination of factors including insulin resistance, where the body's muscle, fat, and liver cells do not use insulin effectively, and the body's inability to produce enough insulin to compensate for this resistance. Gestational diabetes is caused by hormonal changes and metabolic demands of pregnancy, along with genetic and environmental factors. Monogenic forms of diabetes are caused by mutations in single genes that limit insulin production [medquad_ans_0000070-7__dup1].

**Supported citations**: `medquad_ans_0000070-7__dup1`
**Unsupported citations**: _none_

### Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Scores**: context_relevance=1.000, groundedness=0.846, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000016-21` (score=0.5741) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the lab...
2. `medquad_ans_0000207-1__dup2` (score=0.5079) - Summary : Clinical trials are research studies that test how well new medical approaches work in people. Each study answers scientific questions and tries to find better ways to prevent, screen for, diagnose, or treat...
3. `medquad_ans_0000927-1__dup2` (score=0.4991) - Summary : It seems to happen almost every day - you hear about the results of a new medical research study. Sometimes the results of one study seem to disagree with the results of another study. It's important to be c...
4. `medquad_ans_0000016-19` (score=0.4954) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although bo...
5. `medquad_ans_0000027-22` (score=0.4954) - Through studies in the laboratory and with patients, the National Eye Institute is seeking better ways to detect, treat, and prevent vision loss in people with glaucoma. For example, researchers have discovered genes ...

**Answer**:

Research being done for diabetic retinopathy includes studies in the laboratory and with patients to find better ways to detect, treat, and prevent vision loss in people with diabetes. Specifically, researchers are studying drugs for the treatment of proliferative retinopathy that may reduce the need for laser surgery [medquad_ans_0000016-21]. Additionally, clinical trials are conducted to evaluate how well new medical approaches work in people, including comparing new treatments to existing ones, and these trials follow a protocol that ensures ethical standards and participant safety [medquad_ans_0000207-1__dup2]. Such trials may involve volunteers with diabetic retinopathy or healthy individuals, depending on the study's goals [medquad_ans_0000207-1__dup2].

**Supported citations**: `medquad_ans_0000016-21`, `medquad_ans_0000207-1__dup2`
**Unsupported citations**: _none_

### Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Scores**: context_relevance=0.800, groundedness=0.947, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.5361) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart an...
2. `medquad_ans_0000774-1__dup1` (score=0.5104) - Respiratory failure happens when not enough oxygen passes from your lungs into your blood. Your body's organs, such as your heart and brain, need oxygen-rich blood to work well. Respiratory failure also can happen if ...
3. `medquad_ans_0000651-1__dup1` (score=0.4959) - Summary : Nuclear scans use radioactive substances to see structures and functions inside your body. They use a special camera that detects radioactivity. Before the test, you receive a small amount of radioactive mat...
4. `medquad_ans_0000136-1__dup3` (score=0.4918) - A burn is damage to your body's tissues caused by heat, chemicals, electricity, sunlight or radiation. Scalds from hot liquids and steam, building fires and flammable liquids and gases are the most common causes of bu...
5. `medquad_ans_0000725-1__dup1` (score=0.4844) - A poison is any substance that is harmful to your body. You might swallow it, inhale it, inject it, or absorb it through your skin. Any substance can be poisonous if too much is taken. Poisons can include - Prescripti...

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles in the air from fires and toxic fumes. These substances can damage the eyes and respiratory system and may worsen chronic heart and lung diseases. Symptoms may include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, it may worsen. The best way to prevent inhalation injuries is to limit exposure, especially when smoke is smelled or seen or when fires are nearby [medquad_ans_0000516-1__dup1].

**Supported citations**: `medquad_ans_0000516-1__dup1`
**Unsupported citations**: _none_

### Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Scores**: context_relevance=0.375, groundedness=0.717, answer_relevance=0.375

**Top docs**:

1. `medquad_ans_0000134-4__dup2` (score=0.5737) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has ...
2. `medquad_ans_0000359-2` (score=0.5713) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
3. `medquad_ans_0000989-2` (score=0.5659) - This condition affects an estimated 1 in 50,000 people.
4. `medquad_ans_0000484-2` (score=0.5585) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
5. `medquad_ans_0000650-2` (score=0.5578) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Zollinger-Ellison syndrome affects approximately 1 in 1 million people worldwide [medquad_ans_0000134-4__dup2]. It is also reported to affect about 1 per million people globally [medquad_ans_0000359-2]. Additionally, estimates range from 1 in 50,000 to 1 in 100,000 people [medquad_ans_0000650-2], and one source estimates 1 in 87,000 people [medquad_ans_0000484-2]. However, the most consistent and widely cited figure is 1 in 1 million. [medquad_ans_0000134-4\_\_dup2][medquad_ans_0000359-2]

**Supported citations**: `medquad_ans_0000134-4__dup2`, `medquad_ans_0000359-2`, `medquad_ans_0000650-2`, `medquad_ans_0000484-2`
**Unsupported citations**: _none_

### Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Scores**: context_relevance=0.714, groundedness=0.732, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000974-5` (score=0.5706) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Ge...
2. `medquad_ans_0000124-5` (score=0.5384) - These resources address the diagnosis or management of biotin-thiamine-responsive basal ganglia disease: - Gene Review: Gene Review: Biotin-Thiamine-Responsive Basal Ganglia Disease These resources from MedlinePlus of...
3. `medquad_ans_0000482-5` (score=0.5374) - These resources address the diagnosis or management of histidinemia: - Genetic Testing Registry: Histidinemia These resources from MedlinePlus offer information about the diagnosis and management of various health con...
4. `medquad_ans_0000898-5` (score=0.5333) - These resources address the diagnosis or management of sepiapterin reductase deficiency: - Gene Review: Gene Review: Sepiapterin Reductase Deficiency - Genetic Testing Registry: Sepiapterin reductase deficiency These ...
5. `medquad_ans_0000155-3__dup1` (score=0.5312) - How might adenylosuccinase deficiency be treated? At the current time, there are no effective therapies for the treatment of adenylosuccinase deficiency. Treatment is supportive based on the specific features.

**Answer**:

The provided context does not specify treatments for tetrahydrobiopterin deficiency. It lists diagnostic resources and management-related categories such as diagnostic tests, drug therapy, genetic counseling, and palliative care, but does not detail specific treatments [medquad_ans_0000974-5]. Therefore, based on the available information, no specific treatments for tetrahydrobiopterin deficiency can be identified.

**Supported citations**: `medquad_ans_0000974-5`
**Unsupported citations**: _none_

### Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Scores**: context_relevance=0.375, groundedness=0.740, answer_relevance=0.575

**Top docs**:

1. `medquad_ans_0000359-2` (score=0.5836) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
2. `medquad_ans_0000650-2` (score=0.5773) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
3. `medquad_ans_0000989-2` (score=0.5768) - This condition affects an estimated 1 in 50,000 people.
4. `medquad_ans_0000934-2` (score=0.5744) - This condition affects fewer than 1 in 150,000 males and is very rare in females.
5. `medquad_ans_0000736-2` (score=0.5684) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.

**Answer**:

Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns [medquad_ans_0000736-2]. Therefore, the number of people affected by nonsyndromic holoprosencephaly can be estimated as 25 to 50 percent of 1 in 10,000, which translates to approximately 1 in 20,000 to 1 in 40,000 newborns. However, the exact prevalence of nonsyndromic holoprosencephaly is not directly specified in the provided context. [medquad_ans_0000736-2]

**Supported citations**: `medquad_ans_0000736-2`
**Unsupported citations**: _none_

### Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Scores**: context_relevance=0.636, groundedness=0.660, answer_relevance=0.782

**Top docs**:

1. `medquad_ans_0000525-4` (score=0.4809) - Susceptibility to intrahepatic cholestasis of pregnancy is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to increase the risk of developing the disorde...
2. `medquad_ans_0000525-3` (score=0.4756) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt ex...
3. `medquad_ans_0000749-4` (score=0.4433) - This condition has an autosomal dominant inheritance pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all reported cases have resulted from new mutations in th...
4. `medquad_ans_0000403-4` (score=0.4419) - This condition has an autosomal dominant inheritance pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. All reported cases have resulted from new mutations in the gene ...
5. `medquad_ans_0000948-4` (score=0.4304) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The provided context discusses genetic changes related to intrahepatic cholestasis of pregnancy, specifically involving the ABCB11 and ABCB4 genes, but does not mention progressive familial intrahepatic cholestasis. Therefore, based on the given information, the genetic changes related to progressive familial intrahepatic cholestasis cannot be determined.

[medquad_ans_0000525-3] [medquad_ans_0000525-4]

**Supported citations**: `medquad_ans_0000525-3`, `medquad_ans_0000525-4`
**Unsupported citations**: _none_

### Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.6355) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below in...
2. `medquad_ans_0006271-1` (score=0.5599) - What are the signs and symptoms of Ulna metaphyseal dysplasia syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Ulna metaphyseal dysplasia syndrome. If the information is ava...
3. `medquad_ans_0000773-1__dup1` (score=0.5574) - What are the signs and symptoms of Bone dysplasia lethal Holmgren type? The Human Phenotype Ontology provides the following list of signs and symptoms for Bone dysplasia lethal Holmgren type. If the information is ava...
4. `medquad_ans_0001755-1` (score=0.5571) - What are the signs and symptoms of Deafness, epiphyseal dysplasia, short stature? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness, epiphyseal dysplasia, short stature. If th...
5. `medquad_ans_0001626-1` (score=0.5541) - What are the signs and symptoms of Craniodiaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Craniodiaphyseal dysplasia. If the information is available, the table ...

**Answer**:

The symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

### Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Scores**: context_relevance=0.429, groundedness=0.150, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5187) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000678-4` (score=0.5145) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000438-4` (score=0.5145) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000131-4` (score=0.5145) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000452-4` (score=0.5145) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether hepatic veno-occlusive disease with immunodeficiency is inherited. While the context repeatedly describes autosomal recessive inheritance patterns in general terms, it does not mention this specific condition. Therefore, based on the given information, the inheritance pattern of hepatic veno-occlusive disease with immunodeficiency cannot be determined.

Answer: Insufficient context to determine if hepatic veno-occlusive disease with immunodeficiency is inherited.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Scores**: context_relevance=1.000, groundedness=0.880, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005902-1` (score=0.6472) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how o...
2. `medquad_ans_0001768-1` (score=0.6122) - What are the signs and symptoms of Dehydrated hereditary stomatocytosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Dehydrated hereditary stomatocytosis. If the information is a...
3. `medquad_ans_0005210-2` (score=0.5878) - What are the signs and symptoms of Pycnodysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pycnodysostosis. If the information is available, the table below includes how oft...
4. `medquad_ans_0001548-2` (score=0.5863) - What are the signs and symptoms of Congenital porphyria? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital porphyria. If the information is available, the table below includ...
5. `medquad_ans_0001849-1` (score=0.5800) - What are the signs and symptoms of Diffuse cutaneous systemic sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Diffuse cutaneous systemic sclerosis. If the information is a...

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. These signs and symptoms are based on the Human Phenotype Ontology, which compiles data from sources such as Orphanet [medquad_ans_0005902-1].

**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

### Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Scores**: context_relevance=0.667, groundedness=0.915, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000159-5__dup1` (score=0.4978) - Lupus nephritis is treated with medications that suppress the immune system, so it stops attacking and damaging the kidneys. Standard treatment includes a corticosteroid, usually prednisone, to reduce inflammation in ...
2. `medquad_ans_0000159-7` (score=0.4867) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may n...
3. `medquad_ans_0000159-6` (score=0.4792) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
4. `medquad_ans_0000159-8` (score=0.4764) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed...
5. `medquad_ans_0000186-4__dup1` (score=0.4755) - Investigators researching lupus seek to increase scientific understanding of the disorder and to find ways to treat, prevent, and ultimately, cure it. Several components of the National Institutes of Health support re...

**Answer**:

Treatment for lupus nephritis includes the use of medications that suppress the immune system to prevent damage to the kidneys. Standard treatment typically involves a corticosteroid, such as prednisone, to reduce inflammation in the kidneys. An immunosuppressive medication, such as cyclophosphamide or mycophenolate mofetil, is usually prescribed alongside prednisone to further decrease immune system activity and prevent immune cells from attacking the kidneys directly or producing antibodies that target the kidneys [medquad_ans_0000159-5__dup1]. Hydroxychloroquine, a medication used for treating systemic lupus erythematosus (SLE), should also be prescribed or continued for individuals with lupus nephritis [medquad_ans_0000159-5__dup1].

For patients with lupus nephritis causing high blood pressure, blood pressure-lowering medications are essential. Ang

**Supported citations**: `medquad_ans_0000159-5__dup1`
**Unsupported citations**: _none_

### Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.475, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000869-5` (score=0.5201) - These resources address the diagnosis or management of restless legs syndrome: - Agency for Healthcare Research and Quality: Options for Treating Restless Legs Syndrome - Genetic Testing Registry: Restless legs syndro...
2. `medquad_ans_0000246-3__dup1` (score=0.4942) - RLS is generally a life-long condition for which there is no cure. Symptoms may gradually worsen with age. Nevertheless, current therapies can control the disorder, minimizing symptoms and increasing periods of restfu...
3. `medquad_ans_0000246-2__dup2` (score=0.4667) - For those with mild to moderate symptoms, many physicians suggest certain lifestyle changes and activities to reduce or eliminate symptoms. Decreased use of caffeine, alcohol, and tobacco may provide some relief. Phys...
4. `medquad_ans_0000172-2__dup1` (score=0.4614) - Treatment is symptomatic and supportive, and may include drug therapy.
5. `medquad_ans_0000169-4` (score=0.4604) - NINDS supports a broad range of clinical and basic research on diseases causing sleep disorders in an effort to clarify the mechanisms of these conditions and to develop better treatments for them.

**Answer**:

Treatments for Restless Legs Syndrome (RLS) include lifestyle changes, supplements, and medications. For mild to moderate symptoms, physicians may recommend decreasing the use of caffeine, alcohol, and tobacco, as these can provide some relief. Supplemental correction of deficiencies in iron, folate, and magnesium may also be suggested. Non-pharmacological measures such as taking a hot bath, massaging the legs, or using a heating pad or ice pack can help relieve symptoms in some patients. Medications approved by the U.S. Food and Drug Administration for treating moderate to severe RLS include dopaminergics (e.g., ropinirole, pramipexole), anticonvulsants (e.g., gabapentin enacarbil), opioids, and benzodiazepines (central nervous system depressants). The Relaxis pad, which delivers 30 minutes of vibrations when placed at the site of discomfort, has also

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Scores**: context_relevance=1.000, groundedness=0.927, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002573-6` (score=0.4755) - How might glioblastoma be treated? Unfortunately, there is no cure for glioblastoma. Treatment is palliative and may include surgery, radiation therapy and/or chemotherapy. The best treatment options for each person d...
2. `medquad_ans_0000324-1__dup1` (score=0.4511) - Anaplastic astrocytoma is a rare, cancerous (malignant) type of brain tumor that arises from star-shaped brain cells called astrocytes. These cells surround and protect nerve cells in the brain and spinal cord. An ana...
3. `medquad_ans_0002573-2` (score=0.4390) - What are the signs and symptoms of glioblastoma? Signs and symptoms of glioblastoma vary depending on the size and location of the tumor but may include: Headache Nausea and vomiting Drowsiness Changes in personality ...
4. `medquad_ans_0002574-1` (score=0.4343) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...
5. `medquad_ans_0000006_1-3` (score=0.4261) - The cause of most adult brain and spinal cord tumors is not known.

**Answer**:

Glioblastoma is a type of brain tumor that arises from glial cells, specifically astrocytes, which surround and support nerve cells in the brain. It is a malignant (cancerous) tumor that grows rapidly and is often aggressive. Glioblastoma is typically treated with surgery, radiation therapy, and/or chemotherapy, although there is no cure. Treatment options depend on factors such as tumor size, location, extent of growth into surrounding brain tissue, and the patient's age and overall health. In many cases, it is not possible to remove the entire tumor, so additional therapies are necessary [medquad_ans_0002573-6]. Symptoms may include headache, nausea and vomiting, drowsiness, changes in personality, weakness on one side of the body, memory loss, speech difficulty, changes in vision, and seizures [medquad_ans_0002573-2].

**Supported citations**: `medquad_ans_0002573-6`, `medquad_ans_0002573-2`
**Unsupported citations**: _none_

### Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0005496-3` (score=0.5345) - What causes schizencephaly? The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, s...
2. `medquad_ans_0000251-4__dup1` (score=0.5219) - The NINDS conducts and supports a wide range of studies that explore the mechanisms of normal brain development. The knowledge gained from these fundamental studies provides the foundation for understanding how to pre...
3. `medquad_ans_0005496-1` (score=0.5156) - Schizencephaly is a rare congenital (present from birth) brain malformation in which abnormal slits or clefts form in the cerebral hemispheres of the brain. The signs and symptoms of this condition may include develop...
4. `medquad_ans_0000251-3__dup1` (score=0.5029) - The prognosis for individuals with schizencephaly varies depending on the size of the clefts and the extent of neurological disabilities.
5. `medquad_ans_0000251-1__dup2` (score=0.4842) - Schizencephaly is an extremely rare developmental birth defect characterized by abnormal slits, or clefts, in the cerebral hemispheres of the brain. Babies with clefts in both hemispheres (called bilateral clefts) com...

**Answer**:

The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, schizencephaly can affect more than one family member, supporting a genetic cause in some cases. Schizencephaly has also been linked to a variety of non-genetic factors, including young maternal age and certain medications and infections that can cause vascular disruptions (disruption of blood flow or blood supply) in a developing baby [medquad_ans_0005496-3].

**Supported citations**: `medquad_ans_0005496-3`
**Unsupported citations**: _none_

### Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Scores**: context_relevance=0.900, groundedness=0.550, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000551-3` (score=0.4728) - Mutations in the ALS2 gene cause most cases of juvenile primary lateral sclerosis. This gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but its function is not fully u...
2. `medquad_ans_0000551-4` (score=0.4645) - When caused by mutations in the ALS2 gene, juvenile primary lateral sclerosis is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individu...
3. `medquad_ans_0000052-3` (score=0.4595) - Mutations in several genes can cause familial ALS and contribute to the development of sporadic ALS. Mutations in the C9orf72 gene account for 30 to 40 percent of familial ALS in the United States and Europe. Worldwid...
4. `medquad_ans_0000052-1` (score=0.4533) - Amyotrophic lateral sclerosis (ALS) is a progressive disease that affects motor neurons, which are specialized nerve cells that control muscle movement. These nerve cells are found in the spinal cord and the brain. In...
5. `medquad_ans_0000013-9__dup1` (score=0.4527) - A heritable disease is caused by a mutation, or harmful change, in a gene inherited from a parent. Genes are small structures in your body's cells that determine how you look and tell your body how to work. Examples o...

**Answer**:

Mutations in several genes can cause familial ALS and contribute to the development of sporadic ALS. Mutations in the C9orf72 gene account for 30 to 40 percent of familial ALS in the United States and Europe. Worldwide, SOD1 gene mutations cause 15 to 20 percent of familial ALS, and TARDBP and FUS gene mutations each account for about 5 percent of cases. The other genes associated with familial ALS each account for a small proportion of cases. It is estimated that 60 percent of individuals with familial ALS have an identified genetic mutation. The cause of the condition in the remaining individuals is unknown. The C9orf72, SOD1, TARDBP, and FUS genes are key to the normal functioning of motor neurons and other cells. It is unclear how mutations in these genes contribute to the death of motor neurons, but it is thought that motor neurons are more

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.921, answer_relevance=0.571

**Top docs**:

1. `medquad_ans_0000581-5` (score=0.5928) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Te...
2. `medquad_ans_0003473-6` (score=0.5755) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injectio...
3. `medquad_ans_0000583-5` (score=0.5412) - These resources address the diagnosis or management of laryngo-onycho-cutaneous syndrome: - Genetic Testing Registry: Laryngoonychocutaneous syndrome These resources from MedlinePlus offer information about the diagno...
4. `medquad_ans_0000901-5` (score=0.5370) - These resources address the diagnosis or management of Sheldon-Hall syndrome: - Gillette Children's Hospital - NYU Langone Medical Center These resources from MedlinePlus offer information about the diagnosis and mana...
5. `medquad_ans_0000045-5` (score=0.5316) - These resources address the diagnosis or management of Alstrm syndrome: - Gene Review: Gene Review: Alstrom Syndrome - Genetic Testing Registry: Alstrom syndrome - MedlinePlus Encyclopedia: Acanthosis Nigricans - Medl...

**Answer**:

The only specific treatment available for Laron syndrome is subcutaneous injections of insulin-like growth factor 1 (IGF-1), a growth-promoting hormone. IGF-1 stimulates linear growth (height), improves brain growth, and addresses metabolic abnormalities caused by long-term IGF-1 deficiency. It also raises blood glucose levels, reduces cholesterol, and increases muscle growth. IGF-1 and GH levels should be closely monitored during treatment because overdosage can lead to various health problems [medquad_ans_0003473-6].

**Supported citations**: `medquad_ans_0003473-6`
**Unsupported citations**: _none_

### Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Scores**: context_relevance=1.000, groundedness=0.912, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000016-19` (score=0.5310) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although bo...
2. `medquad_ans_0000016-21` (score=0.4973) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the lab...
3. `medquad_ans_0000016-4__dup2` (score=0.4915) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...
4. `medquad_ans_0000016-15` (score=0.4896) - During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels of blood s...
5. `medquad_ans_0000055-7` (score=0.4854) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need othe...

**Answer**:

Treatments for diabetic retinopathy include laser surgery for proliferative retinopathy, specifically scatter laser treatment, which involves placing 1,000 to 2,000 laser burns in the areas of the retina away from the macula to shrink abnormal blood vessels [medquad_ans_0000016-4__dup2]. For macular edema, prompt treatment with anti-VEGF drugs, with or without laser treatment, results in better vision than laser treatment alone or steroid injections [medquad_ans_0000016-4__dup2]. These drugs reduce fluid leakage and inhibit the growth of new blood vessels in the retina. In some cases, focal laser treatment is used alongside eye injections, where up to several hundred small laser burns are placed around the macula to slow fluid leakage [medquad_ans_0000016-4__dup2]. Both

**Supported citations**: `medquad_ans_0000016-4__dup2`
**Unsupported citations**: _none_

### Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Scores**: context_relevance=1.000, groundedness=0.539, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001421-3` (score=0.5374) - How might collagenous colitis be treated? Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases the condition may resolve on its own (spontaneous re...
2. `medquad_ans_0001007-5` (score=0.5191) - These resources address the diagnosis or management of ulcerative colitis: - American Society of Colon and Rectal Surgeons - Cedars-Sinai - Crohn's & Colitis Foundation of America: Colitis Diagnosis and Testing - Croh...
3. `medquad_ans_0000201-5` (score=0.5004) - These resources address the diagnosis or management of Cole disease: - Genetic Testing Registry: Cole disease These resources from MedlinePlus offer information about the diagnosis and management of various health con...
4. `medquad_ans_0000126-7` (score=0.4648) - Treatment for Crohns disease depends on - where the disease is located in the GI tract - what problems you already have from the disease - what past treatments you have had for the disease The goals of treatment are t...
5. `medquad_ans_0001421-1` (score=0.4602) - Collagenous colitis is a type of inflammatory bowel disease that affects the colon. It is a form of microscopic colitis, which means that the inflammation is only visible when a biopsy is examined under a microscope; ...

**Answer**:

Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases, the condition may resolve on its own (spontaneous remission), although most people continue to have ongoing or occasional diarrhea. Dietary changes are usually tried first to alleviate symptoms, including a reduced-fat diet, eliminating foods that contain caffeine and lactose, and avoiding over-the-counter pain relievers such as ibuprofen or aspirin. If these changes are not sufficient, medications can be used to control symptoms. Prescription anti-inflammatory medications such as mesalamine and sulfasalazine may help reduce swelling. Steroids including budesonide and prednisone can be used to reduce inflammation, but are usually only used for sudden attacks of diarrhea; long-term use is typically avoided due to unwanted side effects. Anti-diarrheal medications such as bismuth subsalicylate, diphenoxylate with atropine, and l

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Scores**: context_relevance=0.556, groundedness=0.054, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000241-7` (score=0.4896) - The best way to avoid hookworm infection is not to walk barefoot in areas where hookworm is common and where there may be human fecal contamination of the soil. Also, avoid other skin contact with such soil and avoid ...
2. `medquad_ans_0000440-7` (score=0.4794) - Wearing shoes and taking other protective measures to avoid skin contact with sand or soil will prevent infection with zoonotic hookworms. Travelers to tropical and subtropical climates, especially where beach exposur...
3. `medquad_ans_0000366-7` (score=0.4666) - Prevention No vaccine is available. The best way to prevent schistosomiasis is to take the following steps if you are visiting or live in an area where schistosomiasis is transmitted: - Avoid swimming or wading in fre...
4. `medquad_ans_0000418-6` (score=0.4590) - Safe and effective prescription drugs are available to treat both Trichinella infection and the symptoms that occur as a result of infection. Treatment should begin as soon as possible; a doctor will make the decision...
5. `medquad_ans_0000144-7` (score=0.4572) - Cystic echinococcosis is controlled by preventing transmission of the parasite. Prevention measures include limiting the areas where dogs are allowed and preventing animals from consuming meat infected with cysts. - P...

**Answer**:

The provided context does not contain information on how to prevent trichinellosis (also known as trichinosis). Therefore, no preventive measures can be cited from the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Scores**: context_relevance=1.000, groundedness=0.930, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000182-3__dup1` (score=0.5874) - A combination of autoimmunity, environmental triggers, and a genetic predisposition can lead to autoimmune hepatitis.
2. `medquad_ans_0000536-3` (score=0.5693) - What causes autoimmune hepatitis? Although the exact cause of autoimmune hepatitis is unknown, evidence suggests that liver injury in a patient with autoimmune hepatitis is the result of a cell-mediated immunologic at...
3. `medquad_ans_0000536-1` (score=0.5450) - Autoimmune hepatitis is a disease in which the bodys immune system attacks liver cells. This immune response causes inflammation of the liver, also called hepatitis. The disease can be quite serious and, if not treate...
4. `medquad_ans_0000182-10` (score=0.5400) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing autoimmune hepatitis.
5. `medquad_ans_0000182-1__dup3` (score=0.5299) - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the normal components, or cells, of the liver and causes inflammation and liver damage. The immune system normally prot...

**Answer**:

Autoimmune hepatitis is caused by a combination of autoimmunity, environmental triggers, and a genetic predisposition. The liver injury results from a cell-mediated immunologic attack, which may be triggered by genetic factors, viral infections, or chemical agents. Evidence suggests a genetic cause, as the disease sometimes occurs in relatives of people with autoimmune diseases [medquad_ans_0000536-3]. However, eating, diet, and nutrition do not play a role in causing or preventing autoimmune hepatitis [medquad_ans_0000182-10].

**Supported citations**: `medquad_ans_0000536-3`, `medquad_ans_0000182-10`
**Unsupported citations**: _none_

### Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Scores**: context_relevance=0.800, groundedness=0.958, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000006_3-1` (score=0.4943) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
2. `medquad_ans_0000006_3-4` (score=0.4830) - The signs and symptoms of astrocytomas are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain or spinal cord. - The size of the tumor. - How fast the tumor gr...
3. `medquad_ans_0000006_3-9` (score=0.4797) - Key Points - There are different types of treatment for patients with childhood astrocytoma. - Children with astrocytomas should have their treatment planned by a team of health care providers who are experts in treat...
4. `medquad_ans_0000006_3-2` (score=0.4725) - The cause of most childhood brain tumors is not known.
5. `medquad_ans_0000186-1__dup2` (score=0.4705) - Brain tumors are abnormal growths inside the skull. They are among the most common types of childhood cancers. Some are benign tumors, which aren't cancer. They can still be serious. Malignant tumors are cancerous. Ch...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They are tumors that start in star-shaped brain cells called astrocytes, which are a type of glial cell that holds nerve cells in place, brings food and oxygen to them, and helps protect them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

### Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Scores**: context_relevance=0.857, groundedness=0.974, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000013_3-4__dup2` (score=0.4788) - Treatment of atypical chronic myelogenous leukemia (CML) may include chemotherapy. Check the list of NCI-supported cancer clinical trials that are now accepting patients with atypical chronic myeloid leukemia, BCR-ABL...
2. `medquad_ans_0000013_3-2__dup3` (score=0.4697) - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising or bleeding and feeling tired and weak. These and other signs and symptoms may be caused by atypical CML or by other conditions. Check ...
3. `medquad_ans_0000013_3-1__dup3` (score=0.4651) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia in...
4. `medquad_ans_0000013_3-3__dup3` (score=0.4382) - Certain factors affect prognosis (chance of recovery). The prognosis (chance of recovery) for atypical CML depends on the number of red blood cells and platelets in the blood.
5. `medquad_ans_0000013_3-1__dup1` (score=0.4182) - Key Points - Myelodysplastic/myeloproliferative neoplasm, unclassifiable, is a disease that has features of both myelodysplastic and myeloproliferative diseases but is not chronic myelomonocytic leukemia, juvenile mye...

**Answer**:

Atypical chronic myelogenous leukemia (CML) is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. The body tells too many blood stem cells to become a type of white blood cell called granulocytes, and some of these blood stem cells never become mature white blood cells. These immature white blood cells are called blasts. Over time, the granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. Leukemia cells in atypical CML and CML look alike under a microscope, but in atypical CML, the "Philadelphia chromosome" chromosome change is not present [medquad_ans_0000013_3-1__dup3]. Signs and symptoms include easy bruising or bleeding and feeling tired and weak [medquad_ans_0000013_3-2__dup3].

**Supported citations**: `medquad_ans_0000013_3-1__dup3`, `medquad_ans_0000013_3-2__dup3`
**Unsupported citations**: _none_

### Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Scores**: context_relevance=0.750, groundedness=0.920, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000996-3` (score=0.4867) - Most cases of trisomy 13 result from having three copies of chromosome 13 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the cha...
2. `medquad_ans_0000996-4` (score=0.4591) - Most cases of trisomy 13 are not inherited and result from random events during the formation of eggs and sperm in healthy parents. An error in cell division called nondisjunction results in a reproductive cell with a...
3. `medquad_ans_0000398-1__dup1` (score=0.4407) - Genes are the building blocks of heredity. They are passed from parent to child. They hold DNA, the instructions for making proteins. Proteins do most of the work in cells. They move molecules from one place to anothe...
4. `medquad_ans_0000373-3` (score=0.4385) - Researchers have identified changes in five regions of the X chromosome that are linked to FG syndrome in affected families. Mutations in a gene called MED12, which is located in one of these regions, appear to be the...
5. `medquad_ans_0000872-3` (score=0.4317) - Mutations in the RB1 gene are responsible for most cases of retinoblastoma. RB1 is a tumor suppressor gene, which means that it normally regulates cell growth and keeps cells from dividing too rapidly or in an uncontr...

**Answer**:

The genetic changes related to trisomy 13 include having three copies of chromosome 13 instead of the usual two copies in each cell, which results from an error in cell division called nondisjunction during the formation of reproductive cells (eggs and sperm) [medquad_ans_0000996-3]. This extra genetic material disrupts normal development and causes the characteristic features of trisomy 13. In some cases, trisomy 13 occurs due to a translocation, where a part of chromosome 13 becomes attached to another chromosome during the formation of reproductive cells or very early in fetal development [medquad_ans_0000996-3]. In these translocation cases, individuals have two normal copies of chromosome 13 plus an extra copy attached to another chromosome. Rarely, only part of chromosome 13 is present in three copies, leading to potentially different

**Supported citations**: `medquad_ans_0000996-3`
**Unsupported citations**: _none_

### Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Scores**: context_relevance=0.571, groundedness=0.924, answer_relevance=0.571

**Top docs**:

1. `medquad_ans_0000731-5` (score=0.5410) - These resources address the diagnosis or management of Niemann-Pick disease: - Baby's First Test - Gene Review: Gene Review: Acid Sphingomyelinase Deficiency - Gene Review: Gene Review: Niemann-Pick Disease Type C - G...
2. `medquad_ans_0004415-3` (score=0.5351) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respirato...
3. `medquad_ans_0000218-2__dup1` (score=0.5313) - There is currently no cure for Niemann-Pick disease. Treatment is supportive. Children usually die from infection or progressive neurological loss. There is currently no effective treatment for persons with type A. Bo...
4. `medquad_ans_0000179-5` (score=0.5233) - These resources address the diagnosis or management of cholesteryl ester storage disease: - Genetic Testing Registry: Lysosomal acid lipase deficiency - MedlinePlus Encyclopedia: Atherosclerosis - MedlinePlus Encyclop...
5. `medquad_ans_0000239-3__dup1` (score=0.5112) - There is currently no treatment that can halt progression of any of the TSEs. Treatment is aimed at alleviating symptoms and making the patient as comfortable as possible. A clinical trial of a potential therapy for C...

**Answer**:

There is no specific treatment for Niemann-Pick disease. Treatment is supportive and may include care from specialists such as pulmonologists for respiratory problems, cardiologists for heart problems, liver and spleen specialists, nutritionists, physical therapists, and gastroenterologists. Learning specialists may also be involved. Bone marrow transplantation has been attempted in a few individuals with type B, and enzyme replacement and gene therapies might be helpful for those with type B. However, restricting one's diet does not prevent the buildup of lipids in cells and tissues. [medquad_ans_0004415-3][medquad_ans_0000218-2__dup1]

**Supported citations**: `medquad_ans_0004415-3`, `medquad_ans_0000218-2__dup1`
**Unsupported citations**: _none_

### Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.533, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001714-1` (score=0.5615) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial poly...
2. `medquad_ans_0003857-1` (score=0.5293) - What are the signs and symptoms of Meckel syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Meckel syndrome type 2. If the information is available, the table below in...
3. `medquad_ans_0003858-1` (score=0.5253) - What are the signs and symptoms of Meckel syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Meckel syndrome type 3. If the information is available, the table below in...
4. `medquad_ans_0005971-1` (score=0.5225) - What are the signs and symptoms of Syndactyly-polydactyly-earlobe syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly-polydactyly-earlobe syndrome. If the informatio...
5. `medquad_ans_0005020-1` (score=0.5104) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the tabl...

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, autosomal recessive inheritance, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, wide nasal bridge [med

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.919, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003789-3` (score=0.4717) - What causes Mantle cell lymphoma? Most lymphomas are not inherited, but rather an acquired disease in which the DNAwithin the cells has been damaged. Damage to the DNA occurs by a combination of different factors. Man...
2. `medquad_ans_0003789-1` (score=0.4658) - Mantle cell lymphoma (MCL) belongs to a group of diseases known as non-Hodgkins lymphomas (NHL), which are cancers that affect the the lymphatic system (part of the immune system). MCL accounts for 6% of all non-Hodgk...
3. `medquad_ans_0003789-2` (score=0.4509) - What are the signs and symptoms of Mantle cell lymphoma? Common symptoms of Mantle cell lymphoma include fatigue, loss of appetite, and enlarged lymph nodes, spleen, and/or liver. Other symptoms may include night swea...
4. `medquad_ans_0003789-4` (score=0.4380) - How is Mantle cell lymphoma diagnosed? Mantle cell lymphoma is diagnosed by a biopsy (surgical removal) of the lymph nodes. If lymph nodes are not easily accessible to be biopsied, a fine needle aspiration may be perf...
5. `medquad_ans_0003789-5` (score=0.4353) - How might Mantle cell lympoma be treated? Various treatmentsare currently available for Mantle cell lymphomas. Rare cases of early stage mantle cell lymphomas may be treated with radiation therapy. For more advance st...

**Answer**:

Mantle cell lymphoma is caused by DNA damage within cells, which can result from a combination of factors. Many cases are associated with a chromosomal translocation. Additionally, viral infections such as Epstein-Barr virus, HIV, and human herpesvirus 6 have been linked to non-Hodgkin lymphomas, and immunodeficiencies and environmental factors like hair dyes and pesticides may also contribute to the development of NHLs [medquad_ans_0003789-3].

**Supported citations**: `medquad_ans_0003789-3`
**Unsupported citations**: _none_

### Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.6027) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0001090-4` (score=0.6021) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000447-4` (score=0.6021) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000018-4` (score=0.6021) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000321-4` (score=0.6021) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context discusses autosomal recessive inheritance patterns and the typical absence of signs and symptoms in parents carrying one copy of a mutated gene. However, none of the documents mention Legg-Calvé-Perthes disease or its mode of inheritance. Therefore, based on the given context, it is not possible to determine whether Legg-Calvé-Perthes disease is inherited.

Answer: Insufficient context to determine if Legg-Calvé-Perthes disease is inherited. [doc_id] does not contain information about Legg-Calvé-Perthes disease.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.532, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003965-1` (score=0.5103) - Metachromatic leukodystrophy is an inherited condition characterized by the accumulation of fats called sulfatides in cells, especially cells of the nervous system. This accumulation results in progressive destruction...
2. `medquad_ans_0000192-1__dup1` (score=0.4902) - Metachromatic leukodystrophy (MLD) is one of a group of genetic disorders called the leukodystrophies, which are characterized by the toxic buildup of lipids (fatty materials such as oils and waxes) and other storage ...
3. `medquad_ans_0003965-2` (score=0.4875) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the ta...
4. `medquad_ans_0003965-3` (score=0.4851) - How is metachromatic leukodystrophy inherited? Metachromatic leukodystrophy is inherited in an autosomal recessive manner. This means that both copies of the disease-causing gene in each cell must have a mutation for ...
5. `medquad_ans_0000647-1` (score=0.4819) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the sub...

**Answer**:

Metachromatic leukodystrophy is an inherited condition characterized by the accumulation of fats called sulfatides in cells, especially in the nervous system. This accumulation leads to progressive destruction of the white matter of the brain, which consists of nerve fibers covered by myelin. Affected individuals experience progressive deterioration of intellectual functions and motor skills, such as the ability to walk, loss of sensation in the extremities, incontinence, seizures, paralysis, inability to speak, blindness, and hearing loss. Eventually, they lose awareness of their surroundings and become unresponsive. The condition is inherited in an autosomal recessive pattern and is caused by mutations in the ARSA and PSAP genes. MLD is one of a group of genetic disorders called leukodystrophies, characterized by the toxic buildup of lipids and other storage materials in the white matter of the central nervous system and peripheral nerves. It results from a deficiency of the enzyme arylsulf

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005135-1` (score=0.6124) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
2. `medquad_ans_0005127-2` (score=0.5428) - What are the signs and symptoms of Pseudoachondroplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudoachondroplasia. If the information is available, the table below includ...
3. `medquad_ans_0004509-1` (score=0.5368) - What are the signs and symptoms of Oculoectodermal syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Oculoectodermal syndrome. If the information is available, the table belo...
4. `medquad_ans_0004790-1` (score=0.5310) - What are the signs and symptoms of Pelvic dysplasia arthrogryposis of lower limbs? The Human Phenotype Ontology provides the following list of signs and symptoms for Pelvic dysplasia arthrogryposis of lower limbs. If ...
5. `medquad_ans_0003780-1` (score=0.5299) - What are the signs and symptoms of Mandibuloacral dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Mandibuloacral dysplasia. If the information is available, the table belo...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies, elbow dislocation, hypoplasia of the zygomatic bone, scoliosis, omphalocele, hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, and tongue-like lumbar vertebral deformities [medquad_ans_0005135-1]. These symptoms occur in approximately 90% of patients for most listed features, except for omphalocele, which occurs in about 7.5% of patients [medquad_ans_0005135-1].

**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

### Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.901, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0001664-2` (score=0.5361) - What are the signs and symptoms of Cri du chat syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Cri du chat syndrome. If the information is available, the table below includ...
2. `medquad_ans_0000251-3` (score=0.5091) - Cri-du-chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected individuals; studies suggest that...
3. `medquad_ans_0001664-3` (score=0.5071) - What causes cri du chat syndrome? Cri du chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected...
4. `medquad_ans_0000251-1` (score=0.5014) - Cri-du-chat (cat's cry) syndrome, also known as 5p- (5p minus) syndrome, is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that ...
5. `medquad_ans_0000251-4` (score=0.4829) - Most cases of cri-du-chat syndrome are not inherited. The deletion occurs most often as a random event during the formation of reproductive cells (eggs or sperm) or in early fetal development. Affected people typicall...

**Answer**:

Cri-du-chat syndrome is a genetic disorder caused by a deletion on the short arm of chromosome 5. It is characterized by a range of signs and symptoms, including a high-pitched, cat-like cry in infancy, cognitive impairment, abnormality of the voice, microcephaly, low-set posteriorly rotated ears, epicanthus, and a round face [medquad_ans_0001664-2]. The Human Phenotype Ontology provides a list of signs and symptoms, with many occurring in approximately 90% of individuals, such as abnormality of chromosome segregation, cognitive impairment, and abnormalities of the voice, face, and ears. Additional features include abnormality of the palate, hypertelorism, intrauterine growth retardation, and hearing impairment. The frequency of these symptoms is based on data from Orphanet and represents rough estimates of how often a feature occurs in people with the condition [medquad*ans*

**Supported citations**: `medquad_ans_0001664-2`
**Unsupported citations**: _none_

### Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Scores**: context_relevance=1.000, groundedness=0.978, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000719-2__dup1` (score=0.5304) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmen...
2. `medquad_ans_0000722-2__dup1` (score=0.5012) - What are the signs and symptoms of Bilateral perisylvian polymicrogyria? The signs and symptoms of bilateral perisylvian polymicrogyria (BPP) vary but may include: Partial paralysis of muscles on both sides of the fac...
3. `medquad_ans_0000718-1__dup1` (score=0.4769) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
4. `medquad_ans_0003881-2` (score=0.4760) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speec...
5. `medquad_ans_0004590-1` (score=0.4698) - What are the signs and symptoms of Orofaciodigital syndrome 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 3. If the information is available, the table ...

**Answer**:

The signs and symptoms of bilateral frontoparietal polymicrogyria may include: moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze, ataxia, strabismus, increased muscle tone, finger dysmetria, ankle clonus, Babinski sign, broad-based gait, cerebellar hypoplasia, cerebral dysmyelination, esotropia, exotropia, hyperreflexia, hypertonia, hypoplasia of the brainstem, nystagmus, polymicrogyria with an anterior to posterior gradient, and truncal ataxia [medquad_ans_0000719-2__dup1].

**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

### Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003414-1` (score=0.5433) - Konigsmark Knox Hussels syndrome is an inherited condition that causes both hearing and vision loss. This condition is characterized by late-onset progressive sensorineural deafness and progressive optic atrophy, whic...
2. `medquad_ans_0003414-2` (score=0.5431) - What are the signs and symptoms of Konigsmark Knox Hussels syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Konigsmark Knox Hussels syndrome. If the information is available...
3. `medquad_ans_0003414-3` (score=0.5224) - What causes Konigsmark Knox Hussels syndrome? Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid ar...
4. `medquad_ans_0003384-2` (score=0.4907) - What are the signs and symptoms of Keutel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Keutel syndrome. If the information is available, the table below includes how oft...
5. `medquad_ans_0003414-4` (score=0.4855) - Is genetic testing available for Konigsmark Knox Hussels syndrome? GeneTests lists the names of laboratories that are performing genetic testing for Konigsmark Knox Hussels syndrome. To view the contact information fo...

**Answer**:

Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid arginine with the amino acid histidine at position 445 in the OPA1 protein [medquad_ans_0003414-3]. This condition is inherited in an autosomal dominant fashion [medquad_ans_0003414-1].

**Supported citations**: `medquad_ans_0003414-3`, `medquad_ans_0003414-1`
**Unsupported citations**: _none_

### Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.714, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004592-1` (score=0.6034) - What are the signs and symptoms of Orofaciodigital syndrome 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 5. If the information is available, the table ...
2. `medquad_ans_0004593-1` (score=0.5986) - What are the signs and symptoms of Orofaciodigital syndrome 8? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 8. If the information is available, the table ...
3. `medquad_ans_0004586-1` (score=0.5983) - What are the signs and symptoms of Orofaciodigital syndrome 11? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 11. If the information is available, the tabl...
4. `medquad_ans_0004590-1` (score=0.5966) - What are the signs and symptoms of Orofaciodigital syndrome 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 3. If the information is available, the table ...
5. `medquad_ans_0004584-2` (score=0.5907) - What are the signs and symptoms of Orofaciodigital syndrome 1? Oral features of OFD1 may include a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormali...

**Answer**:

The provided context does not contain information about the symptoms of Orofaciodigital syndrome 1. [medquad_ans_0004592-1] and [medquad_ans_0004593-1] discuss the signs and symptoms of Orofaciodigital syndrome 5 and Orofaciodigital syndrome 8, respectively. Therefore, based on the given context, the symptoms of Orofaciodigital syndrome 1 cannot be determined.

**Supported citations**: `medquad_ans_0004592-1`, `medquad_ans_0004593-1`
**Unsupported citations**: _none_

### Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Scores**: context_relevance=1.000, groundedness=0.846, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000694-1__dup1` (score=0.6193) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of me...
2. `medquad_ans_0000869-1__dup1` (score=0.6104) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can brin...
3. `medquad_ans_0000018-1__dup2` (score=0.5995) - After any operation, you'll have some side effects. There is usually some pain with surgery. There may also be swelling and soreness around the area that the surgeon cut. Your surgeon can tell you which side effects t...
4. `medquad_ans_0000863-1__dup1` (score=0.5948) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For exa...
5. `medquad_ans_0000001-7__dup1` (score=0.5934) - Topics

**Answer**:

Yes, there is information about surgery. After any operation, patients may experience side effects such as pain, swelling, and soreness around the surgical area. The surgeon can inform patients about expected side effects and potential complications, which may include infection, excessive bleeding, reaction to anesthesia, or accidental injury. Individuals with certain medical conditions may have a higher risk of complications. Patients should also ask about how long they will be in the hospital, what supplies or equipment they may need at home, when they can return to work, when it is safe to resume exercise, and any activity restrictions. Following the surgeon's advice can help facilitate a quicker recovery [medquad_ans_0000018-1__dup2].

**Supported citations**: `medquad_ans_0000018-1__dup2`
**Unsupported citations**: _none_

### Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002935-3` (score=0.5374) - What causes human T-cell leukemia virus, type 2? Human T-cell leukemia virus, type 2 (HTLV-2) occurs when a person is infected by the human T-cell leukemia retrovirus. HTLV-2 is spread by blood transfusions, sexual co...
2. `medquad_ans_0002935-1` (score=0.5245) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affe...
3. `medquad_ans_0002935-5` (score=0.5152) - How might human T-cell leukemia virus, type 2 be treated? No cure or treatment exists for human T-cell leukemia virus, type 2 (HTLV-2). Management is focused on early detection and preventing the spread of HTLV-2 to o...
4. `medquad_ans_0002935-4` (score=0.4968) - How is human T-cell leukemia virus, type 2 diagnosed? Human T-cell leukemia virus, type 2 (HTLV-2) is usually diagnosed based on blood tests that detect antibodies to the virus. However, HTLV-2 is often never suspecte...
5. `medquad_ans_0002935-2` (score=0.4883) - What are the signs and symptoms of human T-cell leukemia virus, type 2? Human T-cell leukemia virus, type 2 (HTLV-2) generally causes no signs or symptoms. Although HTLV-2 has not been definitively linked with any spe...

**Answer**:

Human T-cell leukemia virus type 2 (HTLV-2) is a retroviral infection that affects T cells, a type of white blood cell [medquad_ans_0002935-1]. It is spread through blood transfusions, sexual contact, sharing needles, and from mother to child during birth or breastfeeding [medquad_ans_0002935-3]. The virus generally causes no signs or symptoms, and most infected individuals remain asymptomatic throughout life [medquad_ans_0002935-1]. However, scientists suspect that some people may later develop neurological problems such as sensory neuropathies, gait abnormalities, bladder dysfunction, mild cognitive impairment, and motor abnormalities [medquad_ans_0002935-2]. There is no cure or treatment for HTLV-2, and it is considered a lifelong condition [medquad_ans_000

**Supported citations**: `medquad_ans_0002935-1`, `medquad_ans_0002935-3`, `medquad_ans_0002935-2`
**Unsupported citations**: _none_

### Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Scores**: context_relevance=0.833, groundedness=0.961, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000758-1__dup2` (score=0.5232) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
2. `medquad_ans_0000078-6` (score=0.5132) - Idiopathic pulmonary fibrosis (IPF) causes the same kind of scarring and symptoms as some other lung diseases. This makes it hard to diagnose. Seeking medical help as soon as you have symptoms is important. If possibl...
3. `medquad_ans_0000522-1__dup1` (score=0.5122) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...
4. `medquad_ans_0001152-1` (score=0.4943) - Children's interstitial and diffuse lung disease (chILD) is not a single condition, but a group of rare lung diseases found in infants, children and adolescents that can range from mild to severe. All types of chILD d...
5. `medquad_ans_0001152-2` (score=0.4828) - How might chILD be treated? There is no single treatment for interstitial lung diseases in children. Different forms of chILD require different treatments and support depending on the condition. The goals of treatment...

**Answer**:

Interstitial lung diseases are a large group of diseases that inflame or scar the lungs [medquad_ans_0000758-1__dup2].

**Supported citations**: `medquad_ans_0000758-1__dup2`
**Unsupported citations**: _none_

### Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5814) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0001010-4` (score=0.5784) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000263-4` (score=0.5784) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000960-4` (score=0.5784) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000908-4` (score=0.5784) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Melnick-Needles syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Melnick-Needles syndrome is inherited. [No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Scores**: context_relevance=1.000, groundedness=0.977, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006138-6` (score=0.4888) - How might tracheobronchomalacia be treated? Treatment is only medically necessary in people who have signs and symptoms of tracheobronchomalacia (TBM). Management of symptomatic TBM first involves identifying underlyi...
2. `medquad_ans_0006138-1` (score=0.4828) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of ...
3. `medquad_ans_0006138-3` (score=0.4816) - What causes tracheobronchomalacia? The underlying cause of tracheobronchomalacia (TBM) varies by subtype. Most cases of primary TBM (also called congenital TBM) are caused by genetic conditions that weaken the walls o...
4. `medquad_ans_0006138-2` (score=0.4760) - What are the signs and symptoms of tracheobronchomalacia? Tracheobronchomalacia (TBM) is a condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway ...
5. `medquad_ans_0006138-5` (score=0.4724) - How is tracheobronchomalacia diagnosed? A diagnosis of tracheobronchomalacia (TBM) may be suspected based on the presence of characteristic signs and symptoms or abnormal pulmonary function tests. Additional testing s...

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse [medquad_ans_0006138-1]. There are two forms of TBM: a congenital form (called primary TBM) that typically develops during infancy or early childhood, and an acquired form (called secondary TBM) that is usually seen in adults [medquad_ans_0006138-1]. Most cases of primary TBM are caused by genetic conditions that weaken the airway walls, such as mucopolysaccharidoses (e.g., Hunter syndrome and Hurler syndrome), Ehlers-Danlos Syndrome, and various chromosome abnormalities. Primary TBM can also be idiopathic or associated with prematurity and certain birth defects (e.g.,

**Supported citations**: `medquad_ans_0006138-1`
**Unsupported citations**: _none_

### Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Scores**: context_relevance=0.600, groundedness=0.993, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006030-1` (score=0.5429) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people m...
2. `medquad_ans_0000874-2` (score=0.4572) - This condition affects an estimated 1 in 8,500 females.
3. `medquad_ans_0000880-1__dup2` (score=0.4430) - The temporomandibular joint (TMJ) connects your jaw to the side of your head. When it works well, it enables you to talk, chew, and yawn. For people with TMJ dysfunction, problems with the joint and muscles around it ...
4. `medquad_ans_0000531-1__dup2` (score=0.4372) - Your jaw is a set of bones that holds your teeth. It consists of two main parts. The upper part is the maxilla. It doesn't move. The moveable lower part is called the mandible. You move it when you talk or chew. The t...
5. `medquad_ans_0000102-3__dup1` (score=0.4360) - The progression of the disorder is usually 10 years or longer.

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

### Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5749) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000683-4` (score=0.5724) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000898-4` (score=0.5724) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0001090-4` (score=0.5724) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000896-4` (score=0.5724) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Kallmann syndrome or its mode of inheritance. Therefore, it is not possible to determine based on the given information whether Kallmann syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Scores**: context_relevance=1.000, groundedness=0.895, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005967-2` (score=0.6093) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how...
2. `medquad_ans_0005971-1` (score=0.5756) - What are the signs and symptoms of Syndactyly-polydactyly-earlobe syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly-polydactyly-earlobe syndrome. If the informatio...
3. `medquad_ans_0005022-1` (score=0.5681) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the tabl...
4. `medquad_ans_0005020-1` (score=0.5680) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the tabl...
5. `medquad_ans_0006572-2` (score=0.5646) - What are the signs and symptoms of 22q13.3 deletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 22q13.3 deletion syndrome. If the information is available, the table be...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and autosomal dominant inheritance [medquad_ans_0005967-2].

**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

### Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Scores**: context_relevance=0.800, groundedness=0.853, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000197-1__dup3` (score=0.5321) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...
2. `medquad_ans_0000021-4__dup2` (score=0.4962) - Bronchitis is a very common condition. Millions of cases occur every year. Elderly people, infants, and young children are at higher risk for acute bronchitis than people in other age groups. People of all ages can de...
3. `medquad_ans_0000021-7` (score=0.4845) - The main goals of treating acute and chronic bronchitis are to relieve symptoms and make breathing easier. If you have acute bronchitis, your doctor may recommend rest, plenty of fluids, and aspirin (for adults) or ac...
4. `medquad_ans_0000232-1__dup2` (score=0.4725) - COPD (chronic obstructive pulmonary disease) makes it hard for you to breathe. The two main types are chronic bronchitis and emphysema. The main cause of COPD is long-term exposure to substances that irritate and dama...
5. `medquad_ans_0000008-1__dup3` (score=0.4612) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...

**Answer**:

Chronic bronchitis is an inflammation of the bronchial tubes, the airways that carry air to the lungs, characterized by a persistent cough that produces mucus. It is one type of chronic obstructive pulmonary disease (COPD) and is often caused by cigarette smoking, as well as long-term exposure to air pollution, fumes, or dust. The condition leads to difficulty breathing and is a long-term illness that tends to recur or never fully resolves. Diagnosis involves evaluating symptoms, listening to breathing, and possibly other tests. Treatment focuses on relieving symptoms and improving breathing, often including medications to open airways, clear mucus, and, in some cases, oxygen therapy or pulmonary rehabilitation [medquad_ans_0000197-1__dup3]. Women are more than twice as likely as men to be diagnosed with chronic bronchitis, and it is more common in individuals over 45 years of age

**Supported citations**: `medquad_ans_0000197-1__dup3`
**Unsupported citations**: _none_

### Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Scores**: context_relevance=0.750, groundedness=0.929, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000937-5` (score=0.5279) - These resources address the diagnosis or management of SMARD1: - Genetic Testing Registry: Spinal muscular atrophy with respiratory distress 1 These resources from MedlinePlus offer information about the diagnosis and...
2. `medquad_ans_0000934-5` (score=0.5228) - These resources address the diagnosis or management of spinal and bulbar muscular atrophy: - Gene Review: Gene Review: Spinal and Bulbar Muscular Atrophy - Genetic Testing Registry: Bulbo-spinal atrophy X-linked - Med...
3. `medquad_ans_0000936-5` (score=0.5199) - These resources address the diagnosis or management of spinal muscular atrophy with progressive myoclonic epilepsy: - Genetic Testing Registry: Jankovic Rivera syndrome - Muscular Dystrophy Association: Spinal Muscula...
4. `medquad_ans_0000173-2__dup2` (score=0.4946) - There is no cure for SMA. Treatment consists of managing the symptoms and preventing complications.
5. `medquad_ans_0000197-2__dup2` (score=0.4835) - There is no cure or standard treatment for the MNDs. Symptomatic and supportive treatment can help patients be more comfortable while maintaining their quality of life. The drug riluzole (Rilutek), which as of this da...

**Answer**:

There is no cure for spinal muscular atrophy (SMA). Treatment consists of managing symptoms and preventing complications [medquad_ans_0000173-2__dup2]. Management includes diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care [medquad_ans_0000937-5]. Specific therapies may involve physical and speech therapy, occupational therapy, and rehabilitation to improve posture, prevent joint immobility, slow muscle weakness and atrophy, and address swallowing difficulties [medquad_ans_0000197-2__dup2]. Assistive devices such as braces, orthotics, speech synthesizers, and wheelchairs may also be used [medquad_ans_0000197-2__dup2].

**Supported citations**: `medquad_ans_0000173-2__dup2`, `medquad_ans_0000937-5`, `medquad_ans_0000197-2__dup2`
**Unsupported citations**: _none_

### Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Scores**: context_relevance=0.889, groundedness=0.906, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000001_1-7` (score=0.5299) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - C...
2. `medquad_ans_0000001_6-7` (score=0.5298) - Key Points - There are different types of treatment for childhood acute lymphoblastic leukemia (ALL). - Children with ALL should have their treatment planned by a team of doctors who are experts in treating childhood ...
3. `medquad_ans_0000267-5` (score=0.5289) - These resources address the diagnosis or management of cytogenetically normal acute myeloid leukemia: - Fred Hutchinson Cancer Research Center - National Cancer Institute: Acute Myeloid Leukemia Treatment - St. Jude C...
4. `medquad_ans_0000040-18` (score=0.5271) - Treatment depends on a number of factors, including the type of leukemia, the patient's age and general health, where leukemia cells have collected in the body, and whether the leukemia has been treated before. Certai...
5. `medquad_ans_0000040-19` (score=0.5165) - Standard treatments for leukemia include chemotherapy, biological therapy, radiation therapy, and immunotherapy. Some patients receive a combination of treatments. Learn more about treatments for acute myeloid leukemi...

**Answer**:

The treatments for Adult Acute Lymphoblastic Leukemia (ALL) include chemotherapy, radiation therapy, chemotherapy with stem cell transplant, targeted therapy, and biologic therapy. The treatment is typically divided into two phases: remission induction therapy, which aims to kill leukemia cells in the blood and bone marrow to achieve remission, and post-remission therapy, which targets any remaining leukemia cells to prevent relapse. Central nervous system (CNS) sanctuary therapy is also commonly given during each phase to address leukemia cells that may hide in the brain and spinal cord. Additionally, patients may consider participating in clinical trials, which are research studies testing new or improved treatments. [medquad_ans_0000001_1-7]

**Supported citations**: `medquad_ans_0000001_1-7`
**Unsupported citations**: _none_

### Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Scores**: context_relevance=1.000, groundedness=0.733, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000382-1__dup2` (score=0.5084) - Friedreich's ataxia is an inherited disease that damages your nervous system. The damage affects your spinal cord and the nerves that control muscle movement in your arms and legs. Symptoms usually begin between the a...
2. `medquad_ans_0002439-2` (score=0.5038) - What are the signs and symptoms of Friedreich ataxia? Symptoms usually begin between the ages of 5 and 15 but can, on occasion, appear in adulthood or even as late as age 75. The first symptom to appear is usually dif...
3. `medquad_ans_0000133-4__dup1` (score=0.4966) - Friedreich's ataxia is caused by a mutation in the protein frataxin, which is involved in the function of mitochondriathe energy producing power plants of the cell. Frataxin controls important steps in mitochondrial i...
4. `medquad_ans_0002439-3` (score=0.4861) - What causes Friedreich ataxia? Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known...
5. `medquad_ans_0005768-2` (score=0.4799) - What are the signs and symptoms of Spinocerebellar ataxia 2? Early symptoms of spinocerebellar ataxia may include uncoordinated movement (ataxia) and leg cramps. Other symptoms may include tremor; decreased muscle ton...

**Answer**:

The provided context does not specify the cause of Friedreich ataxia. It describes the symptoms, progression, and associated conditions but does not mention the underlying genetic or biological cause. Therefore, based on the given information, the cause cannot be determined. [medquad_ans_0000382-1__dup2] [medquad_ans_0002439-2]

**Supported citations**: `medquad_ans_0000382-1__dup2`, `medquad_ans_0002439-2`
**Unsupported citations**: _none_

### Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Scores**: context_relevance=0.833, groundedness=0.481, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000341-4` (score=0.5441) - Familial atrial fibrillation appears to be inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
2. `medquad_ans_0000341-1` (score=0.5008) - Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm. This condition is characterized by uncoordinated electrical activity in the heart's upper chambers (the atria), which cau...
3. `medquad_ans_0002230-1` (score=0.4889) - Familial atrial fibrillation is an inherited heart condition that disrupts the heart's rhythm. It is characterized by erratic electrical activity in the heart's upper chambers (the atria), causing an irregular respons...
4. `medquad_ans_0000341-3` (score=0.4724) - A small percentage of all cases of familial atrial fibrillation are associated with changes in the KCNE2, KCNJ2, and KCNQ1 genes. These genes provide instructions for making proteins that act as channels across the ce...
5. `medquad_ans_0000341-2` (score=0.4609) - Atrial fibrillation is the most common type of sustained abnormal heart rhythm (arrhythmia), affecting more than 3 million people in the United States. The risk of developing this irregular heart rhythm increases with...

**Answer**:

Familial atrial fibrillation is an inherited heart condition that disrupts the heart's normal rhythm due to uncoordinated or erratic electrical activity in the heart's upper chambers (the atria), leading to a fast and irregular heartbeat (arrhythmia). This condition increases the risk of stroke and sudden death and may present with symptoms such as dizziness, chest pain, palpitations, shortness of breath, or fainting. It can occur at any age, though some individuals never experience associated health problems. The condition is most often inherited in an autosomal dominant manner, though autosomal recessive inheritance has been reported. It may be caused by mutations in various genes, including KCNE2, KCNJ2, and KCNQ1, which are involved in regulating potassium ion flow in cardiac muscle cells. Mutations in these genes increase channel activity, disrupting ion transport and altering heart rhythm, thereby increasing the risk of syncope, stroke, and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.5344) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the assoc...
2. `medquad_ans_0004627-2` (score=0.5334) - What are the signs and symptoms of Osteopathia striata cranial sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopathia striata cranial sclerosis. If the information is...
3. `medquad_ans_0000922-1` (score=0.4338) - SOST-related sclerosing bone dysplasia is a disorder of bone development characterized by excessive bone formation (hyperostosis). As a result of hyperostosis, bones throughout the body are denser and wider than norma...
4. `medquad_ans_0000874-2` (score=0.4257) - This condition affects an estimated 1 in 8,500 females.
5. `medquad_ans_0000645-1__dup2` (score=0.4197) - The brain, spinal cord, and nerves make up the nervous system. Together they control all the workings of the body. When something goes wrong with a part of your nervous system, you can have trouble moving, speaking, s...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary significantly from person to person, even within a single family. Features of the condition are generally present at birth and may include skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected people may also have developmental delay, hearing loss, heart defects, and/or ophthalmoplegia (paralysis of the muscles surrounding the eyes). Osteopathia striata cranial sclerosis is caused by changes (mutations) in the WTX gene and is inherited in an X-linked dominant manner. Treatment is based on the signs and symptoms present in each person [medquad_ans_0004

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Scores**: context_relevance=1.000, groundedness=0.537, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004097-2` (score=0.5739) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus eryt...
2. `medquad_ans_0001849-1` (score=0.5421) - What are the signs and symptoms of Diffuse cutaneous systemic sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Diffuse cutaneous systemic sclerosis. If the information is a...
3. `medquad_ans_0002280-2` (score=0.5328) - What are the signs and symptoms of Familial mixed cryoglobulinemia? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial mixed cryoglobulinemia. If the information is available, ...
4. `medquad_ans_0004097-1` (score=0.5278) - Mixed connective tissue disease (MCTD) is a rare autoimmune disorder that is characterized by features commonly seen in three different connective tissue disorders: systemic lupus erythematosus, scleroderma, and polym...
5. `medquad_ans_0003615-2` (score=0.5275) - What are the signs and symptoms of Limited cutaneous systemic sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Limited cutaneous systemic sclerosis. If the information is a...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. Common symptoms include pain in multiple joints and/or inflammation of joints (arthritis), muscle weakness, fevers, and fatigue. Other signs and symptoms may include accumulation of fluid in the tissue of the hands (edema), skin findings such as lupus-like rashes, reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia). Dysfunction of the esophagus (hypomotility), abnormalities in lung function leading to breathing difficulties or pulmonary hypertension, heart involvement (such as pericarditis, myocarditis, or aortic insufficiency), kidney disease, neuro

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Scores**: context_relevance=0.857, groundedness=0.827, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000098-2__dup1` (score=0.5367) - Drugs to specifically treat Alzheimers disease and some other progressive dementias are now available. Although these drugs do not halt the disease or reverse existing brain damage, they can improve symptoms and slow ...
2. `medquad_ans_0000004-5__dup2` (score=0.5086) - Medications Can Treat Symptoms There is no known cure for Alzheimer's disease, but there are medicines that can treat symptoms of the disease. Most Alzheimers medicines work best for people in the mild or moderate sta...
3. `medquad_ans_0000048-5` (score=0.4948) - These resources address the diagnosis or management of Alzheimer disease: - Alzheimer's Disease Research Center, Washington University School of Medicine - Gene Review: Gene Review: Alzheimer Disease Overview - Gene R...
4. `medquad_ans_0000004-21` (score=0.4931) - An early, accurate diagnosis of Alzheimer's disease helps people and their families plan for the future. It gives them time to discuss care options, find support, and make legal and financial arrangements while the pe...
5. `medquad_ans_0000172-2__dup1` (score=0.4878) - Treatment is symptomatic and supportive, and may include drug therapy.

**Answer**:

The provided context does not contain information about treatments specifically for Alzheimer's caregiving. While it discusses medications that treat symptoms of Alzheimer's disease and cognitive exercises that may benefit individuals in the early stages, there is no mention of treatments or strategies aimed at supporting caregivers. Therefore, based on the given context, it is not possible to answer the question about treatments for Alzheimer's caregiving.

[medquad_ans_0000098-2__dup1] [medquad_ans_0000004-5__dup2]

**Supported citations**: `medquad_ans_0000098-2__dup1`, `medquad_ans_0000004-5__dup2`
**Unsupported citations**: _none_

### Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Scores**: context_relevance=0.900, groundedness=0.910, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000511-3` (score=0.4229) - Idiopathic inflammatory myopathy is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Researchers have identi...
2. `medquad_ans_0000628-3` (score=0.4119) - Mutations in the SIL1 gene cause Marinesco-Sjgren syndrome. The SIL1 gene provides instructions for producing a protein located in a cell structure called the endoplasmic reticulum. Among its many functions, the endop...
3. `medquad_ans_0000705-3` (score=0.4036) - Mutations in several genes can cause myofibrillar myopathy. These genes provide instructions for making proteins that play important roles in muscle fibers. Within muscle fibers, these proteins are involved in the ass...
4. `medquad_ans_0000293-3` (score=0.3996) - A mutation in the MATR3 gene has been identified in people with distal myopathy 2. This gene provides instructions for making a protein called matrin 3, which is found in the nucleus of the cell as part of the nuclear...
5. `medquad_ans_0000202-3` (score=0.3991) - Mutations in the COL6A1, COL6A2, and COL6A3 genes can cause the various forms of collagen VI-related myopathy. These genes each provide instructions for making one component of a protein called type VI collagen. Type ...

**Answer**:

Genetic changes related to idiopathic inflammatory myopathy include variations in several genes, most commonly within the human leukocyte antigen (HLA) complex. These HLA gene variations may influence the risk of developing the disorder by affecting how the immune system distinguishes the body's own proteins from foreign proteins. Researchers are also studying other genes related to immune function to understand their contribution to the risk. However, the specific genetic variations that increase risk are not fully defined, and it is believed that genetic factors combined with environmental triggers lead to the disorder [medquad_ans_0000511-3].

**Supported citations**: `medquad_ans_0000511-3`
**Unsupported citations**: _none_

### Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0005223-5` (score=0.5664) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for...
2. `medquad_ans_0005225-5` (score=0.5544) - Is genetic testing available for pyruvate kinase deficiency? Yes. GeneTests lists laboratories offering clinical genetic testing for this condition. Clinical genetic tests are ordered to help diagnose an affected pers...
3. `medquad_ans_0000855-5` (score=0.5231) - These resources address the diagnosis or management of pyruvate dehydrogenase deficiency: - Genetic Testing Registry: Pyruvate dehydrogenase E1-beta deficiency - Genetic Testing Registry: Pyruvate dehydrogenase E2 def...
4. `medquad_ans_0005222-1` (score=0.5180) - What are the signs and symptoms of Pyruvate decarboxylase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyruvate decarboxylase deficiency. If the information is availab...
5. `medquad_ans_0000855-2` (score=0.5023) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.

**Answer**:

The diagnosis of pyruvate dehydrogenase (PDH) deficiency may be considered in any individual with early-onset neurological disease, especially if it appears to be associated with structural abnormalities in the brain and unexplained lactic acidosis. When lactic acid (also called lactate) and pyruvate in the blood do not seem to be significantly high, an important clue to the diagnosis may be high concentrations of lactate and/or pyruvate in the cerebrospinal fluid (the fluid that surrounds the brain and spinal cord). Additionally, magnetic resonance spectroscopy (MRS) of the brain may show concentrations of lactate in the central nervous system. Analysis of serum and urine amino acids usually shows hyperalaninemia (high levels of the amino acid alanine). When lactic acidosis is present, other disorders involving pyruvate abnormalities are part of the differential diagnosis. However, in all of these conditions,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.704, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006572-2` (score=0.5994) - What are the signs and symptoms of 22q13.3 deletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 22q13.3 deletion syndrome. If the information is available, the table be...
2. `medquad_ans_0006562-2` (score=0.5827) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb a...
3. `medquad_ans_0006555-2` (score=0.5760) - What are the signs and symptoms of 15q13.3 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 15q13.3 microdeletion syndrome. If the information is available, th...
4. `medquad_ans_0006568-1` (score=0.5717) - What are the signs and symptoms of 20p12.3 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 20p12.3 microdeletion syndrome. If the information is available, th...
5. `medquad_ans_0006607-2` (score=0.5615) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...

**Answer**:

The provided context does not contain any information about the symptoms of 17q23.1q23.2 microdeletion syndrome. [medquad_ans_0006572-2] discusses the signs and symptoms of 22q13.3 deletion syndrome, not 17q23.1q23.2 microdeletion syndrome. Therefore, a response cannot be generated based on the given context.

**Supported citations**: `medquad_ans_0006572-2`
**Unsupported citations**: _none_

### Example 186 (`0003860-3`)

**Question**: What causes MECP2 duplication syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0003860-3` (score=0.5252) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
2. `medquad_ans_0000635-3` (score=0.5137) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
3. `medquad_ans_0003860-4` (score=0.5047) - Is MECP2 duplication syndrome inherited? MECP2 duplication syndrome is inherited in an X-linked manner. A condition is considered X-linked if the genetic change that causes the condition is located on the X chromosome...
4. `medquad_ans_0003860-2` (score=0.4868) - What are the signs and symptoms of MECP2 duplication syndrome? MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Infa...
5. `medquad_ans_0000635-4` (score=0.4867) - MECP2 duplication syndrome is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome...

**Answer**:

MECP2 duplication syndrome is caused by a duplication of genetic material located on the long (q) arm of the X chromosome, which results in an extra copy of the MECP2 gene in each cell. This duplication leads to the production of excess MeCP2 protein, which is unable to properly regulate the expression of other genes in the brain. The misregulation of gene expression causes abnormal neuronal function and irregular brain activity, resulting in the signs and symptoms of the syndrome [medquad_ans_0003860-3]. The size of the duplication can vary, ranging from 100,000 to 900,000 base pairs (100 to 900 kilobases), but the size does not appear to affect the severity of the condition [medquad_ans_0000635-3]. The MECP2 gene encodes a protein critical for normal

**Supported citations**: `medquad_ans_0003860-3`, `medquad_ans_0000635-3`
**Unsupported citations**: _none_

### Example 187 (`0004633-2`)

**Question**: What are the symptoms of Osteopetrosis autosomal dominant type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.532, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004633-2` (score=0.6037) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 2. If the informatio...
2. `medquad_ans_0004632-2` (score=0.5865) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 1. If the informatio...
3. `medquad_ans_0004640-2` (score=0.5707) - What are the signs and symptoms of Osteopetrosis autosomal recessive 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 7. If the information is ava...
4. `medquad_ans_0004637-2` (score=0.5685) - What are the signs and symptoms of Osteopetrosis autosomal recessive 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 4. If the information is ava...
5. `medquad_ans_0004638-2` (score=0.5667) - What are the signs and symptoms of Osteopetrosis autosomal recessive 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 5. If the information is ava...

**Answer**:

The symptoms of Osteopetrosis autosomal dominant type 2 include abnormality of epiphysis morphology, abnormality of the metacarpal bones, abnormality of the metaphyses, aseptic necrosis, bone pain, facial palsy, frontal bossing, joint dislocation, macrocephaly, osteoarthritis, osteomyelitis, recurrent fractures, short distal phalanx of finger, anemia, genu valgum, optic atrophy, short stature, visual impairment, abnormality of leukocytes, carious teeth, hearing impairment, hydrocephalus, hypocalcemia, bone marrow hypocellularity, abnormality of pelvic girdle bone morphology, abnormality of the vertebral endplates, generalized osteosclerosis, hip osteoarthritis, juvenile onset, mandibular osteomyelitis, and fractures of the long bones [medquad_ans_0004633

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 188 (`0003459-4`)

**Question**: What are the treatments for Lambert Eaton myasthenic syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.963, answer_relevance=0.578

**Top docs**:

1. `medquad_ans_0003459-4` (score=0.4900) - How might Lambert-Eaton myasthenic syndrome be treated? Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, ...
2. `medquad_ans_0000084-2__dup1` (score=0.4717) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
3. `medquad_ans_0000209-3` (score=0.4662) - With treatment, muscle weakness may improve or be reversed.
4. `medquad_ans_0000175-4__dup1` (score=0.4644) - The NINDS supports research on neuromuscular disorders such as LEMS with the ultimate goal of finding ways to treat, prevent, and cure them.
5. `medquad_ans_0000201-4__dup1` (score=0.4642) - The NINDS supports a broad range of research on neuromuscular disorders with the goal of finding ways to prevent, treat, and, ultimately, cure them.

**Answer**:

Treatments for Lambert-Eaton myasthenic syndrome (LEMS) may include anticholinesterase agents (e.g., pyridostigmine), guanidine hydrochloride, plasmapheresis, IV immunoglobulins, steroids (e.g., prednisone), azathioprine or cyclosporine, and/or 3,4-diaminopyridine. While some evidence suggests that 3,4-diaminopyridine or IV immunoglobulin can improve muscle strength and nerve-to-muscle communication, the degree of benefit still needs to be determined [medquad_ans_0003459-4].

**Supported citations**: `medquad_ans_0003459-4`
**Unsupported citations**: _none_

### Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.944, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000445-2__dup1` (score=0.6083) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes...
2. `medquad_ans_0000930-2__dup1` (score=0.5359) - What are the signs and symptoms of Carbamoyl phosphate synthetase 1 deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Carbamoyl phosphate synthetase 1 deficiency. If the in...
3. `medquad_ans_0002601-2` (score=0.5144) - What are the signs and symptoms of Glutamate formiminotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Glutamate formiminotransferase deficiency. If the inform...
4. `medquad_ans_0005318-1` (score=0.5135) - What are the signs and symptoms of Reticular dysgenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Reticular dysgenesis. If the information is available, the table below includ...
5. `medquad_ans_0002606-2` (score=0.5133) - What are the signs and symptoms of Glutathione synthetase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Glutathione synthetase deficiency. If the information is availab...

**Answer**:

The symptoms of Arginase deficiency include aminoaciduria, behavioral abnormality, cognitive impairment, neurological speech impairment, EEG abnormality, hemiplegia/hemiparesis, hyperammonemia, seizures, anorexia, hyperactivity, intellectual disability, irritability, oroticaciduria, postnatal growth retardation, progressive spastic quadriplegia, and vomiting. These symptoms occur in approximately 90% of patients for aminoaciduria, behavioral abnormality, cognitive impairment, neurological speech impairment, and 50% for EEG abnormality, hemiplegia/hemiparesis, hyperammonemia, and seizures [medquad_ans_0000445-2__dup1].

**Supported citations**: `medquad_ans_0000445-2__dup1`
**Unsupported citations**: _none_

### Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Scores**: context_relevance=0.778, groundedness=0.965, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0001020-5` (score=0.5534) - These resources address the diagnosis or management of von Hippel-Lindau syndrome: - Brigham and Women's Hospital - Gene Review: Gene Review: Von Hippel-Lindau Syndrome - Genetic Testing Registry: Von Hippel-Lindau sy...
2. `medquad_ans_0006376-6` (score=0.5516) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but ar...
3. `medquad_ans_0000022-2__dup1` (score=0.5508) - Treatment for VHL varies according to the location and size of the tumor. In general, the objective of treatment is to treat the tumors before they grow to a size large enough to cause permanent problems by putting pr...
4. `medquad_ans_0000343-5` (score=0.5013) - These resources address the diagnosis or management of familial cylindromatosis: - Genetic Testing Registry: Cylindromatosis, familial These resources from MedlinePlus offer information about the diagnosis and managem...
5. `medquad_ans_0000465-5` (score=0.4967) - These resources address the diagnosis or management of HLRCC: - Gene Review: Gene Review: Hereditary Leiomyomatosis and Renal Cell Cancer - Genetic Testing Registry: Hereditary leiomyomatosis and renal cell cancer - M...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. The goal is to treat tumors when they cause symptoms but are still small enough to avoid permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases, particularly for certain tumors that are treated with focused high-dose irradiation. All individuals with VHL disease should be carefully followed by a physician or medical team familiar with the disorder [medquad_ans_0006376-6].

**Supported citations**: `medquad_ans_0006376-6`
**Unsupported citations**: _none_

### Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Scores**: context_relevance=1.000, groundedness=0.852, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0004453-2` (score=0.5163) - What are the signs and symptoms of Noonan syndrome 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome 3. If the information is available, the table below includes how...
2. `medquad_ans_0004451-2` (score=0.5040) - What are the signs and symptoms of Noonan syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome 1. If the information is available, the table below includes how...
3. `medquad_ans_0004375-1` (score=0.4987) - What are the signs and symptoms of Neurofibromatosis-Noonan syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Neurofibromatosis-Noonan syndrome. If the information is availab...
4. `medquad_ans_0004456-2` (score=0.4972) - What are the signs and symptoms of Noonan syndrome 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome 6. If the information is available, the table below includes how...
5. `medquad_ans_0000738-2` (score=0.4940) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.

**Answer**:

Noonan syndrome 3 is a genetic disorder characterized by a range of physical and developmental signs and symptoms. According to the Human Phenotype Ontology, the signs and symptoms of Noonan syndrome 3 include anteverted nares, atrial septal aneurysm, frontal bossing, hypertelorism, low-set ears, pulmonic stenosis, sagittal craniosynostosis, short nose, short stature, ventricular septal defect, and webbed neck. These features are listed with approximate frequencies based on studies, though specific percentages or fractions are not provided in the context. The condition is associated with autosomal dominant inheritance [medquad_ans_0004453-2].

**Supported citations**: `medquad_ans_0004453-2`
**Unsupported citations**: _none_

### Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Scores**: context_relevance=1.000, groundedness=0.941, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001079-1__dup1` (score=0.6177) - What are the signs and symptoms of Cervical ribs, Sprengel anomaly, anal atresia, and urethral obstruction? The Human Phenotype Ontology provides the following list of signs and symptoms for Cervical ribs, Sprengel an...
2. `medquad_ans_0005865-2` (score=0.6070) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may...
3. `medquad_ans_0002966-1` (score=0.5862) - What are the signs and symptoms of Hydrocephalus, costovertebral dysplasia, and Sprengel anomaly? The Human Phenotype Ontology provides the following list of signs and symptoms for Hydrocephalus, costovertebral dyspla...
4. `medquad_ans_0002328-2` (score=0.5751) - What are the signs and symptoms of Femoral facial syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Femoral facial syndrome. If the information is available, the table below ...
5. `medquad_ans_0003335-2` (score=0.5713) - What are the signs and symptoms of Juvenile-onset dystonia? The Human Phenotype Ontology provides the following list of signs and symptoms for Juvenile-onset dystonia. If the information is available, the table below ...

**Answer**:

The symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additionally, some individuals may have bone, cartilage, or fibrous tissue between the shoulder blade and the adjacent spinal bones (vertebrae). Other associated features can include scoliosis, Klippel-Feil syndrome, limb length discrepancy, an underdeveloped backbone (hemivertebrae), missing, fused, or extra ribs (cervical ribs), and abnormalities of the collarbone [medquad_ans_0005865-2].

**Supported citations**: `medquad_ans_0005865-2`
**Unsupported citations**: _none_

### Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.934, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000882-3` (score=0.4966) - Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene. This gene provides instructions for making a protein whose function is not well understood, although it is involved in chemical signaling p...
2. `medquad_ans_0000979-3` (score=0.4540) - Mutations in the RBM8A gene cause TAR syndrome. The RBM8A gene provides instructions for making a protein called RNA-binding motif protein 8A. This protein is believed to be involved in several important cellular func...
3. `medquad_ans_0000174-3` (score=0.4472) - Mutations in the CHD7 gene cause more than half of all cases of CHARGE syndrome. The CHD7 gene provides instructions for making a protein that most likely regulates gene activity (expression) by a process known as chr...
4. `medquad_ans_0005415-1` (score=0.4468) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
5. `medquad_ans_0000923-3` (score=0.4424) - Mutations in the NSD1 gene are the primary cause of Sotos syndrome, accounting for up to 90 percent of cases. Other genetic causes of this condition have not been identified. The NSD1 gene provides instructions for ma...

**Answer**:

Genetic changes related to Robinow syndrome include mutations in the ROR2 gene, which cause autosomal recessive Robinow syndrome by preventing the production of a functional ROR2 protein and disrupting developmental signaling pathways essential for the formation of the skeleton, heart, and genitals [medquad_ans_0000882-3]. Autosomal dominant Robinow syndrome can result from mutations in the WNT5A or DVL1 gene, with the osteosclerotic form specifically linked to DVL1 gene mutations. These mutations alter the production or function of their respective proteins, impairing critical chemical signaling during early development [medquad_ans_0000882-3]. In some individuals, no mutation is identified in the ROR2, WNT5A, or DVL1 genes, and the cause remains unknown [medquad_ans_0000882-3].

**Supported citations**: `medquad_ans_0000882-3`
**Unsupported citations**: _none_

### Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Scores**: context_relevance=0.909, groundedness=0.773, answer_relevance=0.836

**Top docs**:

1. `medquad_ans_0000180-1__dup3` (score=0.4721) - Kidney failure means your kidneys no longer work well enough to do their job. You need treatment to replace the work your damaged kidneys have stopped doing. The treatments for kidney failure are - hemodialysis - peri...
2. `medquad_ans_0000155-10` (score=0.4673) - Conservative management for kidney failure is the choice to say no to or stop dialysis treatments. For many people, dialysis not only extends life, it also improves the quality of life. For others who have serious con...
3. `medquad_ans_0000179-2__dup2` (score=0.4644) - Kidney failure means your kidneys no longer filter your blood well enough to keep you healthy. Failing kidneys do a poor job of removing wastes and extra fluid from your blood. Wastes and extra fluid begin to build up...
4. `medquad_ans_0000155-5__dup1` (score=0.4637) - You have three treatment options to choose from to filter your blood. A fourth option offers care without replacing the work of the kidneys. None of these treatments helps the kidneys get better. However, they all can...
5. `medquad_ans_0000179-13` (score=0.4632) - - Kidney failure means your kidneys no longer filter your blood well enough to keep you healthy. - The treatments for kidney failure are - hemodialysis - peritoneal dialysis - a kidney transplant - conservative manage...

**Answer**:

The provided context does not contain information about "Kidney Failure: Eat Right to Feel Right on Hemodialysis." Therefore, it is not possible to answer the question based on the given text. [medquad_ans_0000180-1__dup3] discusses dietary considerations for kidney failure, noting that following a special diet helps keep wastes from building up in the blood, but it does not specifically address "Eat Right to Feel Right on Hemodialysis."

**Supported citations**: `medquad_ans_0000180-1__dup3`
**Unsupported citations**: _none_

### Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Scores**: context_relevance=0.556, groundedness=0.971, answer_relevance=0.556

**Top docs**:

1. `medquad_ans_0000677-2` (score=0.6515) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
2. `medquad_ans_0000359-2` (score=0.5741) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
3. `medquad_ans_0000679-2` (score=0.5728) - MPS II occurs in approximately 1 in 100,000 to 1 in 170,000 males.
4. `medquad_ans_0000989-2` (score=0.5717) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000675-2` (score=0.5709) - Mucolipidosis III alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.

**Answer**:

Mucolipidosis type IV is estimated to occur in 1 in 40,000 people [medquad_ans_0000677-2]. About 70 percent of affected individuals have Ashkenazi Jewish ancestry [medquad_ans_0000677-2].

**Supported citations**: `medquad_ans_0000677-2`
**Unsupported citations**: _none_

### Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Scores**: context_relevance=1.000, groundedness=0.994, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.5521) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chi...
2. `medquad_ans_0000012-1__dup4` (score=0.5008) - Adhesions are bands of scar-like tissue. Normally, internal tissues and organs have slippery surfaces so they can shift easily as the body moves. Adhesions cause tissues and organs to stick together. They might connec...
3. `medquad_ans_0000712-1__dup1` (score=0.4981) - Piercings and tattoos are body decorations that go back to ancient times. Body piercing involves making a hole in the skin so that you can insert jewelry. This is often in the earlobe, but can be in other parts of the...
4. `medquad_ans_0000193-1__dup4` (score=0.4942) - Cirrhosis\* is scarring of the liver. Scar tissue forms because of injury or long-term disease. Scar tissue replaces healthy liver tissue and blocks the normal flow of blood through the liver. A healthy liver - makes p...
5. `medquad_ans_0000204-1__dup2` (score=0.4933) - Cirrhosis is scarring of the liver. Scar tissue forms because of injury or long-term disease. Scar tissue cannot do what healthy liver tissue does - make protein, help fight infections, clean the blood, help digest fo...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams. [medquad_ans_0000795-1__dup1]

**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

### Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.953, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0001045-4` (score=0.4681) - X-linked adrenoleukodystrophy is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each...
2. `medquad_ans_0001045-1` (score=0.4226) - X-linked adrenoleukodystrophy is a genetic disorder that occurs primarily in males. It mainly affects the nervous system and the adrenal glands, which are small glands located on top of each kidney. In this disorder, ...
3. `medquad_ans_0001058-4` (score=0.4149) - This condition is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each cell. In males...
4. `medquad_ans_0001045-3` (score=0.4118) - Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. The ABCD1 gene provides instructions for producing the adrenoleukodystrophy protein (ALDP), which is involved in transporting certain fat molecules call...
5. `medquad_ans_0000948-4` (score=0.4118) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The genetic changes related to X-linked adrenoleukodystrophy are mutations in the ABCD1 gene, which is located on the X chromosome. In males, who have only one X chromosome, one altered copy of the ABCD1 gene in each cell is sufficient to cause the disorder. Females, who have two copies of the X chromosome, usually do not develop symptoms with one altered copy of the ABCD1 gene, although some may experience health problems associated with the disorder [medquad_ans_0001045-4].

**Supported citations**: `medquad_ans_0001045-4`
**Unsupported citations**: _none_

### Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Scores**: context_relevance=0.375, groundedness=0.912, answer_relevance=0.375

**Top docs**:

1. `medquad_ans_0000988-2` (score=0.5958) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of Europe...
2. `medquad_ans_0000650-2` (score=0.5717) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
3. `medquad_ans_0000989-2` (score=0.5667) - This condition affects an estimated 1 in 50,000 people.
4. `medquad_ans_0000359-2` (score=0.5581) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
5. `medquad_ans_0000484-2` (score=0.5394) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.

**Answer**:

Transthyretin amyloidosis affects an estimated 1 in 50,000 to 1 in 100,000 people [medquad_ans_0000650-2]. In northern Portugal, the incidence is thought to be one in 538 people [medquad_ans_0000988-2]. In some areas of West Africa, it affects approximately 5 percent of the population, and among African Americans, it affects between 3 percent and 3.9 percent [medquad_ans_0000988-2]. In certain areas of Quebec, Canada, it is much more common, affecting about 1 per million people worldwide [medquad_ans_0000359-2]. The exact incidence is unknown, but estimates range from 1 in 87,000 to 1 in

**Supported citations**: `medquad_ans_0000650-2`, `medquad_ans_0000988-2`, `medquad_ans_0000359-2`
**Unsupported citations**: _none_

### Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Scores**: context_relevance=1.000, groundedness=0.765, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.6013) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the informatio...
2. `medquad_ans_0006174-1` (score=0.5443) - What are the signs and symptoms of Tricho-dento-osseous syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tricho-dento-osseous syndrome. If the information is available, the ...
3. `medquad_ans_0006175-1` (score=0.5421) - What are the signs and symptoms of Tricho-dento-osseous syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Tricho-dento-osseous syndrome 1. If the information is available, ...
4. `medquad_ans_0000096-1__dup1` (score=0.5295) - What are the signs and symptoms of Acroosteolysis dominant type? The Human Phenotype Ontology provides the following list of signs and symptoms for Acroosteolysis dominant type. If the information is available, the ta...
5. `medquad_ans_0003744-1` (score=0.5263) - What are the signs and symptoms of Madokoro Ohdo Sonoda syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Madokoro Ohdo Sonoda syndrome. If the information is available, the ...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include abnormalities related to dental and craniofacial features, as described by the Human Phenotype Ontology. Specific symptoms may include dental anomalies such as abnormal tooth development or structural defects, though the exact list of symptoms and their frequencies is not fully detailed in the provided context. The frequency of these symptoms is typically estimated as a percentage or fraction based on studies, with data often derived from Orphanet, a European rare disease database. For precise symptom details and their occurrence rates, further reference to the Human Phenotype Ontology or related medical databases is recommended [medquad_ans_0001780-1].

**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

### Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Scores**: context_relevance=0.375, groundedness=0.931, answer_relevance=0.275

**Top docs**:

1. `medquad_ans_0000359-2` (score=0.5768) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
2. `medquad_ans_0000782-2` (score=0.5768) - Periventricular heterotopia is a rare condition. Its incidence is unknown.
3. `medquad_ans_0000934-2` (score=0.5565) - This condition affects fewer than 1 in 150,000 males and is very rare in females.
4. `medquad_ans_0000650-2` (score=0.5557) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
5. `medquad_ans_0001092-2` (score=0.5528) - This condition is detected in an estimated 1 in 36,000 newborns worldwide.

**Answer**:

The incidence of periventricular heterotopia is unknown [medquad_ans_0000782-2].

**Supported citations**: `medquad_ans_0000782-2`
**Unsupported citations**: _none_

### Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Scores**: context_relevance=0.818, groundedness=0.697, answer_relevance=0.745

**Top docs**:

1. `medquad_ans_0000169-4` (score=0.4925) - NINDS supports a broad range of clinical and basic research on diseases causing sleep disorders in an effort to clarify the mechanisms of these conditions and to develop better treatments for them.
2. `medquad_ans_0000179-4__dup1` (score=0.4898) - The NINDS conducts and supports a broad program of basic and clinical research on epilepsy including Lennox-Gastaut syndrome. These studies are aimed at finding the causes of these disorders, improving the diagnosis, ...
3. `medquad_ans_0000927-1__dup2` (score=0.4822) - Summary : It seems to happen almost every day - you hear about the results of a new medical research study. Sometimes the results of one study seem to disagree with the results of another study. It's important to be c...
4. `medquad_ans_0000113-4__dup1` (score=0.4762) - The NINDS conducts and supports an extensive research program on seizures and seizure-related disorders. Much of this research is aimed at increasing scientific understanding of these disorders and finding ways to pre...
5. `medquad_ans_0000195-4__dup1` (score=0.4733) - NINDS is the leading supporter of research on stroke and TIA in the U.S. and sponsors studies ranging from clinical trials to investigations of basic biological mechanisms as well as studies with animals.

**Answer**:

The provided context does not mention research or clinical trials specifically for Landau-Kleffner Syndrome. Therefore, based on the given information, it is not possible to answer the question. [medquad_ans_0000169-4][medquad_ans_0000179-4__dup1][medquad_ans_0000113-4\_\_dup1][medquad_ans_0000195-4__dup1]

**Supported citations**: `medquad_ans_0000169-4`, `medquad_ans_0000179-4__dup1`, `medquad_ans_0000113-4__dup1`, `medquad_ans_0000195-4__dup1`
**Unsupported citations**: _none_

### Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Scores**: context_relevance=0.800, groundedness=0.491, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000710-1` (score=0.4728) - Myotonic dystrophy is part of a group of inherited disorders called muscular dystrophies. It is the most common form of muscular dystrophy that begins in adulthood. Myotonic dystrophy is characterized by progressive m...
2. `medquad_ans_0000705-1` (score=0.4713) - Myofibrillar myopathy is part of a group of disorders called muscular dystrophies that affect muscle function and cause weakness. Myofibrillar myopathy primarily affects skeletal muscles, which are muscles that the bo...
3. `medquad_ans_0000334-1` (score=0.4709) - Facioscapulohumeral muscular dystrophy is a disorder characterized by muscle weakness and wasting (atrophy). This condition gets its name from the muscles that are affected most often: those of the face (facio-), arou...
4. `medquad_ans_0000205-4__dup1` (score=0.4656) - The NINDS supports a broad program of research studies on MD. The goals of these studies are to understand MD and to develop techniques to diagnose, treat, prevent, and ultimately cure the disorder. The NINDS is a mem...
5. `medquad_ans_0000205-1__dup2` (score=0.4628) - The muscular dystrophies (MD) are a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or ...

**Answer**:

Muscular dystrophy is a group of inherited disorders characterized by progressive muscle wasting and weakness. These disorders affect muscle function and can involve skeletal muscles, and in some cases, the heart muscle. Myotonic dystrophy, the most common form that begins in adulthood, is characterized by prolonged muscle contractions (myotonia), difficulty relaxing muscles after use, slurred speech, temporary jaw locking, cataracts, and cardiac conduction defects. It is caused by mutations in different genes, with two major types—type 1 and type 2—showing overlapping signs and symptoms, though type 2 is generally milder than type 1. Type 1 primarily affects the lower legs, hands, neck, and face, while type 2 mainly involves the neck, shoulders, elbows, and hips. A congenital form of myotonic dystrophy appears at birth and includes features such as weak muscle tone, clubfoot, breathing problems, delayed

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.342, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.5070) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that t...
2. `medquad_ans_0003728-1` (score=0.4629) - Macrodactyly of the hand is a rare condition in which a person's fingers are abnormally large due to the overgrowth of the underlying bone and soft tissue. This condition is congenital, meaning that babies are born wi...
3. `medquad_ans_0005582-1` (score=0.4507) - What are the signs and symptoms of Short rib-polydactyly syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Short rib-polydactyly syndrome type 3. If the information is...
4. `medquad_ans_0005020-1` (score=0.4462) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the tabl...
5. `medquad_ans_0005022-1` (score=0.4407) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the tabl...

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding with no other physical anomalies or intellectual impairment, or it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits, with surgical management depending greatly on the complexity of the deformity [1] [2].

**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

### Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Scores**: context_relevance=1.000, groundedness=0.862, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002519-2` (score=0.6007) - What are the signs and symptoms of Geniospasm? The Human Phenotype Ontology provides the following list of signs and symptoms for Geniospasm. If the information is available, the table below includes how often the sym...
2. `medquad_ans_0001057-1__dup1` (score=0.5865) - What are the signs and symptoms of Cerebral gigantism jaw cysts? The Human Phenotype Ontology provides the following list of signs and symptoms for Cerebral gigantism jaw cysts. If the information is available, the ta...
3. `medquad_ans_0000096-1__dup1` (score=0.5820) - What are the signs and symptoms of Acroosteolysis dominant type? The Human Phenotype Ontology provides the following list of signs and symptoms for Acroosteolysis dominant type. If the information is available, the ta...
4. `medquad_ans_0002524-2` (score=0.5775) - What are the signs and symptoms of Genoa syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Genoa syndrome. If the information is available, the table below includes how often...
5. `medquad_ans_0000085-1__dup1` (score=0.5683) - What are the signs and symptoms of Acromegaloid facial appearance syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Acromegaloid facial appearance syndrome. If the informatio...

**Answer**:

The symptoms of Geniospasm include chin myoclonus. This information is derived from the Human Phenotype Ontology, which lists chin myoclonus as a sign or symptom associated with Geniospasm [medquad_ans_0002519-2].

**Supported citations**: `medquad_ans_0002519-2`
**Unsupported citations**: _none_

### Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=0.800, groundedness=0.941, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000512-4` (score=0.4497) - Most cases of idiopathic pulmonary fibrosis are sporadic; they occur in people with no history of the disorder in their family. Familial pulmonary fibrosis appears to have an autosomal dominant pattern of inheritance....
2. `medquad_ans_0000512-3` (score=0.4443) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of devel...
3. `medquad_ans_0000262-3` (score=0.4180) - Mutations in the CFTR gene cause cystic fibrosis. The CFTR gene provides instructions for making a channel that transports negatively charged particles called chloride ions into and out of cells. Chloride is a compone...
4. `medquad_ans_0000825-3` (score=0.3989) - Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition. The FLCN gene provides instructions for making a protein called follicu...
5. `medquad_ans_0000849-3` (score=0.3980) - Mutations in the BMPR2 gene are the most common genetic cause of pulmonary arterial hypertension. This gene plays a role in regulating the number of cells in certain tissues. Researchers suggest that a mutation in thi...

**Answer**:

Genetic changes related to idiopathic pulmonary fibrosis include mutations in genes such as TERC and TERT, which have been found in about 15 percent of familial pulmonary fibrosis cases and a smaller percentage of sporadic idiopathic pulmonary fibrosis cases. These genes provide instructions for making components of an enzyme called telomerase, which maintains telomeres at the ends of chromosomes. While the exact mechanism by which defects in telomerase lead to lung damage is not well understood, these genetic changes are associated with increased risk for the disease [medquad_ans_0000512-3]. Additionally, several other genetic changes have been suggested as risk factors, though most account for only a small proportion of cases [medquad_ans_0000512-3].

**Supported citations**: `medquad_ans_0000512-3`
**Unsupported citations**: _none_

### Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.6199) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000974-4` (score=0.6185) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000239-4` (score=0.6185) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0001010-4` (score=0.6185) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000960-4` (score=0.6185) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Potocki-Shaffer syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Potocki-Shaffer syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Scores**: context_relevance=1.000, groundedness=0.400, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000925-1__dup1` (score=0.6151) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
2. `medquad_ans_0000691-1__dup1` (score=0.6093) - If you're a parent, you get plenty of suggestions on how to raise your child. From experts to other parents, people are always ready to offer advice. Parenting tips, parents' survival guides, dos, don'ts, shoulds and ...
3. `medquad_ans_0000897-1__dup1` (score=0.6003) - Summary : Most young children get sick. It is hard for parents to know what is serious. You can learn what the common warning signs are. In the end, trust your intuition. If you are worried about your toddler, call yo...
4. `medquad_ans_0000076-1__dup3` (score=0.5973) - Summary : There are many new responsibilities when you have a baby. One of them is to make sure they get the checkups that they need. Well-baby exams are important in making sure that your baby is growing and developi...
5. `medquad_ans_0000869-1__dup1` (score=0.5851) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can brin...

**Answer**:

No, there is no information about child care in the provided context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Scores**: context_relevance=0.500, groundedness=0.850, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000046-2` (score=0.5900) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
2. `medquad_ans_0000934-2` (score=0.5628) - This condition affects fewer than 1 in 150,000 males and is very rare in females.
3. `medquad_ans_0000650-2` (score=0.5580) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
4. `medquad_ans_0000359-2` (score=0.5567) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
5. `medquad_ans_0000989-2` (score=0.5503) - This condition affects an estimated 1 in 50,000 people.

**Answer**:

Alternating hemiplegia of childhood affects approximately 1 in 1 million people [medquad_ans_0000046-2]. It also affects about 1 per million people worldwide [medquad_ans_0000359-2]. Additionally, it occurs in an estimated 1 in 50,000 to 100,000 people [medquad_ans_0000650-2], and affects fewer than 1 in 150,000 males, with very rare occurrence in females [medquad_ans_0000934-2]. It affects an estimated 1 in 50,000 people [medquad_ans_0000989-2].

**Supported citations**: `medquad_ans_0000046-2`, `medquad_ans_0000359-2`, `medquad_ans_0000650-2`, `medquad_ans_0000934-2`, `medquad_ans_0000989-2`
**Unsupported citations**: _none_

### Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003687-2` (score=0.6341) - What are the signs and symptoms of Lupus? You can read about the signs and symptoms of lupus from MedlinePlus and the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS). The Human Phenotype ...
2. `medquad_ans_0000535-2` (score=0.5744) - What are the signs and symptoms of Autoimmune hemolytic anemia? The Human Phenotype Ontology provides the following list of signs and symptoms for Autoimmune hemolytic anemia. If the information is available, the tabl...
3. `medquad_ans_0005471-2` (score=0.5716) - What are the signs and symptoms of Sarcoidosis? Many people who have sarcoidosis don't have symptoms. Others may feel like they are coming down with the flu or a respiratory infection. While almost any body part or sy...
4. `medquad_ans_0003615-2` (score=0.5622) - What are the signs and symptoms of Limited cutaneous systemic sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Limited cutaneous systemic sclerosis. If the information is a...
5. `medquad_ans_0003690-2` (score=0.5569) - What are the signs and symptoms of Lyme disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Lyme disease. If the information is available, the table below includes how often the...

**Answer**:

The symptoms of Lupus include abdominal pain, abnormality of temperature regulation, abnormality of the heart valves, abnormality of the pericardium, alopecia, arthralgia, arthritis, autoimmunity, chest pain, cutaneous photosensitivity, skin rash, thrombocytopenia, thrombophlebitis, abnormal pyramidal signs, abnormal tendon morphology, abnormality of the autonomic nervous system, abnormality of the endocardium, abnormality of the pleura, anorexia, arterial thrombosis, aseptic leukocyturia, bone marrow hypocellularity, conjunctival telangiectasia, cranial nerve paralysis, cutis marmorata, dry skin, eczema, edema of the lower limbs, glomerulopathy, hallucinations, hematuria, hepatomegaly, hyperkeratosis, hypoproteinemia, increased antibody level in blood

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.840, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001398-2` (score=0.5141) - What are the signs and symptoms of Cockayne syndrome type III? The Human Phenotype Ontology provides the following list of signs and symptoms for Cockayne syndrome type III. If the information is available, the table ...
2. `medquad_ans_0001395-2` (score=0.5117) - What are the signs and symptoms of Cockayne syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Cockayne syndrome. If the information is available, the table below includes how...
3. `medquad_ans_0001397-2` (score=0.5066) - What are the signs and symptoms of Cockayne syndrome type II? The Human Phenotype Ontology provides the following list of signs and symptoms for Cockayne syndrome type II. If the information is available, the table be...
4. `medquad_ans_0000194-3` (score=0.5055) - Cockayne syndrome can result from mutations in either the ERCC6 gene (also known as the CSB gene) or the ERCC8 gene (also known as the CSA gene). These genes provide instructions for making proteins that are involved ...
5. `medquad_ans_0001395-1` (score=0.4964) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...

**Answer**:

Cockayne syndrome is a rare genetic disorder characterized by a range of signs and symptoms, including abnormal blistering of the skin, abnormality of the nose, carious teeth, cognitive impairment, cutaneous photosensitivity, deeply set eyes, hyperreflexia, hypertonia, and incoordination. These features are listed in the Human Phenotype Ontology (HPO) based on data from Orphanet, a European rare disease database. The frequency of these symptoms is often reported as a percentage of patients, such as 90% for several of the listed signs and symptoms [medquad_ans_0001395-2].

Note: The provided context does not specify the exact type (e.g., type I, II, III) of Cockayne syndrome unless otherwise indicated. The information given pertains to Cockayne syndrome in general, not specifically to type III [medquad_ans_000139

**Supported citations**: `medquad_ans_0001395-2`
**Unsupported citations**: _none_

### Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Scores**: context_relevance=0.667, groundedness=0.890, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000413-1__dup1` (score=0.5017) - If you have gum disease, you're not alone. Many U.S. adults currently have some form of the disease. It ranges from simple gum inflammation, called gingivitis, to serious damage to the tissue and bone supporting the t...
2. `medquad_ans_0000029-16` (score=0.4941) - People are not often aware they have gum disease until it is advanced. Any of these symptoms may be a sign of a serious problem and should be checked by a dentist. - bad breath that won't go away - red or swollen gums...
3. `medquad_ans_0000029-8__dup1` (score=0.4882) - Gingivitis is inflammation of the gums. In gingivitis, the gums become red, swollen and can bleed easily. Gingivitis is a mild form of gum disease. It can usually be reversed with daily brushing and flossing, and regu...
4. `medquad_ans_0000029-3__dup2` (score=0.4866) - Symptoms Symptoms of gum disease may include: - bad breath that won't go away - red or swollen gums - tender or bleeding gums - painful chewing - loose teeth - sensitive teeth - receding gums or longer appearing teeth...
5. `medquad_ans_0000029-6__dup1` (score=0.4840) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead...

**Answer**:

Gum (periodontal) disease is a condition that ranges from mild inflammation of the gums, known as gingivitis, to serious damage involving the tissue and bone that support the teeth. Gingivitis is characterized by red, swollen gums that may bleed easily and is a reversible form of gum disease that can be managed with daily brushing and flossing and regular dental cleanings. If left untreated, gingivitis can progress to periodontitis, in which the gums pull away from the teeth and form infected pockets. This advanced stage leads to the destruction of the bones, gums, and connective tissue that support the teeth, potentially resulting in tooth loss [medquad_ans_0000413-1__dup1]. Symptoms may include bad breath, red or swollen gums, bleeding gums, painful chewing, loose teeth, sensitive teeth, and receding gums, though some cases may have no clear symptoms [medquad*ans*

**Supported citations**: `medquad_ans_0000413-1__dup1`
**Unsupported citations**: _none_

### Example 212 (`0000494-2`)

**Question**: How many people are affected by hyperkalemic periodic paralysis ?

**Scores**: context_relevance=0.444, groundedness=0.939, answer_relevance=0.444

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.6089) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0000505-2` (score=0.5718) - Although its exact prevalence is unknown, hypokalemic periodic paralysis is estimated to affect 1 in 100,000 people. Men tend to experience symptoms of this condition more often than women.
3. `medquad_ans_0000650-2` (score=0.5642) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
4. `medquad_ans_0000989-2` (score=0.5606) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000484-2` (score=0.5534) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.

**Answer**:

Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people [medquad_ans_0000494-2].

**Supported citations**: `medquad_ans_0000494-2`
**Unsupported citations**: _none_

### Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Scores**: context_relevance=1.000, groundedness=0.536, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0005212-2` (score=0.5882) - What are the signs and symptoms of Pyle disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyle disease. If the information is available, the table below includes how often the...
2. `medquad_ans_0005212-3` (score=0.5828) - Is Pyle disease inherited? Pyle disease in inherited in an autosomal recessive manner, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive conditio...
3. `medquad_ans_0005212-1` (score=0.5275) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end o...
4. `medquad_ans_0000121-12` (score=0.5199) - Eating, diet, and nutrition can play a role in causing, preventing, and treating some of the diseases and disorders of the digestive system that are affected by smoking, including heartburn and GERD, liver diseases, C...
5. `medquad_ans_0000664-1` (score=0.5176) - Mitochondrial neurogastrointestinal encephalopathy (MNGIE) disease is a condition that affects several parts of the body, particularly the digestive system and nervous system. The major features of MNGIE disease can a...

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end of the bone), widening of the ribs and clavicles (collarbones), platyspondyly (flattening of the bones of the spine), and cortical thinning. It is inherited in an autosomal recessive manner, meaning both copies of the gene in each cell have mutations, and the parents of an individual with the condition each carry one copy of the mutated gene. The condition is rare, with only about 30 cases reported in the literature. Signs and symptoms include genu valgum in 90% of patients, abnormalities of the pelvic girdle bone morphology, clavicle, elbow, ribs, craniofacial hyperostosis, mandibular prognathia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Scores**: context_relevance=0.800, groundedness=0.953, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000256-1__dup3` (score=0.5739) - Deep vein thrombosis, or DVT, is a blood clot that forms in a vein deep in the body. Most deep vein clots occur in the lower leg or thigh. If the vein swells, the condition is called thrombophlebitis. A deep vein thro...
2. `medquad_ans_0000051-4__dup2` (score=0.5608) - The risk factors for deep vein thrombosis (DVT) include: A history of DVT. Conditions or factors that make your blood thicker or more likely to clot than normal. Some inherited blood disorders (such as factor V Leiden...
3. `medquad_ans_0000104-1__dup4` (score=0.5517) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent b...
4. `medquad_ans_0000051-8` (score=0.5419) - You can take steps to prevent deep vein thrombosis (DVT) and pulmonary embolism (PE). If you're at risk for these conditions: See your doctor for regular checkups. Take all medicines as your doctor prescribes. Get out...
5. `medquad_ans_0000051-3__dup2` (score=0.5383) - Blood clots can form in your body's deep veins if: A vein's inner lining is damaged. Injuries caused by physical, chemical, or biological factors can damage the veins. Such factors include surgery, serious injuries, i...

**Answer**:

Deep vein thrombosis (DVT) is caused by conditions or factors that make the blood thicker or more likely to clot than normal. Risk factors include certain genetic disorders (such as factor V Leiden), hormone therapy or birth control pills, injury to a deep vein from surgery, broken bones, or trauma, slow blood flow due to lack of movement (e.g., after surgery, being ill and in bed for a long time, or prolonged travel), pregnancy and the first 6 weeks after giving birth, recent or ongoing cancer treatment, presence of a central venous catheter, older age (over 60), overweight or obesity, and smoking [medquad_ans_0000051-4__dup2]. Additionally, conditions such as atherosclerosis, diabetes, atrial fibrillation, metabolic syndrome, and certain medicines can contribute to excessive blood clotting, which may lead to DVT [medquad_ans_0

**Supported citations**: `medquad_ans_0000051-4__dup2`
**Unsupported citations**: _none_

### Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.508, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004079-4` (score=0.5009) - How might mitochondrial complex II deficiency be treated? Treatment options for complex II deficiency may be similar to those for other mitochondrial disorders in general.[8677] The United Mitochondrial Disease Founda...
2. `medquad_ans_0000661-5` (score=0.4962) - These resources address the diagnosis or management of mitochondrial complex III deficiency: - Gene Review: Gene Review: Mitochondrial Disorders Overview - Genetic Testing Registry: MITOCHONDRIAL COMPLEX III DEFICIENC...
3. `medquad_ans_0004081-2` (score=0.4961) - What are the signs and symptoms of Mitochondrial complex IV deficiency? There are currently 4 known forms of COX deficiency. The range and severity of signs and symptoms can vary widely from case to case. In one form,...
4. `medquad_ans_0000661-2` (score=0.4825) - The prevalence of mitochondrial complex III deficiency is unknown, although the condition is thought to be rare.
5. `medquad_ans_0001445-1` (score=0.4670) - Combined oxidative phosphorylation deficiency 16, also know as infantile hypertrophic cardiomyopathy, is characterized by decreased levels of mitochondrial complexes. The symptoms and signs described include an enlarg...

**Answer**:

Mitochondrial complex IV deficiency, also known as cytochrome c oxidase (COX) deficiency, includes four known forms. The range and severity of signs and symptoms vary widely among cases. One form, referred to as benign infantile mitochondrial myopathy type, may present with limited skeletal muscle symptoms and episodes of lactic acidosis, which can be life-threatening if untreated; however, individuals with this form may spontaneously recover within the first few years of life. The second form, infantile mitochondrial myopathy type, affects skeletal muscles and other tissues such as the heart, kidney, liver, brain, and/or connective tissue, with symptoms beginning within the first few weeks of life and including muscle weakness, heart problems, kidney dysfunction, failure to thrive, difficulties sucking, swallowing, and/or breathing, and/or hypotonia, along with episodes of lactic acidosis. The third form is considered a systemic condition known as Leigh's disease, which

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Scores**: context_relevance=1.000, groundedness=0.955, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000496-1__dup1` (score=0.5335) - What are the signs and symptoms of Ataxia with Oculomotor Apraxia Type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Ataxia with Oculomotor Apraxia Type 2. If the information is...
2. `medquad_ans_0000078-1` (score=0.5059) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...
3. `medquad_ans_0000078-2` (score=0.4975) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.
4. `medquad_ans_0005544-1` (score=0.4961) - What are the signs and symptoms of Sensory ataxic neuropathy, dysarthria, and ophthalmoparesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Sensory ataxic neuropathy, dysarthria,...
5. `medquad_ans_0005780-1` (score=0.4892) - What are the signs and symptoms of Spinocerebellar ataxia 34? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia 34. If the information is available, the table be...

**Answer**:

Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement, with the hallmark being difficulty coordinating movements (ataxia), often the first symptom. Most affected individuals also have oculomotor apraxia, which makes it difficult to move their eyes side-to-side, requiring them to turn their head to see things in their peripheral vision. There are multiple types of ataxia with oculomotor apraxia [medquad_ans_0000078-1].

**Supported citations**: `medquad_ans_0000078-1`
**Unsupported citations**: _none_

### Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Scores**: context_relevance=1.000, groundedness=0.441, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0001744-1` (score=0.5393) - What are the signs and symptoms of Deafness with labyrinthine aplasia microtia and microdontia (LAMM)? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness with labyrinthine apla...
2. `medquad_ans_0000248-1` (score=0.5156) - Craniofacial microsomia is a term used to describe a spectrum of abnormalities that primarily affect the development of the skull (cranium) and face before birth. Microsomia means abnormal smallness of body structures...
3. `medquad_ans_0000221-1` (score=0.5086) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that fo...
4. `medquad_ans_0004054-1` (score=0.4888) - What are the signs and symptoms of Microsomia hemifacial radial defects? The Human Phenotype Ontology provides the following list of signs and symptoms for Microsomia hemifacial radial defects. If the information is a...
5. `medquad_ans_0000221-5` (score=0.4734) - These resources address the diagnosis or management of LAMM syndrome: - Gene Review: Gene Review: Congenital Deafness with Labyrinthine Aplasia, Microtia, and Microdontia - Genetic Testing Registry: Deafness with laby...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (LAMM) is a condition characterized by profound sensorineural hearing impairment, aplasia of the inner ear, microtia (abnormal smallness or underdevelopment of the external ear), microdontia (abnormal smallness of the teeth), and other associated features such as abnormalities of the cranial nerves, abnormal nasal morphology, and facial asymmetry. It is inherited in an autosomal recessive pattern and is associated with a range of signs and symptoms including hypermetropia, strabismus, widened teeth, and preauricular skin tags. The Human Phenotype Ontology (HPO) has compiled data on the frequency of these features, with approximately 90% of patients showing microdontia and abnormalities of the cranial nerves, and 50% showing long face, pointed chin, and wide nasal bridge [med

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000441-4` (score=0.5582) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has a parent and other family ...
2. `medquad_ans_0000785-4` (score=0.5562) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000403-4` (score=0.5537) - This condition has an autosomal dominant inheritance pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. All reported cases have resulted from new mutations in the gene ...
4. `medquad_ans_0000948-4` (score=0.5534) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000342-4` (score=0.5528) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The provided context does not mention Apert syndrome or specify its inheritance pattern. Therefore, based on the given information, it is not possible to determine whether Apert syndrome is inherited. [No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Scores**: context_relevance=1.000, groundedness=0.538, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.4671) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organi...
2. `medquad_ans_0000341-1__dup3` (score=0.4665) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella bu...
3. `medquad_ans_0000341-4__dup2` (score=0.4407) - In the United States, Q fever outbreaks have resulted mainly from occupational exposure involving veterinarians, meat processing plant workers, sheep and dairy workers, livestock farmers, and researchers at facilities...
4. `medquad_ans_0000059-2__dup2` (score=0.4075) - Although the cause of psoriasis is not completely understood, scientists believe it is related to a problem with a type of blood cells called T cells. These cells normally travel through the bloodstream to help fight ...
5. `medquad_ans_0006243-2` (score=0.4070) - What are the symptoms of tularemia? The symptoms of tularemia usually appear 3 to 5 days after exposure to the bacteria, but can take as long as 14 days. Symptoms may include: Fever Chills Headache Diarrhea Muscle pai...

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air contaminated by dried placental material, birth fluids, and excreta of infected animals. Other rare modes of transmission include tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, individuals with acute Q fever develop a chronic form of the disease. Treatment of the acute form involves antibiotics, while treatment for the chronic form depends on the symptoms

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Scores**: context_relevance=1.000, groundedness=0.970, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000038-5__dup1` (score=0.4622) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
2. `medquad_ans_0001152-2` (score=0.4551) - How might chILD be treated? There is no single treatment for interstitial lung diseases in children. Different forms of chILD require different treatments and support depending on the condition. The goals of treatment...
3. `medquad_ans_0000038-9` (score=0.4430) - At this time, most types of childhood interstitial lung disease (chILD) can't be prevented. People who have a family history of inherited (genetic) interstitial lung disease may want to consider genetic counseling. A ...
4. `medquad_ans_0000038-7` (score=0.4242) - Doctors diagnose childhood interstitial lung disease (chILD) based on a child's medical and family histories and the results from tests and procedures. To diagnose chILD, doctors may first need to rule out other disea...
5. `medquad_ans_0000650-2` (score=0.4211) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Children who have a family history of interstitial lung disease or chILD are at risk. Other risk factors include having an inherited surfactant disorder or a family history of this disorder, problems with aspiration, immune system disorders, exposure to environmental lung irritants such as molds and chemicals, systemic or autoimmune diseases like collagen vascular disease or inflammatory bowel disease, undergoing cancer treatments such as radiation and chemotherapy, or having a bone marrow or lung transplant [medquad_ans_0000038-5__dup1].

**Supported citations**: `medquad_ans_0000038-5__dup1`
**Unsupported citations**: _none_

### Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.5183) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in lat...
2. `medquad_ans_0000404-1__dup1` (score=0.4309) - Giant cell arteritis is a disorder that causes inflammation of your arteries, usually in the scalp, neck, and arms. It narrows the arteries, which keeps blood from flowing well. Giant cell arteritis often occurs with ...
3. `medquad_ans_0000555-1` (score=0.4219) - Kawasaki disease is a sudden and time-limited (acute) illness that affects infants and young children. Affected children develop a prolonged fever lasting several days, a skin rash, and swollen lymph nodes in the neck...
4. `medquad_ans_0006001-2` (score=0.4110) - What are the signs and symptoms of Takayasu arteritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Takayasu arteritis. If the information is available, the table below includes h...
5. `medquad_ans_0005897-1` (score=0.4098) - What are the signs and symptoms of STING-associated vasculopathy with onset in infancy? The Human Phenotype Ontology provides the following list of signs and symptoms for STING-associated vasculopathy with onset in in...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels. It is generally diagnosed in late childhood or early adulthood and specifically affects the temporal arteries, located at the lower sides of the skull beneath the temple. Affected individuals often have no signs or symptoms aside from a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically with no family history. It is generally treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

### Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.518, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001082-2` (score=0.5803) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyry...
2. `medquad_ans_0006577-1` (score=0.5774) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particul...
3. `medquad_ans_0001082-1` (score=0.5632) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as...
4. `medquad_ans_0001082-5` (score=0.5624) - These resources address the diagnosis or management of 2-methylbutyryl-CoA dehydrogenase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of 2-methylbutyryl-CoA dehydrogenase These resources from...
5. `medquad_ans_0006577-2` (score=0.5355) - What are the signs and symptoms of 2-methylbutyryl-CoA dehydrogenase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for 2-methylbutyryl-CoA dehydrogenase deficiency. If the ...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a rare metabolic disorder in which individuals lack adequate levels of the enzyme 2-methylbutyryl-CoA dehydrogenase, which is necessary for the proper processing of the amino acid isoleucine. This leads to the buildup of isoleucine and organic acids in the body, which can be toxic and cause a range of health problems. Symptoms vary widely, from severe and life-threatening to mild or absent, and may include poor feeding, lack of energy (lethargy), vomiting, irritability, difficulty breathing, seizures, coma, poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills such as standing and walking. Symptoms can be triggered by fasting, infections, or high-protein diets. The disorder is more common among individuals of Hmong ancestry, with an incidence of 1 in 250 to

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Scores**: context_relevance=1.000, groundedness=0.537, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005840-2` (score=0.6122) - What are the signs and symptoms of Spondyloepiphyseal dysplasia congenita? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia congenita. If the information ...
2. `medquad_ans_0005845-1` (score=0.6004) - What are the signs and symptoms of Spondyloepiphyseal dysplasia-brachydactyly and distinctive speech? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia-bra...
3. `medquad_ans_0005821-2` (score=0.5833) - What are the signs and symptoms of Spondylocostal dysostosis 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylocostal dysostosis 3. If the information is available, the tabl...
4. `medquad_ans_0005857-1` (score=0.5829) - What are the signs and symptoms of Spondylometaphyseal dysplasia X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia X-linked. If the information ...
5. `medquad_ans_0005819-2` (score=0.5825) - What are the signs and symptoms of Spondylocostal dysostosis 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylocostal dysostosis 1. If the information is available, the tabl...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include abnormal form of the vertebral bodies (90%), abnormality of epiphysis morphology (90%), micromelia (90%), narrow chest (90%), short neck (90%), short stature (90%), short thorax (90%), skeletal dysplasia (90%), broad forehead (50%), cleft palate (50%), hyperlordosis (50%), hypertelorism (50%), malar flattening (50%), osteoarthritis (50%), talipes (50%), cataract (7.5%), glaucoma (7.5%), hearing impairment (7.5%), kyphosis (7.5%), myopia (7.5%), nystagmus (7.5%), retinal detachment (7.5%), and scoliosis (7.5%)

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Scores**: context_relevance=0.500, groundedness=0.622, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000853-2` (score=0.5389) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
2. `medquad_ans_0000852-2` (score=0.5330) - Pyridoxal 5'-phosphate-dependent epilepsy is a rare condition; approximately 14 cases have been described in the scientific literature.
3. `medquad_ans_0000853-1` (score=0.5272) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepti...
4. `medquad_ans_0000852-1` (score=0.5238) - Pyridoxal 5'-phosphate-dependent epilepsy is a condition that involves seizures beginning soon after birth or, in some cases, before birth. The seizures typically involve irregular involuntary muscle contractions (myo...
5. `medquad_ans_0005219-1` (score=0.5214) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepti...

**Answer**:

The provided context does not specify the underlying cause of pyridoxine-dependent epilepsy. It describes the clinical features, prevalence, and treatment but does not identify a specific etiology. Therefore, based on the given information, the cause cannot be determined. [medquad_ans_0000853-1][medquad_ans_0000853-2]

**Supported citations**: `medquad_ans_0000853-1`, `medquad_ans_0000853-2`
**Unsupported citations**: _none_

### Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Scores**: context_relevance=0.875, groundedness=0.953, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000004_6-5` (score=0.4453) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whet...
2. `medquad_ans_0000004_6-2` (score=0.4428) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
3. `medquad_ans_0000004_1-2` (score=0.4382) - Age, gender, and Epstein-Barr infection can affect the risk of adult Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will g...
4. `medquad_ans_0000031_2-2` (score=0.4363) - The risk factors for neuroblastoma are not known.
5. `medquad_ans_0000004_6-6` (score=0.4301) - Key Points - After childhood Hodgkin lymphoma has been diagnosed, tests are done to find out if cancer cells have spread within the lymph system or to other parts of the body. - There are three ways that cancer spread...

**Answer**:

Children who are infected with the Epstein-Barr virus, those infected with the human immunodeficiency virus (HIV), children with certain diseases of the immune system, children with a personal history of mononucleosis ("mono"), and children who have a parent or sibling with a personal history of Hodgkin lymphoma are at increased risk for childhood Hodgkin lymphoma. Being exposed to common infections in early childhood may decrease the risk of Hodgkin lymphoma in children due to its effect on the immune system. Having a risk factor does not mean that a child will get cancer; not having risk factors does not mean that a child will not get cancer [medquad_ans_0000004_6-2].

**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

### Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Scores**: context_relevance=1.000, groundedness=0.623, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000095-1__dup4` (score=0.4772) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation ...
2. `medquad_ans_0001795-1` (score=0.4751) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow...
3. `medquad_ans_0001795-3` (score=0.4612) - How might dermatitis herpetiformis be treated? The antibiotic dapsone is extremely effective in treating this condition. Symptomatic improvement may occur in as little as several hours after the first dose. However, d...
4. `medquad_ans_0001795-2` (score=0.4483) - What are the signs and symptoms of Dermatitis herpetiformis ? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatitis herpetiformis . If the information is available, the table be...
5. `medquad_ans_0000095-2__dup3` (score=0.4451) - Dermatitis herpetiformis is caused by the deposit of immunoglobulin A (IgA) in the skin, which triggers further immunologic reactions resulting in lesion formation. DH is an external manifestation of an abnormal immun...

**Answer**:

Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions, which are more common on the knees, elbows, buttocks, and shoulder blades [medquad_ans_0001795-1]. It is often associated with the inability to digest gluten and may present with symptoms such as fluid-filled sores, red lesions resembling hives, itchiness, redness, and burning [medquad_ans_0001795-1]. The condition is frequently linked to celiac disease, and patients may also exhibit dental enamel defects in permanent teeth [medquad_ans_0000095-1]. Signs and symptoms include abnormal blistering of the skin (seen in 90% of patients), autoimmunity (90%), pruritus (90%), malabsorption (90%), and microcytic

**Supported citations**: `medquad_ans_0001795-1`
**Unsupported citations**: `medquad_ans_0000095-1`

### Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Scores**: context_relevance=1.000, groundedness=0.953, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000709-5` (score=0.5381) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Regi...
2. `medquad_ans_0000812-5` (score=0.5298) - These resources address the diagnosis or management of potassium-aggravated myotonia: - Genetic Testing Registry: Potassium aggravated myotonia These resources from MedlinePlus offer information about the diagnosis an...
3. `medquad_ans_0000224-5` (score=0.5140) - These resources address the diagnosis or management of congenital fiber-type disproportion: - Gene Review: Gene Review: Congenital Fiber-Type Disproportion - Genetic Testing Registry: Congenital myopathy with fiber ty...
4. `medquad_ans_0004290-4` (score=0.5137) - What treatment is available for for myotonic dystrophy? There is currently no cure or specific treatment for myotonic dystrophy. Treatment is aimed at managing symptoms and minimizing disability. Routine physical acti...
5. `medquad_ans_0000040-2__dup1` (score=0.5078) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors ...

**Answer**:

Most people with myotonia congenita do not require special treatments; stiff muscles usually resolve with exercise or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors have had some success with medications. [medquad_ans_0000709-5]

**Supported citations**: `medquad_ans_0000709-5`
**Unsupported citations**: _none_

### Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Scores**: context_relevance=0.833, groundedness=0.885, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001037-2` (score=0.5927) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
2. `medquad_ans_0006449-5` (score=0.5920) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lif...
3. `medquad_ans_0000973-1__dup1` (score=0.5882) - Wilson disease is a rare inherited disorder that prevents your body from getting rid of extra copper. You need a small amount of copper from food to stay healthy. Too much copper is poisonous. Normally, your liver rel...
4. `medquad_ans_0000133-11` (score=0.5861) - People with Wilson disease should reduce their dietary copper intake by avoiding foods that are high in copper, such as - shellfish - liver - mushrooms - nuts - chocolate People should not eat these foods during the i...
5. `medquad_ans_0000133-1__dup3` (score=0.5855) - Wilson disease is a genetic disease that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filt...

**Answer**:

For Wilson disease, treatment involves lifelong management to reduce or control copper accumulation in the body. This includes taking medications that help remove excess copper and following a low-copper diet. Specific dietary restrictions include avoiding foods high in copper such as shellfish, liver, mushrooms, nuts, and chocolate. During initial treatment, these foods should be avoided; during maintenance treatment, they may be consumed in moderation only after consulting a healthcare provider. Additionally, individuals should have their drinking water tested for copper content, especially if it comes from a well or runs through copper pipes, and should avoid using copper containers [medquad_ans_0000973-1__dup1]. If treatment is ineffective or liver failure develops, a liver transplant may be necessary [medquad_ans_0006449-5].

**Supported citations**: `medquad_ans_0000973-1__dup1`, `medquad_ans_0006449-5`
**Unsupported citations**: _none_

### Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.803, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000814-3` (score=0.4424) - Mutations in the MECP2 gene cause PPM-X syndrome. The MECP2 gene provides instructions for making a protein called MeCP2 that is critical for normal brain function. Researchers believe that this protein has several fu...
2. `medquad_ans_0001024-1__dup1` (score=0.4394) - A CDKL5-related disorder is a genetic, neuro-developmental condition due to changes (mutations) in the CDKL5 gene. Epileptic encephalopathy (epilepsy accompanied by cognitive and behavioral problems) is the core sympt...
3. `medquad_ans_0000548-3` (score=0.4235) - The genetics of juvenile myoclonic epilepsy are complex and not completely understood. Mutations in one of several genes can cause or increase susceptibility to this condition. The most studied of these genes are the ...
4. `medquad_ans_0000741-3` (score=0.4216) - Mutations in the CLN8 gene cause Northern epilepsy. The CLN8 gene provides instructions for making a protein whose function is not well understood. The CLN8 protein is thought to play a role in transporting materials ...
5. `medquad_ans_0000635-3` (score=0.4202) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...

**Answer**:

The provided context does not contain information about the genetic changes related to Lennox-Gastaut syndrome. [medquad_ans_0001024-1__dup1] mentions that CDKL5 mutations have been found in people diagnosed with Lennox-Gastaut syndrome, but it does not specify the genetic changes or mutations directly associated with Lennox-Gastaut syndrome. Therefore, based on the given context, the genetic changes related to Lennox-Gastaut syndrome cannot be determined.

**Supported citations**: `medquad_ans_0001024-1__dup1`
**Unsupported citations**: _none_

### Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Scores**: context_relevance=0.500, groundedness=0.969, answer_relevance=0.420

**Top docs**:

1. `medquad_ans_0000359-2` (score=0.6271) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
2. `medquad_ans_0000066-2` (score=0.6156) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
3. `medquad_ans_0000650-2` (score=0.6042) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
4. `medquad_ans_0000989-2` (score=0.6023) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000934-2` (score=0.5963) - This condition affects fewer than 1 in 150,000 males and is very rare in females.

**Answer**:

The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families [medquad_ans_0000066-2].

**Supported citations**: `medquad_ans_0000066-2`
**Unsupported citations**: _none_

### Example 231 (`0002399-2`)

**Question**: What are the symptoms of Focal dermal hypoplasia ?

**Scores**: context_relevance=1.000, groundedness=0.917, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002401-1` (score=0.6065) - What are the signs and symptoms of Focal facial dermal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Focal facial dermal dysplasia. If the information is available, the ...
2. `medquad_ans_0002120-1` (score=0.5537) - What are the signs and symptoms of Epidermolysa bullosa simplex with muscular dystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Epidermolysa bullosa simplex with muscular dy...
3. `medquad_ans_0001803-1` (score=0.5536) - What are the signs and symptoms of Dermochondrocorneal dystrophy of Franois? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermochondrocorneal dystrophy of Franois. If the informat...
4. `medquad_ans_0006149-1` (score=0.5535) - What are the signs and symptoms of Transient bullous dermolysis of the newborn? The Human Phenotype Ontology provides the following list of signs and symptoms for Transient bullous dermolysis of the newborn. If the in...
5. `medquad_ans_0000946-2__dup1` (score=0.5486) - What are the signs and symptoms of Cardiofaciocutaneous syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Cardiofaciocutaneous syndrome. If the information is available, the ...

**Answer**:

The symptoms of Focal facial dermal dysplasia include abnormal facial shape, abnormality of the eye, abnormality of the musculature, aplasia/hypoplasia of the skin, atypical scarring of skin, irregular hyperpigmentation, abnormality of the eyebrow, abnormality of the mouth, depressed nasal bridge, palpebral edema, and pointed chin. These symptoms occur in approximately 90% of patients for the first five symptoms and 50% for the remaining symptoms [medquad_ans_0002401-1].

**Supported citations**: `medquad_ans_0002401-1`
**Unsupported citations**: _none_

### Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Scores**: context_relevance=0.818, groundedness=0.921, answer_relevance=0.891

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=0.5107) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
2. `medquad_ans_0000418-1__dup2` (score=0.4882) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...
3. `medquad_ans_0000913-1__dup1` (score=0.4727) - Trichomoniasis is a sexually transmitted disease caused by a parasite. You get it through sexual intercourse with an infected partner. Many people do not have any symptoms. If you do get symptoms, they usually happen ...
4. `medquad_ans_0000397-2__dup1` (score=0.4687) - The tapeworms that cause taeniasis (Taenia saginata, T. solium, and T. asiatica) are found worldwide. Eating raw or undercooked beef or pork is the primary risk factor for acquiring taeniasis. Persons who don't eat ra...
5. `medquad_ans_0000989-2` (score=0.4637) - This condition affects an estimated 1 in 50,000 people.

**Answer**:

People at risk for Trichinellosis (also known as Trichinosis) include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating undercooked wild game, especially bear meat, puts individuals at risk. Additionally, consuming very small amounts of undercooked meat during preparation or cooking can also lead to infection [medquad_ans_0000418-2__dup1]. The most common species causing human disease is Trichinella spiralis, which is frequently found in pigs, while other species are associated with wild animals [medquad_ans_0000418-2__dup1].

**Supported citations**: `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

### Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001924-2` (score=0.5513) - What are the signs and symptoms of Dravet syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Dravet syndrome. If the information is available, the table below includes how oft...
2. `medquad_ans_0004898-1` (score=0.5008) - What are the signs and symptoms of Pitt-Hopkins-like syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pitt-Hopkins-like syndrome. If the information is available, the table ...
3. `medquad_ans_0005347-2` (score=0.4965) - What are the signs and symptoms of Rett syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Rett syndrome. If the information is available, the table below includes how often t...
4. `medquad_ans_0002480-2` (score=0.4874) - What are the signs and symptoms of Galloway-Mowat syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Galloway-Mowat syndrome. If the information is available, the table below ...
5. `medquad_ans_0004143-2` (score=0.4800) - What are the signs and symptoms of Mowat-Wilson syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Mowat-Wilson syndrome. If the information is available, the table below incl...

**Answer**:

The signs and symptoms of Dravet syndrome include absence seizures, ataxia, autosomal dominant inheritance, cerebral atrophy, cortical visual impairment, epileptic encephalopathy, focal seizures with impairment of consciousness or awareness, generalized myoclonic seizures, hemiclonic seizures, infantile onset, mental deterioration, motor delay, postnatal microcephaly, and status epilepticus [medquad_ans_0001924-2].

**Supported citations**: `medquad_ans_0001924-2`
**Unsupported citations**: _none_

### Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Scores**: context_relevance=0.889, groundedness=0.953, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000826-4` (score=0.4880) - Familial forms of prion disease are inherited in an autosomal dominant pattern, which means one copy of the altered PRNP gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherit...
2. `medquad_ans_0000826-3` (score=0.4758) - Between 10 and 15 percent of all cases of prion disease are caused by mutations in the PRNP gene. Because they can run in families, these forms of prion disease are classified as familial. Familial prion diseases, whi...
3. `medquad_ans_0002316-3` (score=0.4568) - What causes fatal familial insomnia? Fatal familial insomnia (FFI) is caused by a specific change (mutation) in the PRNP gene. PRNP encodes the prion protein. Although the exact function of this protein is unknown, sc...
4. `medquad_ans_0002316-1` (score=0.4519) - Fatal familial insomnia (FFI) is an inherited prion disease that affects the brain and other parts of the nervous system. Prion diseases, also known as transmissible spongiform encephalopathies (TSE), are a group of r...
5. `medquad_ans_0002533-3` (score=0.4462) - What causes Gerstmann-Straussler-Scheinker disease? Gerstmann-Straussler-Scheinker disease (GSS) is usually caused by certain changes (mutations) in the PRNP gene. PRNP encodes a protein called prion protein. Although...

**Answer**:

Genetic changes related to prion disease include mutations in the PRNP gene, which account for between 10 and 15 percent of all cases of prion disease. These mutations result in the production of an abnormally shaped protein called PrPSc from one copy of the gene. This abnormal protein can bind to the normal prion protein (PrPC) and promote its transformation into PrPSc, leading to the accumulation of PrPSc in the brain and neuronal damage [medquad_ans_0000826-3]. Familial forms of prion disease, such as familial Creutzfeldt-Jakob disease (CJD), Gerstmann-Strussler-Scheinker syndrome (GSS), and fatal familial insomnia (FFI), are caused by these PRNP gene mutations and are inherited in an autosomal dominant pattern [medquad_ans_0000826-4]. In some

**Supported citations**: `medquad_ans_0000826-3`, `medquad_ans_0000826-4`
**Unsupported citations**: _none_

### Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0006430-1` (score=0.5892) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information ...
2. `medquad_ans_0005238-1` (score=0.5239) - What are the signs and symptoms of Radial ray hypoplasia choanal atresia? The Human Phenotype Ontology provides the following list of signs and symptoms for Radial ray hypoplasia choanal atresia. If the information is...
3. `medquad_ans_0004592-1` (score=0.5237) - What are the signs and symptoms of Orofaciodigital syndrome 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 5. If the information is available, the table ...
4. `medquad_ans_0006242-1` (score=0.5205) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
5. `medquad_ans_0006429-1` (score=0.5205) - What are the signs and symptoms of Weyers acrofacial dysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers acrofacial dysostosis. If the information is available, the ta...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, cleft palate, cleft upper lip, high palate, hydronephrosis, hypoplasia of the radius, hypotelorism, long face, mesomelia, narrow face, oligodactyly (hands), proximal placement of thumb, and proximal radial head dislocation [medquad_ans_0006430-1].

**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

### Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Scores**: context_relevance=1.000, groundedness=0.883, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.6044) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the ta...
2. `medquad_ans_0004524-1` (score=0.5149) - What are the signs and symptoms of Okamoto syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Okamoto syndrome. If the information is available, the table below includes how o...
3. `medquad_ans_0004559-1` (score=0.5117) - What are the signs and symptoms of Optic atrophy 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Optic atrophy 2. If the information is available, the table below includes how oft...
4. `medquad_ans_0003352-1` (score=0.5057) - What are the signs and symptoms of Karak syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Karak syndrome. If the information is available, the table below includes how often...
5. `medquad_ans_0004397-1` (score=0.5023) - What are the signs and symptoms of Neuropathy, hereditary motor and sensory, Okinawa type? The Human Phenotype Ontology provides the following list of signs and symptoms for Neuropathy, hereditary motor and sensory, O...

**Answer**:

The symptoms of Premature aging Okamoto type include abnormality of the blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face [medquad_ans_0005029-1]. The frequency of these symptoms is based on data from the Human Phenotype Ontology, which compiles information from sources such as Orphanet, with frequencies often expressed as percentages or fractions representing the proportion of patients exhibiting each symptom [medquad_ans_0005029-1].

**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

### Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Scores**: context_relevance=0.111, groundedness=0.000, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000934-2` (score=0.6160) - This condition affects fewer than 1 in 150,000 males and is very rare in females.
2. `medquad_ans_0000989-2` (score=0.6138) - This condition affects an estimated 1 in 50,000 people.
3. `medquad_ans_0000650-2` (score=0.6132) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.
4. `medquad_ans_0000359-2` (score=0.6095) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
5. `medquad_ans_0000091-2` (score=0.5950) - This condition is rare, affecting fewer than 1 per million people.

**Answer**:

The provided context does not specify the prevalence of thrombocytopenia-absent radius syndrome. Therefore, it is not possible to determine how many people are affected based on the given information. [No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Scores**: context_relevance=0.909, groundedness=0.126, answer_relevance=0.764

**Top docs**:

1. `medquad_ans_0000005-6__dup1` (score=0.5517) - Clinical trials are part of clinical research and at the heart of all treatment advances. Clinical trials look at new ways to prevent, detect, or treat disease. The National Institute of Mental Health at NIH supports ...
2. `medquad_ans_0000214-4__dup1` (score=0.5194) - The NINDS conducts and supports a wide range of studies that explore the complex systems of brain development. These studies include the identification of the mechanism of action of the known causes of NMD as well as ...
3. `medquad_ans_0000019_3-7` (score=0.5053) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, ta...
4. `medquad_ans_0000207-1__dup2` (score=0.5050) - Summary : Clinical trials are research studies that test how well new medical approaches work in people. Each study answers scientific questions and tries to find better ways to prevent, screen for, diagnose, or treat...
5. `medquad_ans_0000927-1__dup2` (score=0.5040) - Summary : It seems to happen almost every day - you hear about the results of a new medical research study. Sometimes the results of one study seem to disagree with the results of another study. It's important to be c...

**Answer**:

The provided context does not mention any research or clinical trials specifically for Tay-Sachs Disease. Therefore, based on the given information, it is not possible to answer the question. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000008-1` (score=0.4699) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achro...
2. `medquad_ans_0000050-1__dup1` (score=0.4683) - What are the signs and symptoms of Achromatopsia 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Achromatopsia 3. If the information is available, the table below includes how oft...
3. `medquad_ans_0000008-3` (score=0.4500) - Achromatopsia results from changes in one of several genes: CNGA3, CNGB3, GNAT2, PDE6C, or PDE6H. A particular CNGB3 gene mutation underlies the condition in Pingelapese islanders. Achromatopsia is a disorder of the r...
4. `medquad_ans_0000763-2__dup1` (score=0.4493) - What are the signs and symptoms of Blue cone monochromatism? The Human Phenotype Ontology provides the following list of signs and symptoms for Blue cone monochromatism. If the information is available, the table belo...
5. `medquad_ans_0000008-2` (score=0.4419) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on o...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. The condition also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals may have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia is different from common forms of color vision deficiency (color blindness), in which people can perceive color but have difficulty distinguishing between certain colors, such as red and green. The condition results from changes in one of several

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Scores**: context_relevance=1.000, groundedness=0.817, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000049-4__dup3` (score=0.5130) - Treatment Goals: Manage Pain, Improve Function Osteoarthritis treatment plans often include ways to manage pain and improve function. Such plans can include exercise, rest and joint care, pain relief, weight control, ...
2. `medquad_ans_0000945-5` (score=0.5091) - These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliative Care
3. `medquad_ans_0000544-1__dup1` (score=0.5020) - Knee replacement is surgery for people with severe knee damage. Knee replacement can relieve pain and allow you to be more active. Your doctor may recommend it if you have knee pain and medicine and other treatments a...
4. `medquad_ans_0000039-9` (score=0.4912) - The main reason to have knee replacement surgery is to ease pain and disability caused by arthritis or other joint problems, while preserving movement. Less commonly, it is used to correct some kinds of knee deformity.
5. `medquad_ans_0000039-11` (score=0.4884) - Treatments your doctor will likely recommend before knee replacement include - exercises to strengthen the muscles around the knee and improve flexibility - weight loss, if needed, to reduce the load the knee must bea...

**Answer**:

The provided context does not mention specific treatments for knee replacement. While it discusses general osteoarthritis treatments such as exercise, weight control, pain relief, and surgery, it does not detail treatments specifically for knee replacement [medquad_ans_0000049-4__dup3]. Therefore, based on the given information, the treatments for knee replacement cannot be determined.

**Supported citations**: `medquad_ans_0000049-4__dup3`
**Unsupported citations**: _none_

### Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.660, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0001692-2` (score=0.5816) - What are the signs and symptoms of Cushing's syndrome? The signs and symptoms of Cushing's syndrome may include: Upper body obesity Severe fatigue Muscle weakness High blood pressure Backache Elevated blood sugar Easy...
2. `medquad_ans_0005690-2` (score=0.5803) - What are the signs and symptoms of Sotos syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Sotos syndrome. If the information is available, the table below includes how often...
3. `medquad_ans_0003313-1` (score=0.5759) - What are the signs and symptoms of Juberg Marsidi syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Juberg Marsidi syndrome. If the information is available, the table below ...
4. `medquad_ans_0002359-2` (score=0.5729) - What are the signs and symptoms of FG syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for FG syndrome. If the information is available, the table below includes how often the s...
5. `medquad_ans_0004494-1` (score=0.5691) - What are the signs and symptoms of Oculo skeletal renal syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Oculo skeletal renal syndrome. If the information is available, the ...

**Answer**:

The provided context does not contain information about the symptoms of Metabolic Syndrome. Therefore, a response cannot be generated based on the given context. [medquad_ans_0001692-2]

**Supported citations**: `medquad_ans_0001692-2`
**Unsupported citations**: _none_

### Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Scores**: context_relevance=1.000, groundedness=0.957, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005211-1` (score=0.6602) - What are the signs and symptoms of Pyknoachondrogenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyknoachondrogenesis. If the information is available, the table below includ...
2. `medquad_ans_0005210-2` (score=0.6163) - What are the signs and symptoms of Pycnodysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pycnodysostosis. If the information is available, the table below includes how oft...
3. `medquad_ans_0001014-1__dup1` (score=0.5738) - What are the signs and symptoms of Caudal appendage deafness? The Human Phenotype Ontology provides the following list of signs and symptoms for Caudal appendage deafness. If the information is available, the table be...
4. `medquad_ans_0005261-2` (score=0.5679) - What are the signs and symptoms of Reactive arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Reactive arthritis. If the information is available, the table below includes h...
5. `medquad_ans_0002145-1` (score=0.5612) - What are the signs and symptoms of Epiphyseal dysplasia hearing loss dysmorphism? The Human Phenotype Ontology provides the following list of signs and symptoms for Epiphyseal dysplasia hearing loss dysmorphism. If th...

**Answer**:

The symptoms of Pyknoachondrogenesis include abnormal vertebral ossification, abnormality of pelvic girdle bone morphology, abnormality of the mouth, abnormality of the sacrum, depressed nasal ridge, enlarged thorax, increased bone mineral density, low-set, posteriorly rotated ears, macrocephaly, micromelia, palpebral edema, premature birth, short stature, short thorax, thickened nuchal skin fold, and stillbirth. These symptoms are reported in approximately 90% of patients with this condition [medquad_ans_0005211-1].

**Supported citations**: `medquad_ans_0005211-1`
**Unsupported citations**: _none_

### Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003966-1` (score=0.5691) - What are the signs and symptoms of Metachromatic leukodystrophy due to saposin B deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy due to sapos...
2. `medquad_ans_0003965-2` (score=0.5683) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the ta...
3. `medquad_ans_0005330-2` (score=0.5493) - What are the signs and symptoms of Retinal vasculopathy with cerebral leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal vasculopathy with cerebral leukodystroph...
4. `medquad_ans_0003070-2` (score=0.5475) - What are the signs and symptoms of Hypomyelination with atrophy of basal ganglia and cerebellum? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypomyelination with atrophy of basal...
5. `medquad_ans_0001799-1` (score=0.5466) - What are the signs and symptoms of Dermatoleukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatoleukodystrophy. If the information is available, the table below incl...

**Answer**:

The symptoms of Metachromatic leukodystrophy include behavioral abnormality, cognitive impairment, decreased nerve conduction velocity, developmental regression, gait disturbance, genu recurvatum, incoordination, muscle weakness, neurological speech impairment, peripheral neuropathy, reduced consciousness/confusion, seizures, amaurosis fugax, hyperreflexia, hypertonia, and limitation of joint mobility [medquad_ans_0003965-2].

**Supported citations**: `medquad_ans_0003965-2`
**Unsupported citations**: _none_

### Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.5811) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000682-4` (score=0.5754) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000105-4` (score=0.5754) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000809-4` (score=0.5754) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000663-4` (score=0.5754) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether chronic granulomatous disease is inherited. While several documents describe autosomal recessive inheritance patterns in general, none explicitly mention chronic granulomatous disease. Therefore, based on the given information, it cannot be determined if chronic granulomatous disease is inherited.

[Note: The context does not contain information about chronic granulomatous disease specifically.]

**Supported citations**: _none_
**Unsupported citations**: `Note: The context does not contain information about chronic granulomatous disease specifically.`

### Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.910, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000061-3__dup1` (score=0.5431) - The prognosis for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict he likeliho...
2. `medquad_ans_0000266-3__dup1` (score=0.4997) - With treatment, individuals with tethered spinal cord syndrome have a normal life expectancy. However, some neurological and motor impairments may not be fully correctable. Surgery soon after symptoms emerge appears t...
3. `medquad_ans_0000061-4__dup1` (score=0.4974) - Our understanding of central cord syndrome has increased greatly in recent decades as a result of research funded conducted by the National Institute of Neurological Disorders and Stroke (NINDS). Much of this research...
4. `medquad_ans_0000013-3__dup1` (score=0.4945) - The overall prognosis for individuals with AIDS in recent years has improved significantly because of new drugs and treatments. AIDS clinicians often fail to recognize neurological complications of AIDS. Those who sus...
5. `medquad_ans_0000001_1-4` (score=0.4785) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The age of the patient. - Whether the cancer has spread ...

**Answer**:

The outlook for central cord syndrome is generally favorable, with most people experiencing some recovery of neurological function, especially when medical intervention is received soon after injury. Many individuals recover substantial function, and the ability to walk is regained in most cases, although some impairment may remain. Recovery typically begins in the legs, followed by the bladder, and hand function recovers last, if at all. Improvement is generally better in younger patients compared to those over the age of 50. Evaluation of abnormal signals on MRI images can help predict the likelihood of natural neurological recovery [medquad_ans_0000061-3__dup1].

**Supported citations**: `medquad_ans_0000061-3__dup1`
**Unsupported citations**: _none_

### Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.937, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000261-3` (score=0.5597) - Although the causes of cyclic vomiting syndrome have yet to be determined, researchers have proposed several factors that may contribute to the disorder. These factors include changes in brain function, hormonal abnor...
2. `medquad_ans_0000094-1__dup6` (score=0.5238) - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no ap...
3. `medquad_ans_0000261-4` (score=0.5221) - In most cases of cyclic vomiting syndrome, affected people have no known history of the disorder in their family. However, many affected individuals have a family history of related conditions, such as migraines, irri...
4. `medquad_ans_0000094-4__dup3` (score=0.5195) - Cyclic vomiting syndrome is more common in children than adults, although reports of the syndrome in adults have increased in recent years.1 Usually, children are about 5 years old when diagnosed with cyclic vomiting ...
5. `medquad_ans_0000094-13` (score=0.5182) - - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no ...

**Answer**:

The causes of cyclic vomiting syndrome have not been fully determined, but researchers have proposed several contributing factors. These include changes in brain function, hormonal abnormalities, and gastrointestinal problems. Many researchers believe that cyclic vomiting syndrome is a migraine-like condition, related to changes in signaling between nerve cells (neurons) in certain areas of the brain. Abnormalities of the autonomic nervous system, which controls involuntary body functions such as heart rate, blood pressure, and digestion, are also observed. In some cases, particularly those beginning in childhood, changes in mitochondrial DNA have been associated with the disorder. These changes may impair mitochondrial energy production, potentially leading to dysfunction in autonomic nervous system cells and affecting the digestive system. However, it remains unclear how these mitochondrial changes specifically cause symptoms such as nausea, vomiting, lethargy, abdominal pain, or migraines [medquad_ans_0000261-3].

**Supported citations**: `medquad_ans_0000261-3`
**Unsupported citations**: _none_

### Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Scores**: context_relevance=0.875, groundedness=0.827, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000691-1__dup1` (score=0.5742) - If you're a parent, you get plenty of suggestions on how to raise your child. From experts to other parents, people are always ready to offer advice. Parenting tips, parents' survival guides, dos, don'ts, shoulds and ...
2. `medquad_ans_0000189-1__dup2` (score=0.5421) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...
3. `medquad_ans_0000875-1__dup1` (score=0.5407) - Summary : Being a teenager is hard. You're under stress to be liked, do well in school, get along with your family, and make big decisions. You can't avoid most of these pressures, and worrying about them is normal. B...
4. `medquad_ans_0000879-1__dup1` (score=0.5398) - Summary : If you are a teenager, this page is for you! It includes materials specifically for you - not for your parents - about health and safety for teens. There are quizzes, games and lots of cool web sites for you...
5. `medquad_ans_0000180-1__dup2` (score=0.5369) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not alwa...

**Answer**:

Yes, there is information about child mental health. The context indicates that mental health problems in children are real, painful, and sometimes severe, and can be signs of conditions such as persistent sadness, hopelessness, anger, grief, or obsessive behaviors like binge-eating or substance use. Early recognition and treatment of mental illnesses in children are important, as they can become a regular part of a child's behavior once developed. Specific warning signs include often feeling very angry or worried, prolonged grief after a loss, feeling that one's mind is out of control, using alcohol or drugs, obsessive exercise or dieting, hurting others, or engaging in reckless behavior [medquad_ans_0000875-1__dup1]. Additionally, it is recommended to seek help from parents, school counselors, or healthcare providers if these signs are present [medquad_ans_0000875-1__dup1].

**Supported citations**: `medquad_ans_0000875-1__dup1`
**Unsupported citations**: _none_

### Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.536, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001618-2` (score=0.5783) - What are the signs and symptoms of Cowden syndrome? Cowden syndrome is characterized primarily by multiple, noncancerous growths (called hamartomas) on various parts of the body. Approximately 99% of people affected b...
2. `medquad_ans_0000740-2__dup1` (score=0.5420) - What are the signs and symptoms of Birt-Hogg-Dube syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Birt-Hogg-Dube syndrome. If the information is available, the table below ...
3. `medquad_ans_0005115-2` (score=0.5282) - What are the signs and symptoms of Proteus syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Proteus syndrome. If the information is available, the table below includes how o...
4. `medquad_ans_0003532-1` (score=0.5273) - What are the signs and symptoms of Leiomyoma of vulva and esophagus? The Human Phenotype Ontology provides the following list of signs and symptoms for Leiomyoma of vulva and esophagus. If the information is available...
5. `medquad_ans_0000842-2__dup1` (score=0.5268) - What are the signs and symptoms of BRCA2 hereditary breast and ovarian cancer syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for BRCA2 hereditary breast and ovarian cancer syn...

**Answer**:

The symptoms of Cowden syndrome include multiple noncancerous growths (hamartomas) on the skin and/or in the mouth, with approximately 99% of affected individuals having these by the third decade of life. A majority also develop hamartomatous polyps in the gastrointestinal tract. Other common symptoms include benign diseases of the breast, thyroid, and endometrium; a rare noncancerous brain tumor called Lhermitte-Duclos disease; enlarged head (macrocephaly); autism spectrum disorder; intellectual disability; and vascular abnormalities. Specific signs and symptoms reported in the Human Phenotype Ontology include abnormality of the pupil, abnormality of the tongue, aplasia/hypoplasia of the cerebellum, arteriovenous malformation, cognitive impairment, conjunctival hamartoma, dental malocclusion, epibulbar dermoid, exostoses, foot polydactyly,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.776, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000395-5` (score=0.5362) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
2. `medquad_ans_0000576-5` (score=0.4868) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...
3. `medquad_ans_0000224-5` (score=0.4730) - These resources address the diagnosis or management of congenital fiber-type disproportion: - Gene Review: Gene Review: Congenital Fiber-Type Disproportion - Genetic Testing Registry: Congenital myopathy with fiber ty...
4. `medquad_ans_0000208-2__dup1` (score=0.4696) - Treatments for the myopathies depend on the disease or condition and specific causes. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders. Treatment for other disor...
5. `medquad_ans_0000209-3` (score=0.4684) - With treatment, muscle weakness may improve or be reversed.

**Answer**:

The provided context does not specify the exact treatments for Fukuyama congenital muscular dystrophy. It mentions that resources are available for the diagnosis or management of Fukuyama congenital muscular dystrophy, including gene reviews, genetic testing registry entries, and MedlinePlus encyclopedia entries, and that general treatments for myopathies may include supportive and symptomatic care, drug therapy, physical therapy, bracing, and surgery [medquad_ans_0000395-5]. However, no specific treatments for Fukuyama congenital muscular dystrophy are detailed in the context. Therefore, based on the available information, the precise treatments cannot be determined.

**Supported citations**: `medquad_ans_0000395-5`
**Unsupported citations**: _none_

### Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.867, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000004_6-4` (score=0.5360) - Tests that examine the lymph system are used to detect (find) and diagnose childhood Hodgkin lymphoma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general...
2. `medquad_ans_0000004_6-6` (score=0.5259) - Key Points - After childhood Hodgkin lymphoma has been diagnosed, tests are done to find out if cancer cells have spread within the lymph system or to other parts of the body. - There are three ways that cancer spread...
3. `medquad_ans_0000004_7-6` (score=0.5145) - Key Points - After childhood non-Hodgkin lymphoma has been diagnosed, tests are done to find out if cancer cells have spread within the lymph system or to other parts of the body. - There are three ways that cancer sp...
4. `medquad_ans_0000003_4-3` (score=0.4999) - Diagnostic tests and a biopsy are used to detect (find) and diagnose childhood rhabdomyosarcoma. The diagnostic tests that are done depend in part on where the cancer forms. The following tests and procedures may be u...
5. `medquad_ans_0000004_7-2` (score=0.4995) - Signs of childhood non-Hodgkin lymphoma include breathing problems and swollen lymph nodes. These and other signs may be caused by childhood non-Hodgkin lymphoma or by other conditions. Check with a doctor if your chi...

**Answer**:

Childhood Hodgkin lymphoma is diagnosed through a combination of tests that examine the lymph system. These include a physical exam and history, CT scan (CAT scan), PET scan (positron emission tomography scan), chest x-ray, complete blood count (CBC), blood chemistry studies, sedimentation rate (ESR), and lymph node biopsy. The physical exam and history assess general signs of health and past medical history. Imaging tests such as CT and PET scans help detect and locate malignant tumor cells, with PET scans showing increased glucose uptake by malignant cells. Chest x-rays evaluate the chest area, while blood tests like CBC, blood chemistry studies, and sedimentation rate provide information on blood cell counts and signs of inflammation, which may indicate lymphoma. A lymph node biopsy, which involves removing all or part of a lymph node, is often performed to confirm the diagnosis [medquad_ans_0000004_6-4].

**Supported citations**: `medquad_ans_0000004_6-4`
**Unsupported citations**: _none_

### Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.6195) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000572-4` (score=0.6172) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000621-4` (score=0.6172) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000418-4` (score=0.6172) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000131-4` (score=0.6172) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention KBG syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether KBG syndrome is inherited.

Answer: Insufficient context [doc_id].

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Scores**: context_relevance=1.000, groundedness=0.539, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000012-1__dup2` (score=0.5089) - Aicardi-Goutieres syndrome (AGS) is an inherited encephalopathy that affects newborn infants and usually results in severe mental and physical handicap. There are two forms of the syndrome: an early-onset form that is...
2. `medquad_ans_0000029-1` (score=0.4963) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. Howeve...
3. `medquad_ans_0000193-2__dup1` (score=0.4836) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 1. If the information is availab...
4. `medquad_ans_0000029-4` (score=0.4766) - Aicardi-Goutieres syndrome can have different inheritance patterns. In most cases it is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an i...
5. `medquad_ans_0000197-2__dup1` (score=0.4719) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 5. If the information is availab...

**Answer**:

Aicardi-Goutieres syndrome (AGS) is an inherited encephalopathy that affects newborn infants and usually results in severe mental and physical handicap. There are two forms of the syndrome: an early-onset form that is severe and a late-onset form that has less impact upon neurological function. The early-onset form affects about 20 percent of all babies who have AGS and is characterized by neurological and liver abnormalities, such as enlargement of the liver and spleen and elevated liver enzymes. Jittery behavior and poor feeding ability mimic congenital viral infection. The later-onset form begins after the first weeks or months of normal development, with symptoms including progressive decline in head growth, weak or stiffened muscles (spasticity), cognitive and developmental delays, irritability, inconsolable crying, intermittent fever, seizures, loss of developmental skills, puffy swelling on the fingers, toes, and ears resembling chilblains,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Scores**: context_relevance=0.917, groundedness=0.917, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000026_2-8` (score=0.5643) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000026_1-7` (score=0.5601) - Key Points - There are different types of treatment for patients with renal cell cancer. - Five types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Biologic therapy - Targeted therapy ...
3. `medquad_ans_0000028_4-4` (score=0.5515) - Cancer prevention clinical trials are used to study ways to prevent cancer. Cancer prevention clinical trials are used to study ways to lower the risk of developing certain types of cancer. Some cancer prevention tria...
4. `medquad_ans_0000007_4-4` (score=0.5484) - Cancer prevention clinical trials are used to study ways to prevent cancer. Cancer prevention clinical trials are used to study ways to lower the risk of developing certain types of cancer. Some cancer prevention tria...
5. `medquad_ans_0000014_3-4` (score=0.5480) - Cancer prevention clinical trials are used to study ways to prevent cancer. Cancer prevention clinical trials are used to study ways to lower the risk of developing certain types of cancer. Some cancer prevention tria...

**Answer**:

Research being done for Renal Cell Cancer includes clinical trials testing new treatments such as regional chemotherapy and regional biologic therapy, where anticancer drugs or biologic substances are placed directly into the renal pelvis or the ureter to target cancer cells in that area [medquad_ans_0000026_2-8]. Additionally, new types of treatments like fulguration (using an electric current to destroy tissue), laser surgery (using a laser beam to remove or kill cancer cells), and segmental resection of the renal pelvis (a surgical procedure to remove localized cancer without removing the entire kidney) are being studied in clinical trials [medquad_ans_0000026_2-8]. These trials aim to determine if new treatments are safe, effective, or better than standard treatments, and patients may have the opportunity to participate before, during, or after starting their cancer treatment [medquad_ans_00

**Supported citations**: `medquad_ans_0000026_2-8`
**Unsupported citations**: _none_

### Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Scores**: context_relevance=0.857, groundedness=0.475, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000190-6` (score=0.4998) - Treatment for primary biliary cirrhosis depends on how early a health care provider diagnoses the disease and whether complications are present. In the early stages of primary biliary cirrhosis, treatment can slow the...
2. `medquad_ans_0000824-5` (score=0.4799) - These resources address the diagnosis or management of primary sclerosing cholangitis: - American Liver Foundation: Primary Sclerosing Cholangitis (PSC) - Genetic Testing Registry: Primary sclerosing cholangitis - Med...
3. `medquad_ans_0000193-7` (score=0.4779) - To prevent cirrhosis, - see your doctor for treatment of your liver disease. Many of the causes of cirrhosis are treatable. Early treatment may prevent cirrhosis. - try to keep your weight in the normal range. Being o...
4. `medquad_ans_0000190-5__dup1` (score=0.4674) - A health care provider may use the following tests to diagnose primary biliary cirrhosis: - a medical and family history - a physical exam - blood tests - imaging tests - a liver biopsy A health care provider usually ...
5. `medquad_ans_0000007_1-5` (score=0.4666) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer (the size of the tumor, whether ...

**Answer**:

Treatment for primary biliary cirrhosis depends on the stage of the disease and the presence of complications. In the early stages, treatment can slow the progression of liver damage to cirrhosis. The primary medication prescribed is ursodiol (Actigall, Urso), a nontoxic bile acid that replaces more toxic bile acids produced by the liver, reducing levels of bilirubin and liver enzymes in the blood. Early treatment with ursodiol reduces the likelihood of needing a liver transplant and improves survival, although it does not cure the disease. Treatment benefits are greatest when started early, but ursodiol can still slow disease progression even when used later.

Patients should avoid alcohol and illegal substances, as these can worsen liver damage. They should also avoid complementary and alternative medications, such as herbs, and should consult a healthcare provider before taking any new prescription, over-the-counter medications, or vitamins, as many can affect liver

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Scores**: context_relevance=0.750, groundedness=0.873, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=0.5737) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People ...
2. `medquad_ans_0000023-1__dup1` (score=0.5406) - Cerebral hypoxia refers to a condition in which there is a decrease of oxygen supply to the brain even though there is adequate blood flow. Drowning, strangling, choking, suffocation, cardiac arrest, head trauma, carb...
3. `medquad_ans_0000967-1__dup1` (score=0.5139) - Summary : Playing in the water - whether swimming, boating or diving - can be fun. It can also be dangerous, especially for children. Being safe can help prevent injuries and drowning. To stay safe in the water - Avoi...
4. `medquad_ans_0000916-1__dup2` (score=0.4824) - Summary : A tsunami is a series of huge ocean waves created by an underwater disturbance. Causes include earthquakes, landslides, volcanic eruptions, or meteorites--chunks of rock from space that strike the surface of...
5. `medquad_ans_0000977-1__dup1` (score=0.4593) - An injury is damage to your body. It is a general term that refers to harm caused by accidents, falls, hits, weapons, and more. In the U.S., millions of people injure themselves every year. These injuries range from m...

**Answer**:

Drowning is a condition that occurs when a person takes in too much water, leading to a lack of oxygen in the lungs. It can happen quickly and silently, even with as little as an inch or two of water. Babies can drown in a sink or bathtub, and preschoolers are most at risk in swimming pools. People with seizure disorders are also at increased risk in the water. Drowning can result in cerebral hypoxia, a condition where the brain does not receive enough oxygen, which can lead to brain cell death within five minutes of oxygen deprivation. In severe cases, prolonged hypoxia can cause coma, seizures, or even brain death [medquad_ans_0000299-1\_\_dup2][medquad_ans_0000023-1__dup1]. Drowning is also the most common cause of death related to tsunamis [medquad_ans_00009

**Supported citations**: `medquad_ans_0000299-1__dup2`, `medquad_ans_0000023-1__dup1`
**Unsupported citations**: _none_

### Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Scores**: context_relevance=1.000, groundedness=0.974, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002058-2` (score=0.5112) - What are the signs and symptoms of Ehlers-Danlos syndrome, kyphoscoliosis type? The signs and symptoms of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type vary but may include: Hyperextensible skin that is fragile an...
2. `medquad_ans_0002058-6` (score=0.4975) - How might Ehlers-Danlos syndrome, kyphoscoliosis type be treated? The treatment of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is focused on preventing serious complications and relieving associated signs and sy...
3. `medquad_ans_0002058-1` (score=0.4920) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragi...
4. `medquad_ans_0002058-3` (score=0.4823) - What causes Ehlers-Danlos syndrome, kyphoscoliosis type? Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is caused by changes (mutations) in the PLOD1 gene. This gene encodes an enzyme that helps process molecules w...
5. `medquad_ans_0002058-5` (score=0.4815) - How is Ehlers-Danlos syndrome, kyphoscoliosis type diagnosed? A diagnosis of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is typically based on the presence of characteristic signs and symptoms. The following tes...

**Answer**:

Ehlers-Danlos syndrome, kyphoscoliosis type is a genetic disorder characterized by hyperextensible skin that is fragile and bruises easily, joint hypermobility leading to frequent dislocations and subluxations, severe hypotonia at birth, progressive kyphoscoliosis (kyphosis and scoliosis) present at birth or within the first year of life, scleral fragility, abnormal wound healing, "marfanoid habitus" (long, slender fingers; unusually long limbs; sunken or protruding chest), fragile arteries prone to rupture, delayed motor development, unusually small cornea, osteopenia (low bone density), congenital clubfoot, and cardiovascular abnormalities such as mitral valve prolapse or aortic root dilatation [medquad_ans_0002058-2]. The condition is associated with a high frequency of symptoms such as abnormality of the mitral valve (9

**Supported citations**: `medquad_ans_0002058-2`
**Unsupported citations**: _none_

### Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.971, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001860-2` (score=0.5745) - What are the signs and symptoms of Dihydropteridine reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropteridine reductase deficiency. If the information is...
2. `medquad_ans_0006567-2` (score=0.5305) - What are the signs and symptoms of 2,4-Dienoyl-CoA reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for 2,4-Dienoyl-CoA reductase deficiency. If the information is a...
3. `medquad_ans_0004302-1` (score=0.5157) - What are the signs and symptoms of NADH cytochrome B5 reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for NADH cytochrome B5 reductase deficiency. If the informatio...
4. `medquad_ans_0001862-2` (score=0.5012) - What are the signs and symptoms of Dihydropyrimidine dehydrogenase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropyrimidine dehydrogenase deficiency. If the info...
5. `medquad_ans_0001902-1` (score=0.4740) - What are the signs and symptoms of Dopamine beta hydroxylase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dopamine beta hydroxylase deficiency. If the information is a...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include cognitive impairment, microcephaly, cerebral calcification, choreoathetosis, dysphagia, dystonia, episodic fever, excessive salivation, hyperphenylalaninemia, hypertonia, infantile onset, intellectual disability, irritability, muscular hypotonia, myoclonus, progressive neurologic deterioration, seizures, tremor, and variable expressivity [medquad_ans_0001860-2].

**Supported citations**: `medquad_ans_0001860-2`
**Unsupported citations**: _none_

### Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Scores**: context_relevance=0.286, groundedness=0.709, answer_relevance=0.286

**Top docs**:

1. `medquad_ans_0000669-2` (score=0.6322) - The prevalence of monilethrix is unknown.
2. `medquad_ans_0000359-2` (score=0.5845) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
3. `medquad_ans_0000989-2` (score=0.5656) - This condition affects an estimated 1 in 50,000 people.
4. `medquad_ans_0000685-2` (score=0.5585) - MONA is rare; its prevalence is unknown. This condition has been reported in multiple populations worldwide.
5. `medquad_ans_0000650-2` (score=0.5558) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

The prevalence of monilethrix is unknown [medquad_ans_0000669-2]. However, one source estimates that it affects about 1 per million people worldwide, and another estimates 1 in 50,000 people [medquad_ans_0000359-2][medquad_ans_0000989-2]. A range of 1 in 50,000 to 1 in 100,000 is also reported [medquad_ans_0000650-2]. Despite these estimates, the exact prevalence remains uncertain. [medquad_ans_0000669-2][medquad_ans_0000359-2][medquad_ans_0000989-2]medquad_ans_0000650-2

**Supported citations**: `medquad_ans_0000669-2`, `medquad_ans_0000359-2`, `medquad_ans_0000989-2`, `medquad_ans_0000650-2`
**Unsupported citations**: _none_

### Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Scores**: context_relevance=1.000, groundedness=0.505, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.5876) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...
2. `medquad_ans_0002733-2` (score=0.5810) - What are the signs and symptoms of Hartnup disease? The signs and symptoms of Hartnup disease may vary and include the following: Skin findings: sensitivity to sunlight Neurologic symptoms: ataxia, spasticity, headach...
3. `medquad_ans_0002890-1` (score=0.4924) - Hirschsprung disease is a disease of the large intestine or colon. People with this disease do not have the nerve cells in the intestine required to expel stools from the body normally. Symptoms of Hirschsprung diseas...
4. `medquad_ans_0000102-3__dup1` (score=0.4854) - The progression of the disorder is usually 10 years or longer.
5. `medquad_ans_0000650-2` (score=0.4769) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system, making it a type of aminoaciduria. It is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. The condition may be diagnosed through newborn screening tests. Most individuals are asymptomatic, but those who do show symptoms typically experience onset between ages 3 and 9, although it can occasionally present in adulthood. Signs and symptoms include skin photosensitivity, neurologic findings (such as ataxia, spasticity, headaches, and hypotonia), psychiatric symptoms (like anxiety, emotional instability, and mood changes), and ocular findings (such as double vision, nystagmus, strabismus, and photophobia). Symptoms may be triggered by sunlight exposure, fever, drugs, or emotional or physical stress and often last 1–4 weeks before spontaneous

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Scores**: context_relevance=0.417, groundedness=0.000, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000683-3` (score=0.4656) - Mutations in the GUSB gene cause MPS VII. This gene provides instructions for producing the beta-glucuronidase (-glucuronidase) enzyme, which is involved in the breakdown of large sugar molecules called glycosaminogly...
2. `medquad_ans_0000342-4` (score=0.4615) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
3. `medquad_ans_0001013-4` (score=0.4610) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000354-4` (score=0.4609) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Rarely, both copies of the gene are altered, leading to more sev...
5. `medquad_ans_0000112-4` (score=0.4595) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. All reported cases have resulted from new mutations in the gene,...

**Answer**:

The provided context does not contain any information about genetic changes related to glycogen storage disease type VII. The context discusses mutations in the GUSB gene and their role in causing mucopolysaccharidosis type VII (MPS VII), a lysosomal storage disorder, not glycogen storage disease type VII. Therefore, based on the given information, the genetic changes related to glycogen storage disease type VII cannot be determined.

Answer: The context does not provide information on genetic changes related to glycogen storage disease type VII. [medquad_ans_0000683-3, medquad_ans_0000342-4, medquad_ans_0001013-4, medquad_ans_0000354-4, medquad_ans_0000112-4]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000683-3, medquad_ans_0000342-4, medquad_ans_0001013-4, medquad_ans_0000354-4, medquad_ans_0000112-4`

### Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Scores**: context_relevance=0.556, groundedness=0.771, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000977-5` (score=0.5810) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megal...
2. `medquad_ans_0000172-2__dup1` (score=0.5144) - Treatment is symptomatic and supportive, and may include drug therapy.
3. `medquad_ans_0000503-5` (score=0.5070) - These resources address the diagnosis or management of hypochromic microcytic anemia with iron overload: - Genetic Testing Registry: Hypochromic microcytic anemia with iron overload These resources from MedlinePlus of...
4. `medquad_ans_0000513-5` (score=0.5052) - These resources address the diagnosis or management of Imerslund-Grsbeck syndrome: - MedlinePlus Encyclopedia: Anemia - B12 deficiency These resources from MedlinePlus offer information about the diagnosis and managem...
5. `medquad_ans_0000124-5` (score=0.5044) - These resources address the diagnosis or management of biotin-thiamine-responsive basal ganglia disease: - Gene Review: Gene Review: Biotin-Thiamine-Responsive Basal Ganglia Disease These resources from MedlinePlus of...

**Answer**:

The treatments for thiamine-responsive megaloblastic anemia syndrome include drug therapy, as indicated by the context. [medquad_ans_0000172-2__dup1] Additionally, management may involve symptomatic and supportive care. [medquad_ans_0000977-5]

**Supported citations**: `medquad_ans_0000172-2__dup1`, `medquad_ans_0000977-5`
**Unsupported citations**: _none_

### Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.935, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000167-18` (score=0.4693) - - Polycystic kidney disease (PKD) is a genetic disorder that causes numerous cysts to grow in the kidneys. - A gene mutation, or defect, causes polycystic kidney disease. - Autosomal dominant PKD is the most common fo...
2. `medquad_ans_0000587-4__dup1` (score=0.4627) - Is there a cure or treatment for autosomal recessive polycystic kidney disease? Although a cure or treatment for the underlying genetic cause of autosomal recessive polycystic kidney disease does not exist, advancemen...
3. `medquad_ans_0000537-1__dup1` (score=0.4604) - A cyst is a fluid-filled sac. There are two types of kidney cysts. Polycystic kidney disease (PKD) runs in families. In PKD, the cysts take the place of the normal tissue. They enlarge the kidneys and make them work p...
4. `medquad_ans_0004939-2` (score=0.4499) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their abilit...
5. `medquad_ans_0000167-1__dup4` (score=0.4462) - Polycystic kidney disease is a genetic disorder that causes numerous cysts to grow in the kidneys. A kidney cyst is an abnormal sac filled with fluid. PKD cysts can greatly enlarge the kidneys while replacing much of ...

**Answer**:

Treatment for polycystic kidney disease (PKD) focuses on managing symptoms and complications. For autosomal dominant PKD, although there is no cure, treatment can ease symptoms and prolong life. This includes managing pain, high blood pressure, kidney failure, urinary tract infections (UTIs), kidney stones, liver cysts, pancreatic cysts, abnormal heart valves, diverticula, and brain aneurysms. People with PKD should talk with a dietitian about dietary changes to support kidney health and avoid foods that may be harmful. Lifestyle changes, diet, and blood pressure medications can help slow the progression of kidney damage caused by high blood pressure [medquad_ans_0000167-18]. For autosomal recessive PKD, treatment also focuses on symptom and complication management, though no cure exists for the underlying genetic cause [medquad_ans_0000167-18]. Scientists have

**Supported citations**: `medquad_ans_0000167-18`
**Unsupported citations**: _none_

### Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Scores**: context_relevance=1.000, groundedness=0.856, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000061-5__dup2` (score=0.5859) - Scientists are making rapid progress in understanding the complexities of rheumatoid arthritis. They are learning more about how and why it develops and why some people have more severe symptoms than others. Research ...
2. `medquad_ans_0000207-1__dup2` (score=0.5833) - Summary : Clinical trials are research studies that test how well new medical approaches work in people. Each study answers scientific questions and tries to find better ways to prevent, screen for, diagnose, or treat...
3. `medquad_ans_0000040-23` (score=0.5681) - Clinical trials are research studies in which new treatments -- drugs, diagnostics, procedures, vaccines, and other therapies -- are tested in people to see if they are safe, effective, and better than the current sta...
4. `medquad_ans_0000019_3-7` (score=0.5595) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, ta...
5. `medquad_ans_0000090-4__dup1` (score=0.5586) - Much of the on-going research on RMDs is aimed at prevention and rehabilitation. The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) funds research on RMDs.

**Answer**:

Research on rheumatoid arthritis is focused on understanding how and why the disease develops, particularly identifying possible triggers such as infectious agents like viruses and bacteria that may contribute to the disease in individuals with a genetic predisposition [medquad_ans_0000061-5__dup2]. Scientists are also investigating the complex relationships between the hormonal, nervous, and immune systems to explain why more women than men develop the disease, including examining how changes in steroid hormones like estrogen and testosterone affect disease development and flares [medquad_ans_0000061-5__dup2]. Additionally, researchers are exploring why rheumatoid arthritis often improves during pregnancy.

Clinical trials are being conducted to test the effectiveness of new medical approaches for preventing, screening, diagnosing, or treating rheumatoid arthritis. These trials may compare new treatments to existing ones and follow strict protocols that define the study's procedures, objectives, and participant eligibility [med

**Supported citations**: `medquad_ans_0000061-5__dup2`
**Unsupported citations**: _none_

### Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.988, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0002561-4` (score=0.5575) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (an...
2. `medquad_ans_0000071-2__dup1` (score=0.5112) - There is no standard course of treatment for Sotos syndrome. Treatment is symptomatic.
3. `medquad_ans_0000172-2__dup1` (score=0.4971) - Treatment is symptomatic and supportive, and may include drug therapy.
4. `medquad_ans_0005571-3` (score=0.4903) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
5. `medquad_ans_0000803-5` (score=0.4792) - These resources address the diagnosis or management of Poland syndrome: - Children's Medical Center of Dallas - Great Ormond Street Hospital (UK): Treatment Options for Symbrachydactyly - St. Louis Children's Hospital...

**Answer**:

Surgery is typically the treatment for Glass-Chapman-Hockley syndrome, and it is based on the person's specific signs and symptoms. The goal is to increase the space in the front (anterior) part of the skull. The operation is usually performed when the person is between 9 to 12 months of age. If other sutures, other than the coronal suture, are involved, other surgeries may be performed [medquad_ans_0002561-4].

**Supported citations**: `medquad_ans_0002561-4`
**Unsupported citations**: _none_

### Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.715, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000868-5` (score=0.5713) - These resources address the diagnosis or management of Renpenning syndrome: - Genetic Testing Registry: Renpenning syndrome 1 - Greenwood Genetics Center: X-Linked Intellectual Disability - Kennedy Krieger Institute: ...
2. `medquad_ans_0000184-5` (score=0.5203) - These resources address the diagnosis or management of Christianson syndrome: - Genetic Testing Registry: Christianson syndrome - MedlinePlus Encyclopedia: Seizures These resources from MedlinePlus offer information a...
3. `medquad_ans_0000009-5` (score=0.5075) - These resources address the diagnosis or management of acral peeling skin syndrome: - Birmingham Children's Hospital, National Health Service (UK) - Genetic Testing Registry: Peeling skin syndrome, acral type These re...
4. `medquad_ans_0000278-5` (score=0.5068) - These resources address the diagnosis or management of Denys-Drash syndrome: - Gene Review: Gene Review: Wilms Tumor Overview - Genetic Testing Registry: Drash syndrome - MedlinePlus Encyclopedia: Nephrotic Syndrome T...
5. `medquad_ans_0000713-5` (score=0.5065) - These resources address the diagnosis or management of Nager syndrome: - Genetic Testing Registry: Nager syndrome - University of California San Francisco Medical Center These resources from MedlinePlus offer informat...

**Answer**:

The provided context does not specify any treatments for Renpenning syndrome. While it mentions resources related to the diagnosis or management of Renpenning syndrome, such as the Genetic Testing Registry and Greenwood Genetics Center, and references to MedlinePlus topics like Diagnostic Tests, Drug Therapy, Surgery and Rehabilitation, Genetic Counseling, and Palliative Care, no specific treatments are detailed. Therefore, based on the given information, the treatments for Renpenning syndrome cannot be determined. [medquad_ans_0000868-5]

**Supported citations**: `medquad_ans_0000868-5`
**Unsupported citations**: _none_

### Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Scores**: context_relevance=1.000, groundedness=0.681, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001108-1` (score=0.6073) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2J? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2J. If the information is ava...
2. `medquad_ans_0001106-1` (score=0.6058) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2H? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2H. If the information is ava...
3. `medquad_ans_0001119-1` (score=0.6037) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 4B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 4B2. If the information is a...
4. `medquad_ans_0001110-1` (score=0.6028) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is ava...
5. `medquad_ans_0001105-1` (score=0.5989) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is ava...

**Answer**:

The provided context does not contain information about the symptoms of Charcot-Marie-Tooth disease type 2G. [medquad_ans_0001108-1] and [medquad_ans_0001106-1] discuss symptoms for types 2J and 2H, respectively, but there is no mention of type 2G. Therefore, the symptoms of Charcot-Marie-Tooth disease type 2G cannot be determined from the given context.

**Supported citations**: `medquad_ans_0001108-1`, `medquad_ans_0001106-1`
**Unsupported citations**: _none_

### Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001393-2` (score=0.5510) - What signs and symptoms are associated with coccygodynia? The classic symptom is pain when pressure is applied to the tailbone, such as when sitting on a hard chair. Symptoms usually improve with relief of pressure wh...
2. `medquad_ans_0001807-1` (score=0.4676) - What are the signs and symptoms of Desmoid tumor? The Human Phenotype Ontology provides the following list of signs and symptoms for Desmoid tumor. If the information is available, the table below includes how often t...
3. `medquad_ans_0005261-2` (score=0.4643) - What are the signs and symptoms of Reactive arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Reactive arthritis. If the information is available, the table below includes h...
4. `medquad_ans_0001692-2` (score=0.4497) - What are the signs and symptoms of Cushing's syndrome? The signs and symptoms of Cushing's syndrome may include: Upper body obesity Severe fatigue Muscle weakness High blood pressure Backache Elevated blood sugar Easy...
5. `medquad_ans_0001163-2` (score=0.4437) - What are the signs and symptoms of Chondrocalcinosis 2? The signs and symptoms of chondrocalcinosis 2 vary from person to person. Some affected people may not have any symptoms of the condition aside from the appearan...

**Answer**:

The symptoms of coccygodynia include pain when pressure is applied to the tailbone, such as when sitting on a hard chair, which usually improves with relief of pressure when standing or walking. Other symptoms may include immediate and severe pain when moving from sitting to standing, pain during bowel movements, pain during sex, and a deep ache in the region of the tailbone [medquad_ans_0001393-2].

**Supported citations**: `medquad_ans_0001393-2`
**Unsupported citations**: _none_

### Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Scores**: context_relevance=1.000, groundedness=0.890, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0004025-1` (score=0.5719) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgen...
2. `medquad_ans_0000292-1__dup1` (score=0.5468) - What are the signs and symptoms of Amish lethal microcephaly? The Human Phenotype Ontology provides the following list of signs and symptoms for Amish lethal microcephaly. If the information is available, the table be...
3. `medquad_ans_0004593-1` (score=0.5445) - What are the signs and symptoms of Orofaciodigital syndrome 8? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 8. If the information is available, the table ...
4. `medquad_ans_0006607-2` (score=0.5340) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
5. `medquad_ans_0004973-1` (score=0.5334) - What are the signs and symptoms of Pontocerebellar hypoplasia type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Pontocerebellar hypoplasia type 3. If the information is availab...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, and microcephaly. Preaxial hand polydactyly, proptosis, and seizures are also listed as symptoms. The frequency of these symptoms is based on data from the Human Phenotype Ontology, which aggregates information from sources such as Orphanet, and represents rough estimates of the percentage of patients who have each feature [medquad_ans_0004025-1].

**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

### Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.980, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003402-5` (score=0.5146) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to d...
2. `medquad_ans_0003402-2` (score=0.5085) - What are the signs and symptoms of Klippel Feil syndrome? Klippel Feil syndrome is characterized by the fusion of 2 or more spinal bones in the neck (cervical vertebrae). The condition is present from birth (congenita...
3. `medquad_ans_0003402-6` (score=0.4881) - How might Klippel-Feil syndrome be treated? There is no cure for Klippel Feil syndrome (KFS); treatment is generally symptomatic and supportive. Management depends on the features and severity in each person, and can ...
4. `medquad_ans_0000170-1__dup1` (score=0.4731) - Klippel-Feil Syndrome is a rare disorder characterized by the congenital fusion of two or more cervical (neck) vertebrae. It is caused by a failure in the normal segmentation or division of the cervical vertebrae duri...
5. `medquad_ans_0000170-3__dup1` (score=0.4691) - The prognosis for most individuals with Klippel-Feil Syndrome is good if the disorder is treated early and appropriately. Activities that can injure the neck should be avoided.

**Answer**:

Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal abnormalities, and additional imaging studies may be needed to assess the extent of the abnormality. Other initial exams are required to detect additional physical abnormalities or underlying conditions, including examination of the chest to rule out heart and lung involvement, examination of the chest wall for rib anomalies, MRI for spinal stenosis or neurological deficits, ultrasound of the kidneys for renal abnormalities, hearing evaluation due to the high incidence of hearing loss, and various lab tests to assess organ function. Additional tests or consultations with specialists may be recommended depending on the features present [medquad_ans_0003402-5].

**Supported citations**: `medquad_ans_0003402-5`
**Unsupported citations**: _none_

### Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Scores**: context_relevance=1.000, groundedness=0.509, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000222-2` (score=0.5101) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
2. `medquad_ans_0001497-1` (score=0.5012) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severit...
3. `medquad_ans_0000222-1` (score=0.4673) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
4. `medquad_ans_0000464-1__dup1` (score=0.4646) - A hernia happens when part of an internal organ or tissue bulges through a weak area of muscle. Most hernias are in the abdomen. There are several types of hernias, including - Inguinal, in the groin. This is the the ...
5. `medquad_ans_0001497-2` (score=0.4407) - What are the signs and symptoms of Congenital diaphragmatic hernia? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital diaphragmatic hernia. If the information is available, ...

**Answer**:

Congenital diaphragmatic hernia (CDH) is a defect in the diaphragm resulting from abnormal development before birth. The diaphragm, composed of muscle and fibrous tissue, normally separates the abdominal organs from those in the chest cavity. In CDH, this development is incomplete, leading to defects ranging from a thinned area in the diaphragm to its complete absence. This creates an abnormal opening (hernia) that allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs. This crowding can lead to underdevelopment of the lungs (pulmonary hypoplasia), potentially causing life-threatening breathing difficulties at birth. CDH may occur with other malformations and is classified by location, such as Bochdalek hernias (80–90% of cases, involving the side or back of the diaphragm) and Morgagni hernias (about

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Scores**: context_relevance=1.000, groundedness=0.545, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000173-1` (score=0.5746) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
2. `medquad_ans_0001088-1__dup1` (score=0.5518) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
3. `medquad_ans_0000074-1__dup1` (score=0.5472) - Charcot-Marie-Tooth disease (CMT) is one of the most common inherited neurological disorders, affecting approximately 1 in 2,500 people in theUnited States. CMT, also known as hereditary motor and sensory neuropathy (...
4. `medquad_ans_0006495-1` (score=0.5456) - What are the signs and symptoms of X-linked Charcot-Marie-Tooth disease type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked Charcot-Marie-Tooth disease type 4. If the in...
5. `medquad_ans_0001119-1` (score=0.5426) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 4B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 4B2. If the information is a...

**Answer**:

Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves, which connect the brain and spinal cord to muscles and sensory cells that detect sensations such as touch, pain, heat, and sound. Damage to these nerves can result in loss of sensation and wasting (atrophy) of muscles in the feet, legs, and hands. The disease typically becomes apparent in adolescence or early adulthood, though onset can occur from early childhood through late adulthood. Symptoms vary in severity even among family members and may include balance difficulties, clumsiness, muscle weakness in the feet, foot abnormalities such as high arches (pes cavus), flat feet (pes planus), or curled toes (hammer toes), difficulty flexing the foot or walking on the heel, a higher than normal step, and decreased sensitivity to touch, heat, and cold. In some cases, individuals may experience gradual hearing loss, deafness, or loss of vision. There are

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Scores**: context_relevance=1.000, groundedness=0.952, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000669-4` (score=0.6139) - Monilethrix can have multiple patterns of inheritance. When the condition is caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene...
2. `medquad_ans_0000785-4` (score=0.5803) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000577-4` (score=0.5785) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000961-4` (score=0.5785) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000949-4` (score=0.5785) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

Yes, monilethrix can be inherited. When caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder. In rare cases, the condition results from a new mutation and is not inherited. When caused by mutations in the DSG4 gene, it is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations [medquad_ans_0000669-4].

**Supported citations**: `medquad_ans_0000669-4`
**Unsupported citations**: _none_

### Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Scores**: context_relevance=0.875, groundedness=0.934, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000223-10` (score=0.5190) - You can take the following steps to help prevent your child from getting a UTI: - Teach your child not to hold in urine and to go to the bathroom whenever your child feels the urge. - Teach your child how to properly ...
2. `medquad_ans_0000215-4__dup2` (score=0.5179) - Urinary tract infections affect about 3 percent of children in the United States every year. UTIs account for more than 1 million visits to pediatricians offices every year.1
3. `medquad_ans_0000215-3__dup2` (score=0.5137) - Most UTIs are caused by bacteria that live in the bowel. The bacterium Escherichia coli (E. coli) causes the vast majority of UTIs. The urinary tract has several systems to prevent infection. The points where the uret...
4. `medquad_ans_0000223-12` (score=0.5111) - - A urinary tract infection (UTI) is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscope. - Most UTIs are caused by bacteria that live in the bowel, ...
5. `medquad_ans_0000223-8` (score=0.5110) - Bacteria-fighting medicines called antibiotics are used to treat a UTI. While the lab is doing the urine culture, the health care provider may begin treatment with an antibiotic that treats the bacteria most likely to...

**Answer**:

A urinary tract infection (UTI) in children is an infection in the urinary tract caused by microbes, primarily bacteria that live in the bowel, such as Escherichia coli (E. coli). Most UTIs are not serious, but some can lead to serious problems. Any child can get a UTI, though girls are more likely to develop them than boys. Symptoms may range from mild to severe or may be absent altogether. UTIs are diagnosed through testing [medquad_ans_0000223-12][medquad_ans_0000215-3__dup2].

**Supported citations**: `medquad_ans_0000223-12`, `medquad_ans_0000215-3__dup2`
**Unsupported citations**: _none_

### Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.965, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0004939-2` (score=0.5000) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their abilit...
2. `medquad_ans_0000804-1` (score=0.4626) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the...
3. `medquad_ans_0000167-18` (score=0.4596) - - Polycystic kidney disease (PKD) is a genetic disorder that causes numerous cysts to grow in the kidneys. - A gene mutation, or defect, causes polycystic kidney disease. - Autosomal dominant PKD is the most common fo...
4. `medquad_ans_0000167-7` (score=0.4559) - The complications of autosomal dominant PKD include the following: - Pain. Cyst infection, other types of urinary tract infections (UTIs), bleeding into cysts, kidney stones, or stretching of the fibrous tissue around...
5. `medquad_ans_0000135-2__dup2` (score=0.4500) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems wi...

**Answer**:

Complications of Polycystic Kidney Disease include dangerously high blood pressure (hypertension), severe pain in the back or sides, blood in the urine (hematuria), recurrent urinary tract infections, kidney stones (nephrolithiasis), heart valve abnormalities, and an increased risk of aortic aneurysm in the brain (cerebral aneurysm) [medquad_ans_0004939-2]. Additionally, cysts may develop in other organs, particularly the liver, leading to cystic liver disease [medquad_ans_0004939-2]. Other complications include renal insufficiency, proteinuria, and abnormality of prenatal development or birth [medquad_ans_0004939-2]. The frequency of these symptoms varies, with hypertension, hematuria, nephrolithiasis, proteinuria, and renal insufficiency each

**Supported citations**: `medquad_ans_0004939-2`
**Unsupported citations**: _none_

### Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Scores**: context_relevance=1.000, groundedness=0.944, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000649-1__dup1` (score=0.5356) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contami...
2. `medquad_ans_0000123-6` (score=0.4899) - Viral gastroenteritis is usually diagnosed based on symptoms alone. People who have symptoms that are severe or last for more than a few days may want to see a health care provider for additional tests. A health care ...
3. `medquad_ans_0000391-1__dup1` (score=0.4798) - Have you ever had the "stomach flu?" What you probably had was gastroenteritis - not a type of flu at all. Gastroenteritis is an inflammation of the lining of the intestines caused by a virus, bacteria or parasites. V...
4. `medquad_ans_0000123-4__dup2` (score=0.4580) - Four types of viruses cause most cases of viral gastroenteritis. Rotavirus Rotavirus is the leading cause of gastroenteritis among infants and young children. Rotavirus infections are most common in infants 3 to 15 mo...
5. `medquad_ans_0000096-2__dup2` (score=0.4510) - Acute diarrhea is usually caused by a bacterial, viral, or parasitic infection. Chronic diarrhea is usually related to a functional disorder such as irritable bowel syndrome or an intestinal disease such as Crohns dis...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by a group of related viruses. Infection leads to inflammation of the stomach and intestines, with common symptoms including diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms are fever, headache, or body aches. The infection can spread from person to person, through contaminated food or water, or by touching a contaminated surface. It can be serious, especially in young children and older adults. There is no specific medicine to treat norovirus infections; treatment focuses on bed rest and drinking plenty of fluids to prevent dehydration. Prevention includes proper hand washing and safe food preparation [medquad_ans_0000649-1__dup1]. Norovirus is a common cause of viral gastroenteritis, and during outbreaks, stool samples may be tested to identify the virus [medquad_ans_0000123-6].

**Supported citations**: `medquad_ans_0000649-1__dup1`, `medquad_ans_0000123-6`
**Unsupported citations**: _none_

### Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.900, groundedness=0.958, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0001497-3` (score=0.4623) - What causes congenital diaphragmatic hernia? Congenital diaphragmatic hernia (CDH) can occur as an isolated finding, as part of a genetic syndrome or chromosome abnormality, or as part of a complex but nonsyndromic se...
2. `medquad_ans_0000538-3` (score=0.4369) - Changes in the DNA near the SOX9 gene are the most common genetic cause of isolated Pierre Robin sequence. It is likely that changes in other genes, some of which have not been identified, also cause isolated Pierre R...
3. `medquad_ans_0000076-3` (score=0.4342) - Mutations in at least 11 genes have been found to cause asphyxiating thoracic dystrophy. Genetic changes in the IFT80 gene were the first to be associated with this condition. Later, researchers discovered that mutati...
4. `medquad_ans_0000948-4` (score=0.4337) - This condition is probably inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000823-4` (score=0.4329) - This condition is generally not inherited but arises from gene mutations that occur in early blood-forming cells after conception. These alterations are called somatic mutations.

**Answer**:

The genetic changes related to congenital diaphragmatic hernia (CDH) include:

- A currently undetectable chromosomal microdeletion or microduplication (tiny loss or extra copy of genetic material) [medquad_ans_0001497-3].
- A mutation in a major gene important for diaphragm development [medquad_ans_0001497-3].
- Combined effects of multiple minor genetic mutations or variants (polygenic inheritance) [medquad_ans_0001497-3].
- Effects of gene-environment interactions (multifactorial inheritance) [medquad_ans_0001497-3].
- Effects of non-genetic factors such as epigenetic or teratogenic influences [medquad_ans_0001497-3].

Approximately 15%-20% of individuals with CDH have

**Supported citations**: `medquad_ans_0001497-3`
**Unsupported citations**: _none_

### Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.807, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004534-2` (score=0.5854) - What are the signs and symptoms of Omenn syndrome? Infants with Omenn syndrome typically present shortly after birth, usually by 3 months of age. This is similar to other types of severe combined immunodeficiency (SCI...
2. `medquad_ans_0004534-1` (score=0.5391) - Omenn syndrome is an autosomal recessive form of severe combined immunodeficiency (SCID) characterized by erythroderma (skin redness), desquamation (peeling skin), alopecia (hair loss), chronic diarrhea, failure to th...
3. `medquad_ans_0004534-3` (score=0.5211) - What causes Omenn syndrome? Omenn syndrome is a genetically heterogeneous condition (meaning that it may be caused by a number of different genes). While most cases are attributed to mutations in the RAG genes (RAG-1 ...
4. `medquad_ans_0004534-4` (score=0.5083) - How might Omenn syndrome be treated? The standard treatment for Omenn syndrome is bone marrow transplantation or cord blood stem cell transplantation. General care for any patient with severe combined immunodeficiency...
5. `medquad_ans_0001661-2` (score=0.4838) - What are the signs and symptoms of CREST syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for CREST syndrome. If the information is available, the table below includes how often...

**Answer**:

The provided context does not specify the cause of Omenn syndrome. While it mentions that Omenn syndrome is an autosomal recessive form of severe combined immunodeficiency (SCID) and lists associated symptoms, it does not detail the genetic or molecular cause. Therefore, based on the given information, the cause of Omenn syndrome cannot be determined.

[medquad_ans_0004534-1]

**Supported citations**: `medquad_ans_0004534-1`
**Unsupported citations**: _none_

### Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Scores**: context_relevance=0.375, groundedness=0.110, answer_relevance=0.575

**Top docs**:

1. `medquad_ans_0000785-4` (score=0.4931) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000305-4` (score=0.4928) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000430-4` (score=0.4928) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000424-4` (score=0.4928) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000349-4` (score=0.4928) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain information about autosomal recessive cerebellar ataxia type 1. Therefore, the answer cannot be determined from the given context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Scores**: context_relevance=0.750, groundedness=0.871, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000049-2__dup1` (score=0.5689) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
2. `medquad_ans_0000050-2__dup1` (score=0.5689) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
3. `medquad_ans_0000123-1__dup2` (score=0.5300) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Symptoms may include - A limp or paralyzed...
4. `medquad_ans_0000232-3__dup1` (score=0.5186) - With treatment, most people recover from pinched nerve. However, in some cases, the damage is irreversible.
5. `medquad_ans_0000049-4__dup1` (score=0.5078) - The NINDS conducts and supports research on injuries to the nervous system such as brachial plexus injuries. Much of this research is aimed at finding ways to prevent and treat these disorders.

**Answer**:

The treatments for brachial plexus injuries include physical therapy and, in some cases, surgery [medquad_ans_0000049-2\_\_dup1][medquad_ans_0000050-2__dup1][medquad_ans_0000123-1__dup2].

**Supported citations**: `medquad_ans_0000049-2__dup1`, `medquad_ans_0000050-2__dup1`, `medquad_ans_0000123-1__dup2`
**Unsupported citations**: _none_

### Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Scores**: context_relevance=0.833, groundedness=0.876, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000216-6` (score=0.5830) - Treatment for urine blockage depends on the cause and severity of the blockage. Hydronephrosis discovered before the baby is born rarely requires immediate action, especially if it is only on one side. The condition o...
2. `medquad_ans_0000216-7` (score=0.5631) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
3. `medquad_ans_0000216-4__dup1` (score=0.5370) - When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis. Hydronephrosis is the most common problem found during prenatal ...
4. `medquad_ans_0000216-3__dup1` (score=0.5340) - Before leaving the hospital, a baby with urine blockage may urinate only small amounts or may not urinate at all. As part of the routine newborn exam, the health care provider may feel an enlarged kidney or find a clo...
5. `medquad_ans_0000071-11` (score=0.5202) - Although everyone has some risk for UTIs, some people are more likely to get UTIs than others. These include people who have - spinal cord injuries or other nerve damage around the bladder - a blockage in the urinary ...

**Answer**:

Urine blockage in newborns is caused by a defect in the urinary tract that blocks the flow of urine, leading to urine backing up and causing the ureters to swell (hydroureter) and hydronephrosis [medquad_ans_0000216-4__dup1]. The specific cause of the defect may vary, but it is not related to the mother's eating, diet, or nutrition [medquad_ans_0000216-7].

**Supported citations**: `medquad_ans_0000216-4__dup1`, `medquad_ans_0000216-7`
**Unsupported citations**: _none_

### Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Scores**: context_relevance=0.444, groundedness=0.868, answer_relevance=0.356

**Top docs**:

1. `medquad_ans_0000989-2` (score=0.5620) - This condition affects an estimated 1 in 50,000 people.
2. `medquad_ans_0000359-2` (score=0.5604) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
3. `medquad_ans_0000934-2` (score=0.5588) - This condition affects fewer than 1 in 150,000 males and is very rare in females.
4. `medquad_ans_0000526-2` (score=0.5574) - Intranuclear rod myopathy is a rare disorder that has been identified in only a small number of individuals. Its exact prevalence is unknown.
5. `medquad_ans_0000650-2` (score=0.5548) - This condition occurs in an estimated 1 in 50,000 to 100,000 people.

**Answer**:

The exact prevalence of intranuclear rod myopathy is unknown. It has been identified in only a small number of individuals, and no specific prevalence rate is provided in the context [medquad_ans_0000526-2].

**Supported citations**: `medquad_ans_0000526-2`
**Unsupported citations**: _none_

### Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000036-11` (score=0.5906) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. There are two main types of high blood pressure: primary and secondary high blood pressure...
2. `medquad_ans_0000036-1__dup3` (score=0.5877) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. What Is Blood Pressure? Blood pressure is the force of blood pushing against the walls of ...
3. `medquad_ans_0000107-1__dup4` (score=0.5844) - High blood pressure, also called hypertension, usually has no symptoms. But it can cause serious problems such as stroke, heart failure, heart attack and kidney failure. If you cannot control your high blood pressure ...
4. `medquad_ans_0000036-12` (score=0.5755) - Abnormal blood pressure is higher than 120/80 mmHg. If either your systolic or diastolic blood pressure is higher than normal (120/80) but not high enough to be considered high blood pressure (140/90), you have pre-hy...
5. `medquad_ans_0000150-8` (score=0.5620) - - Blood pressure is the force of blood pushing against blood vessel walls as the heart pumps out blood, and high blood pressure, also called hypertension, is an increase in the amount of force that blood places on blo...

**Answer**:

High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. It is defined as the force of blood pushing against the walls of the blood vessels as the heart pumps blood. If blood pressure rises and stays high over time, it is called high blood pressure. High blood pressure is dangerous because it makes the heart work too hard and the high force of blood flow can harm arteries and organs such as the heart, kidneys, brain, and eyes [medquad_ans_0000036-1__dup3]. There are two main types: primary (or essential) high blood pressure, which is the most common type and develops over years as a person ages, and secondary high blood pressure, which is caused by another medical condition or use of certain medicines and usually resolves after the cause is treated or removed [medquad_ans_0000036-11].

**Supported citations**: `medquad_ans_0000036-1__dup3`, `medquad_ans_0000036-11`
**Unsupported citations**: _none_

### Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Scores**: context_relevance=1.000, groundedness=0.904, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0001051-4` (score=0.5293) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
2. `medquad_ans_0000342-4` (score=0.5291) - This condition is inherited in an autosomal dominant pattern from an affected parent; one copy of the altered gene in each cell is sufficient to cause the disorder.
3. `medquad_ans_0000286-4` (score=0.5290) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000682-4` (score=0.5290) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000721-4` (score=0.5290) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

Yes, X-linked dystonia-parkinsonism is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, and in males, one altered copy of the gene is sufficient to cause the condition. Females with one altered copy are typically carriers and may develop milder movement abnormalities, though this is less common [medquad_ans_0001051-4].

**Supported citations**: `medquad_ans_0001051-4`
**Unsupported citations**: _none_

### Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.539, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000120-2` (score=0.5626) - Beta-ketothiolase deficiency appears to be very rare. It is estimated to affect fewer than 1 in 1 million newborns.
2. `medquad_ans_0000120-3` (score=0.5322) - Mutations in the ACAT1 gene cause beta-ketothiolase deficiency. This gene provides instructions for making an enzyme that is found in the energy-producing centers within cells (mitochondria). This enzyme plays an esse...
3. `medquad_ans_0000120-1` (score=0.5244) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process keto...
4. `medquad_ans_0000702-1__dup1` (score=0.5108) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ket...
5. `medquad_ans_0000702-2__dup1` (score=0.4893) - What are the signs and symptoms of Beta ketothiolase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Beta ketothiolase deficiency. If the information is available, the ta...

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. Signs and symptoms typically appear between the ages of 6 months and 24 months and include episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, known as ketoacidotic attacks, may lead to coma and are frequently triggered by infections, periods without food (fasting), or increased intake of protein-rich foods. The condition is inherited in an autosomal recessive fashion and is caused by mutations in the ACAT1 gene, which provides instructions for making an enzyme found in the energy-producing centers within cells (mitochondria). This enzyme plays an essential role in breaking down proteins and fats from the diet

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 285 (`0000006_5-7`)

**Question**: What are the stages of Childhood Brain Stem Glioma ?

**Scores**: context_relevance=0.778, groundedness=0.954, answer_relevance=0.511

**Top docs**:

1. `medquad_ans_0000006_5-7` (score=0.4513) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the...
2. `medquad_ans_0000006_3-7` (score=0.4173) - Key Points - The grade of the tumor is used to plan cancer treatment. - Low-grade astrocytomas - High-grade astrocytomas - An MRI is done after surgery. The grade of the tumor is used to plan cancer treatment. Staging...
3. `medquad_ans_0000006_5-4` (score=0.4158) - The signs and symptoms of brain stem glioma are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain. - The size of the tumor and whether it has spread all thro...
4. `medquad_ans_0000006_5-1` (score=0.4147) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - T...
5. `medquad_ans_0000006_6-3` (score=0.4134) - Signs and symptoms of childhood CNS embryonal tumors or pineoblastomas depend on the child's age and where the tumor is. These and other signs and symptoms may be caused by childhood CNS embryonal tumors, pineoblastom...

**Answer**:

There is no standard staging system for childhood brain stem glioma [medquad_ans_0000006_5-7].

**Supported citations**: `medquad_ans_0000006_5-7`
**Unsupported citations**: _none_

### Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000266-1__dup2` (score=0.5490) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal c...
2. `medquad_ans_0004257-2` (score=0.4987) - What are the signs and symptoms of myelomeningocele? A baby born with a myelomeningocele may have a sac sticking out of the mid to lower back that the doctor cannot see through when shining a light behind it. Symptoms...
3. `medquad_ans_0000266-3__dup1` (score=0.4762) - With treatment, individuals with tethered spinal cord syndrome have a normal life expectancy. However, some neurological and motor impairments may not be fully correctable. Surgery soon after symptoms emerge appears t...
4. `medquad_ans_0005813-1` (score=0.4478) - What are the signs and symptoms of Split hand urinary anomalies spina bifida? The Human Phenotype Ontology provides the following list of signs and symptoms for Split hand urinary anomalies spina bifida. If the inform...
5. `medquad_ans_0000933-1` (score=0.4450) - Spina bifida is a condition in which the neural tube, a layer of cells that ultimately develops into the brain and spinal cord, fails to close completely during the first few weeks of embryonic development. As a resul...

**Answer**:

Tethered Spinal Cord Syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course. Symptoms in children may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is often linked to improper growth of the neural tube during fetal development and is closely associated with spina bifida. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. Delayed symptom presentation is related to the degree of strain on the spinal cord over time

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Scores**: context_relevance=0.250, groundedness=0.000, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000043_1-4` (score=0.4817) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The patient's age and general...
2. `medquad_ans_0000033_1-5` (score=0.4800) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - Whether the calcium level in the blood can be controlled...
3. `medquad_ans_0000014_2-5` (score=0.4787) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The type and size of the tumo...
4. `medquad_ans_0000222-3__dup1` (score=0.4762) - The prognosis for those with paresthesia depends on the severity of the sensations and the associated disorders.
5. `medquad_ans_0000034_1-5` (score=0.4714) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The location and size of the ...

**Answer**:

The provided context does not contain information about the outlook for Carpal Tunnel Syndrome. Therefore, based on the given context, it is not possible to determine the prognosis or outlook for Carpal Tunnel Syndrome. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.736, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000296-5` (score=0.6013) - These resources address the diagnosis or management of Donnai-Barrow syndrome: - Gene Review: Gene Review: Donnai-Barrow Syndrome - Genetic Testing Registry: Donnai Barrow syndrome - MedlinePlus Encyclopedia: Diaphrag...
2. `medquad_ans_0000130-2__dup1` (score=0.5166) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportiv...
3. `medquad_ans_0000297-5` (score=0.5137) - These resources address the diagnosis or management of Donohue syndrome: - Genetic Testing Registry: Leprechaunism syndrome These resources from MedlinePlus offer information about the diagnosis and management of vari...
4. `medquad_ans_0000749-5` (score=0.5113) - These resources address the diagnosis or management of Ohdo syndrome, SBBYS variant: - Gene Review: Gene Review: KAT6B-Related Disorders - Genetic Testing Registry: Young Simpson syndrome These resources from MedlineP...
5. `medquad_ans_0001103-5` (score=0.4930) - These resources address the diagnosis or management of 9q22.3 microdeletion: - Gene Review: Gene Review: 9q22.3 Microdeletion - Gene Review: Gene Review: Nevoid Basal Cell Carcinoma Syndrome - Genetic Testing Registry...

**Answer**:

The provided context does not specify any treatments for Donnai-Barrow syndrome. While resources for diagnosis and management are listed, including genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, palliative care, and specific information on conditions like diaphragmatic hernia and hearing loss, no explicit treatments for Donnai-Barrow syndrome are described [medquad_ans_0000296-5]. Therefore, based on the given information, the treatments for Donnai-Barrow syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000296-5`
**Unsupported citations**: _none_

### Example 289 (`0000811-1`)

**Question**: What is (are) porphyria ?

**Scores**: context_relevance=1.000, groundedness=0.990, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0001548-2` (score=0.5244) - What are the signs and symptoms of Congenital porphyria? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital porphyria. If the information is available, the table below includ...
2. `medquad_ans_0000189-1__dup3` (score=0.5217) - Porphyrias are rare disorders that affect mainly the skin or nervous system and may cause abdominal pain. These disorders are usually inherited, meaning they are caused by abnormalities in genes passed from parents to...
3. `medquad_ans_0000729-1__dup1` (score=0.5215) - Porphyrias are a group of genetic disorders caused by problems with how your body makes a substance called heme. Heme is found throughout the body, especially in your blood and bone marrow, where it carries oxygen. Th...
4. `medquad_ans_0000189-10` (score=0.5189) - - Porphyrias are rare disorders that affect mainly the skin or nervous system and may cause abdominal pain. - Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosyntheti...
5. `medquad_ans_0002164-1` (score=0.5175) - Erythropoietic protoporphyria is a type of porphyria. Porphyrias are caused by an abnormality in the heme production process. Heme is essential in enabling our blood cells to carry oxygen and in breaking down chemical...

**Answer**:

Porphyria are rare disorders that affect mainly the skin or nervous system and may cause abdominal pain. These disorders are usually inherited, meaning they are caused by abnormalities in genes passed from parents to children. When a person has a porphyria, cells fail to change body chemicals called porphyrins and porphyrin precursors into heme, the substance that gives blood its red color. The body makes heme mainly in the bone marrow and liver [medquad_ans_0000189-1__dup3].

**Supported citations**: `medquad_ans_0000189-1__dup3`
**Unsupported citations**: _none_

### Example 290 (`0000585-1__dup1`)

**Question**: What is (are) Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy ?

**Scores**: context_relevance=0.833, groundedness=0.981, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000585-1__dup1` (score=0.4930) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
2. `medquad_ans_0001900-1` (score=0.4879) - Dominant optic atrophy (DOA) is an inherited optic nerve disorder characterized by degeneration of the optic nerves. It typically starts during the first decade of life. Affected people usually develop moderate visual...
3. `medquad_ans_0000785-4` (score=0.4824) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000866-4` (score=0.4794) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000957-4` (score=0.4794) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. It is inherited in an autosomal recessive manner, meaning both copies of the gene in each cell have mutations, and the parents of an individual with the condition each carry one copy of the mutated gene but typically do not show signs and symptoms [medquad_ans_0000585-1__dup1].

**Supported citations**: `medquad_ans_0000585-1__dup1`
**Unsupported citations**: _none_

### Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Scores**: context_relevance=0.700, groundedness=0.893, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0000794-3` (score=0.5038) - Phosphoglycerate kinase deficiency is caused by mutations in the PGK1 gene. This gene provides instructions for making an enzyme called phosphoglycerate kinase, which is involved in a critical energy-producing process...
2. `medquad_ans_0000794-4` (score=0.4570) - This condition is inherited in an X-linked recessive pattern. The PGK1 gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of the g...
3. `medquad_ans_0000794-2` (score=0.4511) - Phosphoglycerate kinase deficiency appears to be a rare disorder. About 30 families with affected members have been reported in the scientific literature.
4. `medquad_ans_0000652-3` (score=0.4489) - Mutations in the MVK gene cause mevalonate kinase deficiency. The MVK gene provides instructions for making the mevalonate kinase enzyme. This enzyme is involved in the production of cholesterol, which is later conver...
5. `medquad_ans_0000984-3` (score=0.4401) - As the condition name suggests, mutations in the TK2 gene cause TK2-MDS. The TK2 gene provides instructions for making an enzyme called thymidine kinase 2 that functions within cell structures called mitochondria, whi...

**Answer**:

The genetic changes related to phosphoglycerate kinase deficiency are mutations in the PGK1 gene. These mutations reduce the activity of the phosphoglycerate kinase enzyme, which is involved in glycolysis and energy production in cells [medquad_ans_0000794-3]. The PGK1 gene is located on the X chromosome, and the condition is inherited in an X-linked recessive pattern [medquad_ans_0000794-4]. In males, one altered copy of the gene is sufficient to cause the condition, while in females, mutations in both copies are typically required to cause the disorder, though females with one altered copy may exhibit some features such as anemia [medquad_ans_0000794-4]. The specific effects of these mutations on enzyme activity may vary depending on the type of mutation and the cell type involved [medquad_ans_0000

**Supported citations**: `medquad_ans_0000794-3`, `medquad_ans_0000794-4`
**Unsupported citations**: _none_

### Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Scores**: context_relevance=0.700, groundedness=0.960, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000849-3` (score=0.4338) - Mutations in the BMPR2 gene are the most common genetic cause of pulmonary arterial hypertension. This gene plays a role in regulating the number of cells in certain tissues. Researchers suggest that a mutation in thi...
2. `medquad_ans_0000823-4` (score=0.4301) - This condition is generally not inherited but arises from gene mutations that occur in early blood-forming cells after conception. These alterations are called somatic mutations.
3. `medquad_ans_0000850-3` (score=0.4244) - The primary genetic cause of PVOD is mutations in the EIF2AK4 gene. Mutations in other genes may cause a small percentage of cases. Other suspected causes of PVOD include viral infection and exposure to toxic chemical...
4. `medquad_ans_0000109-4__dup2` (score=0.4151) - Pulmonary hypertension (PH) begins with inflammation and changes in the cells that line your pulmonary arteries. Other factors also can affect the pulmonary arteries and cause PH. For example, the condition may develo...
5. `medquad_ans_0000850-4` (score=0.4146) - When caused by mutations in the EIF2AK4 gene, PVOD is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recess...

**Answer**:

The primary genetic cause of pulmonary veno-occlusive disease (PVOD) is mutations in the EIF2AK4 gene. Mutations in other genes may cause a small percentage of cases. The protein produced from the EIF2AK4 gene helps cells respond appropriately to changes that could damage the cell, such as low levels of amino acids. Mutations in the EIF2AK4 gene involved in PVOD likely eliminate functional EIF2AK4 protein, although it is unknown how the absence of this protein's function leads to the pulmonary vessel abnormalities underlying PVOD [medquad_ans_0000850-3].

**Supported citations**: `medquad_ans_0000850-3`
**Unsupported citations**: _none_

### Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Scores**: context_relevance=0.750, groundedness=0.982, answer_relevance=0.350

**Top docs**:

1. `medquad_ans_0001006-1` (score=0.4643) - Tyrosinemia is a genetic disorder characterized by disruptions in the multistep process that breaks down the amino acid tyrosine, a building block of most proteins. If untreated, tyrosine and its byproducts build up i...
2. `medquad_ans_0001006-3` (score=0.4496) - Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. In the liver, enzymes break down tyrosine in a five step process, resulting in molecules that are either excreted by t...
3. `medquad_ans_0001005-3` (score=0.4463) - Mutations in the TH gene cause TH deficiency. The TH gene provides instructions for making the enzyme tyrosine hydroxylase, which is important for normal functioning of the nervous system. Tyrosine hydroxylase takes p...
4. `medquad_ans_0006262-1` (score=0.4461) - Tyrosinemia type 3 is a genetic disorder characterized by elevated blood levels of the amino acid tyrosine, a building block of most proteins. This condition is caused by a deficiency of the enzyme 4-hydroxyphenylpyru...
5. `medquad_ans_0006260-1` (score=0.4454) - Tyrosinemia type 1 is a genetic disorder characterized by elevated blood levels of the amino acid tyrosine, a building block of most proteins. This condition is caused by a shortage of the enzyme fumarylacetoacetate h...

**Answer**:

Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively [medquad_ans_0001006-3]. The FAH gene provides instructions for the fumarylacetoacetate hydrolase enzyme, which is responsible for the final step of tyrosine breakdown [medquad_ans_0001006-3].

**Supported citations**: `medquad_ans_0001006-3`
**Unsupported citations**: _none_

### Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Scores**: context_relevance=0.857, groundedness=0.480, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000004-6__dup2` (score=0.6022) - People with Graves disease have three treatment options: radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves disease in the United States. Graves disease ...
2. `medquad_ans_0000004-7__dup2` (score=0.6000) - Treatment for Graves disease can sometimes affect pregnancy. After treatment with surgery or radioactive iodine, TSI antibodies can still be present in the blood, even when thyroid levels are normal. If a pregnant wom...
3. `medquad_ans_0000436-5` (score=0.5907) - These resources address the diagnosis or management of Graves disease: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Graves disease 2 - Genetic Testing Registry: Graves disease 3 -...
4. `medquad_ans_0000209-2` (score=0.5706) - Treatment involves restoring normal levels of thyroid hormone and may include thyroid drugs, radioactive iodine, and sometimes partial or complete surgical removal of the thyroid.
5. `medquad_ans_0000008-8__dup2` (score=0.5591) - Health care providers treat hyperthyroidism with medications, radioiodine therapy, or thyroid surgery. The aim of treatment is to bring thyroid hormone levels to a normal state, thus preventing long-term complications...

**Answer**:

The treatments for Graves' disease include radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy involves taking radioactive iodine-131 by mouth, which is absorbed by the thyroid gland and gradually destroys thyroid cells, leading to reduced hormone production. This treatment may require multiple rounds and can take weeks or months to show results; it is not used in pregnant or breastfeeding women due to risks to the fetus or infant, and patients are advised to wait at least one year before becoming pregnant. Radioiodine therapy may worsen Graves' ophthalmopathy (GO) in some individuals, and steroid use may help prevent this complication. Almost everyone who receives radioiodine therapy eventually develops hypothyroidism and will need to take synthetic thyroid hormone replacement. Beta blockers are prescribed to reduce symptoms such as tremors, rapid heartbeat, and nervousness but do not stop thyroid hormone production. Anti-thyroid medications interfere with thyroid hormone production and are not

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Scores**: context_relevance=0.800, groundedness=0.288, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000545-1` (score=0.5479) - Juvenile Batten disease is an inherited disorder that primarily affects the nervous system. After a few years of normal development, children with this condition develop progressive vision loss, intellectual and motor...
2. `medquad_ans_0000039-1__dup2` (score=0.5429) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, ...
3. `medquad_ans_0000039-2__dup1` (score=0.5324) - As yet, no specific treatment is known that can halt or reverse the symptoms of Batten disease. However, seizures can sometimes be reduced or controlled with anticonvulsant drugs, and other medical problems can be tre...
4. `medquad_ans_0000039-3__dup1` (score=0.5165) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten dise...
5. `medquad_ans_0000545-3` (score=0.5155) - Most cases of juvenile Batten disease are caused by mutations in the CLN3 gene. This gene provides instructions for making a protein whose function is unknown. It is unclear how mutations in the CLN3 gene lead to the ...

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. It is characterized by a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. Vision loss is often an early sign, leading to suspicion during an eye exam, and may prompt referral to a neurologist. The disorder typically causes progressive problems with vision, movement, and thinking ability. It is one of a group of disorders known as neuronal ceroid lipofuscinoses (NCLs), which are distinguished by the age at which symptoms first appear. Some refer to the entire group of NCLs as Batten disease, while others limit the term to the juvenile form [medquad_ans_0000039-1__dup2, medquad_ans_0000545-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000039-1__dup2, medquad_ans_0000545-1`

### Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Scores**: context_relevance=1.000, groundedness=0.490, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002061-5` (score=0.4996) - How might Ehlers-Danlos syndrome progeroid type be treated? Individuals with Ehlers-Danlos Syndrome progeroid type can benefit from a variety of treatments depending on their symptoms. Affected children with weak musc...
2. `medquad_ans_0002061-2` (score=0.4903) - What are the signs and symptoms of Ehlers-Danlos syndrome, progeroid type? Ehlers-Danlos syndrome refers to a group of connective tissue disorders characterized by stretchy or kneadable skin, double jointedness, and d...
3. `medquad_ans_0002061-1` (score=0.4763) - Ehlers-Danlos syndrome progeroid type is a genetic disorder of the connective tissue, which is the material between the cells of the body that gives tissues form and strength. The disorder primarily affects the skin, ...
4. `medquad_ans_0002061-3` (score=0.4562) - What causes Ehlers-Danlos syndrome progeroid type? Ehlers-Danlos syndrome progeroid type is caused by changes (mutations) in both of an individual's copies of the B4GALT7 gene, which is located on chromosome 5. This g...
5. `medquad_ans_0002061-4` (score=0.4552) - How is Ehlers-Danlos syndrome progeroid type inherited? Ehlers-Danlos syndrome progeroid type is inherited in an autosomal recessive pattern. This means that an individual must have two non-functional copies of the B4...

**Answer**:

Ehlers-Danlos syndrome, progeroid type is a group of connective tissue disorders characterized by stretchy or kneadable skin, double jointedness, and delayed healing of skin wounds. Individuals with this type also have thin curly hair, sparse eyebrows and eyelashes, loose elastic skin on the face, and may have uneven facial features. Although the term "progeroid" suggests an appearance similar to old age, individuals do not experience premature aging or a shortened lifespan. Other symptoms include poor muscle tone, fragile bones due to low bone mineral density, abnormal teeth, gum infections, delayed growth leading to short stature (less than 152 cm), and mild intellectual or learning disabilities. The condition is associated with a high frequency of symptoms such as muscular hypotonia (90%), thin skin (90%), short stature (90%), and abnormality of the aortic and pulmonary valves (90%). It is inherited

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.980, answer_relevance=0.480

**Top docs**:

1. `medquad_ans_0000416-3` (score=0.5298) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
2. `medquad_ans_0000442-3` (score=0.4971) - Mutations in the GAMT gene cause guanidinoacetate methyltransferase deficiency. The GAMT gene provides instructions for making the enzyme guanidinoacetate methyltransferase. This enzyme participates in the two-step pr...
3. `medquad_ans_0000066-3` (score=0.4834) - Mutations in the GATM gene cause arginine:glycine amidinotransferase deficiency. The GATM gene provides instructions for making the enzyme arginine:glycine amidinotransferase. This enzyme participates in the two-step ...
4. `medquad_ans_0000416-2` (score=0.4717) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
5. `medquad_ans_0000112-4` (score=0.4663) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. All reported cases have resulted from new mutations in the gene,...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the metabolism of the amino acid histidine and in the production of folate [medquad_ans_0000416-3].

**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

### Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001169-2` (score=0.5891) - What are the signs and symptoms of Chondrodysplasia punctata 1, X-linked recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 1, X-linked recessive. I...
2. `medquad_ans_0001170-2` (score=0.5733) - What are the signs and symptoms of Chondrodysplasia punctata 2 X-linked dominant? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 2 X-linked dominant. If th...
3. `medquad_ans_0005365-1` (score=0.4995) - What are the signs and symptoms of Rhizomelic chondrodysplasia punctata type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Rhizomelic chondrodysplasia punctata type 3. If the in...
4. `medquad_ans_0001047-1` (score=0.4933) - X-linked chondrodysplasia punctata 1 is a disorder of cartilage and bone development that occurs almost exclusively in males. Chondrodysplasia punctata is an abnormality that appears on x-rays as spots (stippling) nea...
5. `medquad_ans_0000043-2__dup1` (score=0.4889) - What are the signs and symptoms of Achondrogenesis type 1A? The Human Phenotype Ontology provides the following list of signs and symptoms for Achondrogenesis type 1A. If the information is available, the table below ...

**Answer**:

The symptoms of Chondrodysplasia punctata 1, X-linked recessive include abnormality of the vertebral column, anosmia, cataract, depressed nasal bridge, epiphyseal stippling, hearing impairment, hypogonadism, ichthyosis, microcephaly, short distal phalanx of finger, short nasal septum, short nose, short stature, and X-linked recessive inheritance [medquad_ans_0001169-2].

**Supported citations**: `medquad_ans_0001169-2`
**Unsupported citations**: _none_

### Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Scores**: context_relevance=0.857, groundedness=0.980, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000132-10` (score=0.5353) - A person with Whipple disease and malabsorption may need - a diet high in calories and protein - vitamins - nutritional supplements
2. `medquad_ans_0000873-5` (score=0.5070) - These resources address the diagnosis or management of retroperitoneal fibrosis: - Johns Hopkins Medicine These resources from MedlinePlus offer information about the diagnosis and management of various health conditi...
3. `medquad_ans_0000165-2__dup1` (score=0.5066) - The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfa...
4. `medquad_ans_0000137-2__dup2` (score=0.5011) - There is no cure for GSS, nor are there any known treatments to slow progression of the disease. Current therapies are aimed at alleviating symptoms and making the patient as comfortable as possible.
5. `medquad_ans_0000132-9` (score=0.4996) - Experts have not yet found a way to prevent Whipple disease.

**Answer**:

The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfadizine or solfamethoxazole can treat neurological symptoms. Relapsing neurologic Whipple's disease, marked by bouts of worsening of symptoms, is sometimes treated with a combination of antibiotics and weekly injections of interferon gamma, a substance made by the body that activates the immune system [medquad_ans_0000165-2__dup1].

**Supported citations**: `medquad_ans_0000165-2__dup1`
**Unsupported citations**: _none_

### Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Scores**: context_relevance=0.714, groundedness=0.976, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000676-4__dup1` (score=0.5770) - How might Bell's palsy be treated?
2. `medquad_ans_0000042-2__dup2` (score=0.4956) - Steroids such as prednisone -- used to reduce inflammation and swelling -- are an effective treatment for Bell's palsy. Antiviral drugs may have some benefit in shortening the course of the disease. Analgesics such as...
3. `medquad_ans_0000121-2__dup1` (score=0.4938) - There is no treatment for Todd's paralysis. Individuals must rest as comfortably as possible until the paralysis disappears.
4. `medquad_ans_0000257-3__dup1` (score=0.4930) - Recovery depends upon how quickly treatment is received and how severely the body is compromised. Paralysis may persist for many weeks or be permanent. Most individuals have a good chance of recovery.
5. `medquad_ans_0000676-1__dup1` (score=0.4832) - Bell's palsy is a form of temporary facial paralysis which results from damage or trauma to one of the facial nerves. This disorder is characterized by the sudden onset of facial paralysis that often affects just one ...

**Answer**:

Steroids such as prednisone are an effective treatment for Bell's palsy as they reduce inflammation and swelling. Antiviral drugs like acyclovir may have some benefit in shortening the course of the disease. Analgesics such as aspirin, acetaminophen, or ibuprofen may relieve pain. Keeping the eye moist and protecting it from debris and injury, especially at night, is important; lubricating eye drops can help. Other therapies such as physical therapy, facial massage, or acupuncture may provide a potential small improvement in facial nerve function and pain [medquad_ans_0000042-2__dup2].

**Supported citations**: `medquad_ans_0000042-2__dup2`
**Unsupported citations**: _none_
