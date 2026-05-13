# Generation Evaluation

## Summary

- `avg_answer_chars`: 537.4800
- `avg_answer_relevance`: 0.7898
- `avg_citation_presence_rate`: 0.8633
- `avg_context_relevance`: 0.9312
- `avg_empty_answer`: 0.0000
- `avg_end_to_end_latency_ms`: 2981.8532
- `avg_generation_latency_ms`: 2925.3982
- `avg_groundedness`: 0.7982
- `avg_insufficient_context`: 0.0233
- `avg_num_retrieved_docs`: 5.0000
- `avg_rerank_latency_ms`: 0.0000
- `avg_retrieval_latency_ms`: 56.2556
- `avg_supported_citation_rate`: 0.8128
- `avg_unsupported_citation_rate`: 0.0506
- `num_examples`: 300
- `reranker_enabled_rate`: 0.0000

## Examples

### Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.792, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0006511-1` (score=30.5608) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the informatio...
2. `medquad_ans_0001059-5` (score=29.7870) - These resources address the diagnosis or management of XLP: - Children's Hospital of Philadelphia - Gene Review: Gene Review: Lymphoproliferative Disease, X-Linked - Genetic Testing Registry: Lymphoproliferative syndr...
3. `medquad_ans_0006510-1` (score=29.6272) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the informatio...
4. `medquad_ans_0006509-2` (score=27.9971) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is...
5. `medquad_ans_0000088-5` (score=25.0518) - These resources address the diagnosis or management of ALPS: - Gene Review: Gene Review: Autoimmune Lymphoproliferative Syndrome - Genetic Testing Registry: Autoimmune lymphoproliferative syndrome - Genetic Testing Re...

**Answer**:

The provided context does not include the signs and symptoms of X-linked lymphoproliferative syndrome 1. [medquad_ans_0006510-1] mentions that the Human Phenotype Ontology provides a list of signs and symptoms for X-linked lymphoproliferative syndrome 1, but the specific symptoms are not detailed in the given text. Therefore, based on the available information, the symptoms cannot be identified.

**Supported citations**: `medquad_ans_0006510-1`
**Unsupported citations**: _none_

### Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000312-1` (score=28.8231) - Early-onset primary dystonia is a condition characterized by progressive problems with movement, typically beginning in childhood. Dystonia is a movement disorder that involves involuntary tensing of the muscles (musc...
2. `medquad_ans_0000312-4` (score=25.4009) - Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, which means one of the two copies of the gene is altered in each cell. Many people who have a mutation in this gene are not affected by the d...
3. `medquad_ans_0000312-5` (score=23.8604) - These resources address the diagnosis or management of early-onset primary dystonia: - Gene Review: Gene Review: DYT1 Early-Onset Primary Dystonia - Genetic Testing Registry: Dystonia 1 - MedlinePlus Encyclopedia: Mov...
4. `medquad_ans_0000312-2` (score=23.3436) - Early-onset primary dystonia is among the most common forms of childhood dystonia. This disorder occurs most frequently in people of Ashkenazi (central and eastern European) Jewish heritage, affecting 1 in 3,000 to 9,...
5. `medquad_ans_0000312-3` (score=21.4311) - A particular mutation in the TOR1A gene (also known as DYT1) is responsible for most cases of early-onset primary dystonia. The TOR1A gene provides instructions for making a protein called torsinA. Although little is ...

**Answer**:

Yes, early-onset primary dystonia is inherited. Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, and everyone who has been diagnosed with early-onset primary dystonia has inherited a TOR1A mutation from one parent [medquad_ans_0000312-4].

**Supported citations**: `medquad_ans_0000312-4`
**Unsupported citations**: _none_

### Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Scores**: context_relevance=0.857, groundedness=0.758, answer_relevance=0.971

**Top docs**:

1. `medquad_ans_0000096-2__dup2` (score=19.1534) - Acute diarrhea is usually caused by a bacterial, viral, or parasitic infection. Chronic diarrhea is usually related to a functional disorder such as irritable bowel syndrome or an intestinal disease such as Crohns dis...
2. `medquad_ans_0001853-3` (score=19.0557) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
3. `medquad_ans_0000038-2__dup3` (score=18.8074) - People can get infected with Babesia parasites in several ways: - The main way is through the bite of an infected tick—during outdoor activities in areas where babesiosis is found (see below). - A less common way is b...
4. `medquad_ans_0000689-1__dup2` (score=18.4068) - Parasites are living things that use other living things - like your body - for food and a place to live. You can get them from contaminated food or water, a bug bite, or sexual contact. Some parasitic diseases are ea...
5. `medquad_ans_0000344-1__dup2` (score=18.2076) - Baylisascaris worms are intestinal parasites found in a wide variety of animals. Different species of Baylisascaris are associated with different animal hosts. For example, Baylisascaris procyonis is found in raccoons...

**Answer**:

The provided context does not mention Taeniasis or who is at risk for parasites causing Taeniasis. Therefore, the cause or risk group for Taeniasis cannot be determined from the given information. [medquad_ans_0000096-2__dup2]

**Supported citations**: `medquad_ans_0000096-2__dup2`
**Unsupported citations**: _none_

### Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.736, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000029-1` (score=27.8522) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. Howeve...
2. `medquad_ans_0000194-2__dup1` (score=27.2215) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
3. `medquad_ans_0000197-2__dup1` (score=26.7667) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 5. If the information is availab...
4. `medquad_ans_0000195-2__dup1` (score=26.4821) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 3. If the information is availab...
5. `medquad_ans_0000196-2__dup1` (score=25.9636) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 4. If the information is availab...

**Answer**:

The provided context does not mention Aicardi-Goutieres syndrome type 2 or its specific symptoms. Therefore, it is not possible to answer the question based on the given information. [medquad_ans_0000029-1]

**Supported citations**: `medquad_ans_0000029-1`
**Unsupported citations**: _none_

### Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.942, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003421-1` (score=27.0221) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
3. `medquad_ans_0005488-1` (score=20.7288) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
4. `medquad_ans_0006607-2` (score=20.5187) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
5. `medquad_ans_0002946-1` (score=20.4979) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...

**Answer**:

The symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

### Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.908, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0002788-1` (score=19.0406) - Hemophilia B is a bleeding disorder that slows the blood clotting process. People with this disorder experience prolonged bleeding or oozing following an injury or surgery. In severe cases of hemophilia, heavy bleedin...
2. `medquad_ans_0002786-1` (score=18.9607) - Hemophilia is a bleeding disorder that slows the blood clotting process. People with this disorder experience prolonged bleeding following an injury, surgery, or having a tooth pulled. In severe cases, heavy bleeding ...
3. `medquad_ans_0000069-1__dup3` (score=18.9359) - Espaol Hemophilia (heem-o-FILL-ee-ah) is a rare bleeding disorder in which the blood doesn't clot normally. If you have hemophilia, you may bleed for a longer time than others after an injury. You also may bleed insid...
4. `medquad_ans_0000450-5` (score=17.9808) - These resources address the diagnosis or management of hemophilia: - Gene Review: Gene Review: Hemophilia A - Gene Review: Gene Review: Hemophilia B - Genetic Testing Registry: HEMOPHILIA B(M) - Genetic Testing Regist...
5. `medquad_ans_0000450-3` (score=17.7953) - Changes in the F8 gene are responsible for hemophilia A, while mutations in the F9 gene cause hemophilia B. The F8 gene provides instructions for making a protein called coagulation factor VIII. A related protein, coa...

**Answer**:

Yes, hemophilia is inherited. It is inherited in an X-linked recessive pattern and is caused by mutations in the F9 gene [medquad_ans_0002788-1]. Additionally, hemophilia usually is inherited, meaning the disorder is passed from parents to children through genes [medquad_ans_0000069-1__dup3].

**Supported citations**: `medquad_ans_0002788-1`, `medquad_ans_0000069-1__dup3`
**Unsupported citations**: _none_

### Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.970, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003284-1` (score=24.7289) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often t...
2. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
3. `medquad_ans_0005488-1` (score=20.7288) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
4. `medquad_ans_0006607-2` (score=20.5187) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
5. `medquad_ans_0002946-1` (score=20.4979) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...

**Answer**:

The symptoms of IVIC syndrome include hearing impairment (90%), limitation of joint mobility (90%), short stature (90%), strabismus (90%), abnormal dermatoglyphics (50%), aplasia/hypoplasia of the thumb (50%), radioulnar synostosis (50%), scoliosis (50%), synostosis of carpal bones (50%), triphalangeal thumb (50%), abnormality of the clavicle (7.5%), arrhythmia (7.5%), leukocytosis (7.5%), preaxial hand polydactyly (7.5%), thrombocytopenia (7.5%), and urogenital fistula (7.5%) [medquad_ans_0003284-1].

**Supported citations**: `medquad_ans_0003284-1`
**Unsupported citations**: _none_

### Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Scores**: context_relevance=0.833, groundedness=0.550, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000775-1` (score=41.9194) - Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. The disorder affects red blood cells (erythrocytes), which carry oxygen; white bloo...
2. `medquad_ans_0000775-5` (score=38.1423) - These resources address the diagnosis or management of paroxysmal nocturnal hemoglobinuria: - Duke University School of Medicine: Hemostasis & Thrombosis Center - Genetic Testing Registry: Paroxysmal nocturnal hemoglo...
3. `medquad_ans_0000775-2` (score=37.3679) - Paroxysmal nocturnal hemoglobinuria is a rare disorder, estimated to affect between 1 and 5 per million people.
4. `medquad_ans_0000775-3` (score=34.7292) - Mutations in the PIGA gene cause paroxysmal nocturnal hemoglobinuria. The PIGA gene provides instructions for making a protein called phosphatidylinositol glycan class A. This protein takes part in a series of steps t...
5. `medquad_ans_0000006-4__dup2` (score=25.2920) - Lower than normal numbers of red blood cells, white blood cells, and platelets cause most of the signs and symptoms of aplastic anemia. Signs and Symptoms of Low Blood Cell Counts Red Blood Cells The most common sympt...

**Answer**:

Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. It affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes), which protect the body from infection; and platelets (thrombocytes), which are involved in blood clotting. The disorder affects both sexes equally and can occur at any age, although it is most often diagnosed in young adulthood. People with paroxysmal nocturnal hemoglobinuria experience sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses on the body, such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to hemoglobinuria—dark-colored urine due to the presence of hemoglobin. In many, but not all cases, hemoglobinuria is most noticeable in the morning. The premature destruction

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000037_3-1` (score=22.2072) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...
2. `medquad_ans_0000037_4-1` (score=21.7740) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...
3. `medquad_ans_0000010-2__dup3` (score=21.4237) - Scientists don't know exactly what causes colorectal cancer, but they have been able to identify some risk factors for the disease. A risk factor is anything that increases your chances of getting a disease. Studies s...
4. `medquad_ans_0000037_3-2` (score=21.0036) - Key Points - Avoiding risk factors and increasing protective factors may help prevent cancer. - The following risk factors increase the risk of colorectal cancer: - Age - Family history of colorectal cancer - Personal...
5. `medquad_ans_0000010-1__dup5` (score=20.9923) - How Tumors Form The body is made up of many types of cells. Normally, cells grow, divide, and produce more cells as needed to keep the body healthy and functioning properly. Sometimes, however, the process goes wrong ...

**Answer**:

Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. [medquad_ans_0000037_3-1]

**Supported citations**: `medquad_ans_0000037_3-1`
**Unsupported citations**: _none_

### Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Scores**: context_relevance=0.833, groundedness=0.550, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000544-5` (score=41.4591) - These resources address the diagnosis or management of junctional epidermolysis bullosa: - Epidermolysis Bullosa Center, Cincinnati Children's Hospital Medical Center - Gene Review: Gene Review: Junctional Epidermolys...
2. `medquad_ans_0002121-1` (score=35.2365) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main ty...
3. `medquad_ans_0000544-2` (score=32.4339) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.
4. `medquad_ans_0000544-1` (score=31.7736) - Junctional epidermolysis bullosa (JEB) is one of the major forms of epidermolysis bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. Blisters and skin erosions form in...
5. `medquad_ans_0002121-2` (score=30.9450) - How is epidermolysis bullosa inherited? Inherited epidermolysis bullosa (EB) may follow either an autosomal dominant or autosomal recessive inheritance pattern, depending on the type and subtype of inherited EB in the...

**Answer**:

Junctional epidermolysis bullosa (JEB) is one of the major forms of epidermolysis bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. Blisters and skin erosions form in response to minor injury or friction, such as rubbing or scratching. Researchers classify JEB into two main types: Herlitz JEB and non-Herlitz JEB. Although the types differ in severity, their features overlap significantly, and they can be caused by mutations in the same genes. Herlitz JEB is the more severe form of the condition. From birth or early infancy, affected individuals have blistering over large regions of the body. Blistering also affects the mucous membranes, such as the moist lining of the mouth and digestive tract, which can make it difficult to eat and digest food. As a result, many affected children have chronic malnutrition

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Scores**: context_relevance=0.700, groundedness=0.924, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0000668-3` (score=43.5792) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same ...
2. `medquad_ans_0000668-5` (score=42.8441) - These resources address the diagnosis or management of molybdenum cofactor deficiency: - Genetic Testing Registry: Combined molybdoflavoprotein enzyme deficiency - Genetic Testing Registry: Molybdenum cofactor deficie...
3. `medquad_ans_0000668-1` (score=41.3611) - Molybdenum cofactor deficiency is a rare condition characterized by brain dysfunction (encephalopathy) that worsens over time. Babies with this condition appear normal at birth, but within a week they have difficulty ...
4. `medquad_ans_0000668-2` (score=35.8795) - Molybdenum cofactor deficiency is a rare condition that is estimated to occur in 1 in 100,000 to 200,000 newborns worldwide. More than 100 cases have been reported in the medical literature, although it is thought tha...
5. `medquad_ans_0000668-4` (score=34.9840) - Molybdenum cofactor deficiency has an autosomal recessive pattern of inheritance, which means both copies of the gene in each cell have mutations. An affected individual usually inherits one altered copy of the gene f...

**Answer**:

The genetic changes related to molybdenum cofactor deficiency are mutations in the MOCS1, MOCS2, or GPHN gene. Mutations in the MOCS1 gene cause type A, mutations in the MOCS2 gene cause type B, and mutations in the GPHN gene cause type C [medquad_ans_0000668-3]. These mutations reduce or eliminate the function of the associated protein, impairing molybdenum cofactor biosynthesis [medquad_ans_0000668-3].

**Supported citations**: `medquad_ans_0000668-3`
**Unsupported citations**: _none_

### Example 12 (`0000756-1`)

**Question**: What is (are) ornithine translocase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.448, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000756-1` (score=37.6379) - Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The...
2. `medquad_ans_0000756-3` (score=34.3186) - Mutations in the SLC25A15 gene cause ornithine translocase deficiency. Ornithine translocase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions tha...
3. `medquad_ans_0000756-2` (score=32.1250) - Ornithine translocase deficiency is a very rare disorder. Fewer than 100 affected individuals have been reported worldwide.
4. `medquad_ans_0004582-1` (score=25.2738) - What are the signs and symptoms of Ornithine translocase deficiency syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Ornithine translocase deficiency syndrome. If the inform...
5. `medquad_ans_0000755-1` (score=23.4571) - Ornithine transcarbamylase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high...

**Answer**:

Ornithine translocase deficiency is an inherited disorder caused by mutations in the SLC25A15 gene, which results in a defective mitochondrial ornithine transporter. This transporter is responsible for moving ornithine across the inner membrane of mitochondria to the mitochondrial matrix, where it participates in the urea cycle. When the transporter is unstable or misshapen, ornithine cannot reach the matrix, leading to an interruption of the urea cycle. This causes ammonia to accumulate in the blood, which is toxic, especially to the nervous system. The disorder belongs to a class of genetic diseases known as urea cycle disorders. Symptoms vary in severity and age of onset, with infants often presenting as lethargy, poor feeding, breathing or temperature regulation issues, seizures, or coma, while later-onset cases are usually less severe and may include intolerance to high-protein foods, episodes of vomiting, lethargy, atax

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Scores**: context_relevance=0.857, groundedness=0.966, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000362-1__dup2` (score=25.7061) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
2. `medquad_ans_0000045-28` (score=16.7974) - The general number for Medicare is 1-800-Medicare (1-800-633-4227). TTY users should call 1-877-486-2048. You can also visit http://www.medicare.gov. The "Medicare & You" handbook is mailed out to all Medicare enrolle...
3. `medquad_ans_0000045-21` (score=15.3885) - Nursing home care can be very expensive. Medicare generally doesn't cover nursing home care. There are many ways people can pay for nursing home care. For example, they can use their own money, they may be able to get...
4. `medquad_ans_0000042-18` (score=15.2143) - Many agencies and organizations in the community provide assistance and information to people who have low vision and to their families and caregivers. State agencies for the blind and visually impaired can make refer...
5. `medquad_ans_0000036-16` (score=15.0171) - - Diabetes management and treatment is expensive. Many people who have diabetes need help paying for their care. For those who qualify, a variety of government and nongovernment programs can help cover health care exp...

**Answer**:

Yes, information about financial assistance is available. If you don't have health insurance or need help with costs that aren't covered, financial assistance might be available through certain government programs and nonprofit organizations. You can also discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2].

**Supported citations**: `medquad_ans_0000362-1__dup2`
**Unsupported citations**: _none_

### Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0004105-2` (score=30.5424) - What are the symptoms of Mollaret meningitis? The symptoms of Mollaret meningitis are the same as those found in other types of meningitis. In Mollaret meningitis, however, the symptoms are recurring and are often acc...
2. `medquad_ans_0004105-1` (score=29.8369) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About hal...
3. `medquad_ans_0000115-1__dup1` (score=14.5557) - Meningitis is an infection of the meninges, the membranes that surround the brain and spinal cord. Encephalitis is inflammation of the brain itself. Causes of encephalitis and meningitis include viruses, bacteria, fun...
4. `medquad_ans_0000598-1__dup2` (score=14.0424) - Meningitis is inflammation of the thin tissue that surrounds the brain and spinal cord, called the meninges. There are several types of meningitis. The most common is viral meningitis, which you get when a virus enter...
5. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches, separated by weeks or months of no symptoms. About half of affected individuals may also experience long-term abnormalities of the nervous system that come and go, such as seizures, double vision, abnormal reflexes, paralysis of a cranial nerve (paresis), hallucinations, or coma. The exact cause remains unknown, although recent data suggests that herpes simplex virus (HSV-2 and, less frequently, HSV-1) may cause some, if not most cases. Other potential causes include trauma and viral infections other than herpes simplex [medquad_ans_0004105-1]. The symptoms of Mollaret meningitis are similar to those of other types of meningitis and include high fever, severe headache, nausea, vomiting, stiff neck, photophobia, and

**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

### Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.914, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003319-3` (score=41.5537) - What causes juvenile amyotrophic lateral sclerosis? Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases the underlying gene abnormality...
2. `medquad_ans_0000052-5` (score=35.4198) - These resources address the diagnosis or management of amyotrophic lateral sclerosis: - Gene Review: Gene Review: ALS2-Related Disorders - Gene Review: Gene Review: Amyotrophic Lateral Sclerosis Overview - Gene Review...
3. `medquad_ans_0003319-1` (score=32.1136) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. U...
4. `medquad_ans_0003331-3` (score=29.9182) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...
5. `medquad_ans_0003319-4` (score=28.1941) - How might juvenile amyotrophic lateral sclerosis be treated? Treatments and therapies are available to relieve symptoms and improve the quality of life of people with juvenile ALS. Medications, such as those that redu...

**Answer**:

Juvenile amyotrophic lateral sclerosis is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases, the underlying gene abnormality cannot be determined. Juvenile ALS may be inherited in an autosomal dominant or autosomal recessive fashion [medquad_ans_0003319-3]. Additionally, mutations in the ALS2 gene are specifically associated with juvenile primary lateral sclerosis, which is closely related to juvenile ALS [medquad_ans_0003331-3].

**Supported citations**: `medquad_ans_0003319-3`, `medquad_ans_0003331-3`
**Unsupported citations**: _none_

### Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Scores**: context_relevance=0.778, groundedness=0.725, answer_relevance=0.689

**Top docs**:

1. `medquad_ans_0000038-22` (score=37.5448) - During your next health care visit, talk to your provider about your test results and how to manage your kidney disease. Below is a list of questions you may want to ask. Add any questions you think are missing, and m...
2. `medquad_ans_0000035-9` (score=34.0837) - People can manage their diabetes with meal planning, physical activity, and if needed, medications. More information about taking care of type 1 or type 2 diabetes is provided in the NIDDK health topics: - What I need...
3. `medquad_ans_0000193-6` (score=33.6352) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need t...
4. `medquad_ans_0000036-19` (score=32.7012) - If you are diagnosed with high blood pressure, here are questions to ask your doctor. - Your Blood Pressure Numbers - What is my blood pressure reading in numbers? - What is my goal blood pressure? - Is my blood press...
5. `medquad_ans_0000080-1__dup3` (score=31.3568) - If you have diabetes,\* the best time to control your blood glucose, also called blood sugar, is before you get pregnant. High blood glucose levels can be harmful to your baby during the first weeks of pregnancyeven be...

**Answer**:

The provided context does not contain any information about erectile dysfunction or what to do regarding it. Therefore, a specific answer cannot be generated from the given context. [medquad_ans_0000038-22]

**Supported citations**: `medquad_ans_0000038-22`
**Unsupported citations**: _none_

### Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.924, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0006267-3` (score=39.2164) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
2. `medquad_ans_0006267-1` (score=36.9189) - Ullrich congenital muscular dystrophy is a condition that mainly affects skeletal muscles (the muscles used for movement). Affected individuals show severe muscle weakness soon after birth, develop stiff joints (contr...
3. `medquad_ans_0000202-5` (score=35.8380) - These resources address the diagnosis or management of collagen VI-related myopathy: - Gene Review: Gene Review: Collagen Type VI-Related Disorders - Genetic Testing Registry: Bethlem myopathy - Genetic Testing Regist...
4. `medquad_ans_0000202-1` (score=35.3703) - Collagen VI-related myopathy is a group of disorders that affect skeletal muscles (which are the muscles used for movement) and connective tissue (which provides strength and flexibility to the skin, joints, and other...
5. `medquad_ans_0000202-4` (score=32.9815) - Collagen VI-related myopathy can be inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Bethlem myopathy is typically inherited in an ...

**Answer**:

The treatments for Ullrich congenital muscular dystrophy include physical therapy focusing on early mobilization, regular stretching, and splinting as part of supportive care. Respiratory support and nighttime ventilation may become necessary in the first or second decade of life. Prevention of chest infections can be achieved with antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports suggest that cyclosporin A may benefit individuals with Ullrich congenital muscular dystrophy, although more studies are needed to confirm its effectiveness [medquad_ans_0006267-3].

**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

### Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Scores**: context_relevance=0.875, groundedness=0.927, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0004005-3` (score=59.0231) - What causes microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1)? Microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) has been shown to be caused by mutations in the RNU4ATAC gene.
2. `medquad_ans_0004005-5` (score=49.9288) - How might microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) be treated? At this time there are no specific treatments for MOPD1. Treatment is generally supportive. The prognosis is poor for affected ind...
3. `medquad_ans_0004006-1` (score=45.8110) - Microcephalic osteodysplastic primordial dwarfism type 2 (MOPD2) is a condition characterized by short stature (dwarfism), skeletal abnormalities and an unusually small head size (microcephaly). Other signs and sympto...
4. `medquad_ans_0000653-5` (score=45.1575) - These resources address the diagnosis or management of MOPDII: - Genetic Testing Registry: Microcephalic osteodysplastic primordial dwarfism type 2 These resources from MedlinePlus offer information about the diagnosi...
5. `medquad_ans_0004005-4` (score=42.7773) - How is microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) inherited? MOPD1 is thought to be inherited in an autosomal recessive manner. This means that affected individuals have abnormal gene changes (mu...

**Answer**:

Yes, microcephalic osteodysplastic primordial dwarfism type 2 (MOPD2) is inherited in an autosomal recessive manner [medquad_ans_0004006-1].

**Supported citations**: `medquad_ans_0004006-1`
**Unsupported citations**: _none_

### Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.788, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000742-3` (score=22.1056) - Ochoa syndrome can be caused by mutations in the HPSE2 gene. This gene provides instructions for making a protein called heparanase 2. The function of this protein is not well understood. Mutations in the HPSE2 gene t...
2. `medquad_ans_0000742-1` (score=21.4615) - Ochoa syndrome is a disorder characterized by urinary problems and unusual facial expressions. The urinary problems associated with Ochoa syndrome typically become apparent in early childhood or adolescence. People wi...
3. `medquad_ans_0000742-2` (score=19.9121) - Ochoa syndrome is a rare disorder. About 150 cases have been reported in the medical literature.
4. `medquad_ans_0000742-5` (score=17.0144) - These resources address the diagnosis or management of Ochoa syndrome: - Gene Review: Gene Review: Urofacial Syndrome - Genetic Testing Registry: Ochoa syndrome - National Institute of Diabetes and Digestive and Kidne...
5. `medquad_ans_0001853-3` (score=14.3676) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

The context does not provide information on whether Ochoa syndrome is inherited. While it mentions that mutations in the HPSE2 gene can cause the syndrome and that some individuals do not have such mutations, it does not specify the inheritance pattern. Therefore, based on the provided information, the inheritance of Ochoa syndrome cannot be determined. [medquad_ans_0000742-3]

**Supported citations**: `medquad_ans_0000742-3`
**Unsupported citations**: _none_

### Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.665, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000531-1` (score=41.9486) - Isodicentric chromosome 15 syndrome is a developmental disorder with a broad spectrum of features. The signs and symptoms vary among affected individuals. Poor muscle tone is commonly seen in individuals with isodicen...
2. `medquad_ans_0003266-1` (score=39.5061) - Isodicentric chromosome 15 syndrome is a chromosome abnormality that affects many different parts of the body. As the name suggests, people with this condition have an extra chromosome (called an isodicentric chromoso...
3. `medquad_ans_0000531-3` (score=39.2819) - Isodicentric chromosome 15 syndrome results from the presence of an abnormal extra chromosome, called an isodicentric chromosome 15, in each cell. An isodicentric chromosome contains mirror-image segments of genetic m...
4. `medquad_ans_0000531-4` (score=30.4417) - Isodicentric chromosome 15 syndrome is usually not inherited. The chromosomal change that causes the disorder typically occurs as a random event during the formation of reproductive cells (eggs or sperm) in a parent o...
5. `medquad_ans_0000531-2` (score=29.6101) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.

**Answer**:

The provided context does not mention any specific treatments for isodicentric chromosome 15 syndrome. Therefore, based on the available information, treatments cannot be determined. [medquad_ans_0000531-1] [medquad_ans_0003266-1]

**Supported citations**: `medquad_ans_0000531-1`, `medquad_ans_0003266-1`
**Unsupported citations**: _none_

### Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.928, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006224-3` (score=20.5290) - What causes Troyer syndrome? Troyer syndrome is caused by mutations in the SPG20 gene. This gene gives the body instructions to make a protein called spartin, which is present in many body tissues, including those of ...
2. `medquad_ans_0000998-1` (score=17.8627) - Troyer syndrome is part of a group of genetic disorders known as hereditary spastic paraplegias. These disorders are characterized by progressive muscle stiffness (spasticity) and the development of paralysis of the l...
3. `medquad_ans_0000998-5` (score=16.5823) - These resources address the diagnosis or management of Troyer syndrome: - Gene Review: Gene Review: Hereditary Spastic Paraplegia Overview - Gene Review: Gene Review: Troyer Syndrome - Genetic Testing Registry: Troyer...
4. `medquad_ans_0000272-4__dup1` (score=15.8135) - The NINDS supports research on genetic disorders such as the hereditary spastic paraplegias. A gene for Troyer syndrome has been identified and others may be identified in the future. Understanding how these genes cau...
5. `medquad_ans_0006224-5` (score=15.4501) - How might Troyer syndrome be treated? There are currently no treatments known to prevent or slow the progression of Troyer syndrome. Treatment aims to relieve symptoms of the disease and improve quality of life. Treat...

**Answer**:

Troyer syndrome is caused by mutations in the SPG20 gene. This gene provides instructions for making a protein called spartin, which is present in many body tissues, including those of the nervous system. It is thought that the syndrome is caused by a loss of function of the spartin protein, although the exact role of this protein and how the mutations lead to the features of Troyer syndrome are not fully understood [medquad_ans_0006224-3].

**Supported citations**: `medquad_ans_0006224-3`
**Unsupported citations**: _none_

### Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.939, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0003672-1` (score=32.3244) - Timothy syndrome is a type of long QT syndrome. It affects many parts of the body including the heart, fingers, toes, face, and the nervous system. It is characterized by long QT syndrome, although some people with Ti...
2. `medquad_ans_0000983-1` (score=28.0223) - Timothy syndrome is a rare disorder that affects many parts of the body including the heart, digits (fingers and toes), and the nervous system. Timothy syndrome is characterized by a heart condition called long QT syn...
3. `medquad_ans_0003663-3` (score=27.4624) - What causes long QT syndrome? Acquired long QT syndrome can be caused by certain medicines and medical conditions. Some medications that cause long QT syndrome include antihistamines and decongestants, antibiotics, an...
4. `medquad_ans_0000983-3` (score=25.1149) - Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. This gene provides instructions for making a protein that acts as a channel across cell membranes. This channel, known as CaV1....
5. `medquad_ans_0000983-2` (score=24.7425) - Timothy syndrome is a rare condition; fewer than 20 people with this disorder have been reported worldwide. The classic type of Timothy syndrome appears to be more common than the atypical type, which has been identif...

**Answer**:

The genetic changes related to Timothy syndrome are changes in the CACNA1C gene. All cases of Timothy syndrome appear to be due to changes in the CACNA1C gene [medquad_ans_0003672-1].

**Supported citations**: `medquad_ans_0003672-1`
**Unsupported citations**: _none_

### Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Scores**: context_relevance=0.857, groundedness=0.945, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000061-12` (score=32.8676) - There is no single, definitive test for rheumatoid arthritis. Common tests for rheumatoid arthritis include - The rheumatoid factor test. Rheumatoid factor is an antibody that is present eventually in the blood of mos...
2. `medquad_ans_0000061-3__dup3` (score=31.9275) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by i...
3. `medquad_ans_0000061-9` (score=30.9919) - Scientists believe that rheumatoid arthritis may result from the interaction of many factors such as genetics, hormones, and the environment. Although rheumatoid arthritis sometimes runs in families, the actual cause ...
4. `medquad_ans_0000061-2__dup2` (score=30.7289) - Actual Cause Is Unknown Scientists believe that rheumatoid arthritis may result from the interaction of many factors such as genetics, hormones, and the environment. Although rheumatoid arthritis sometimes runs in fam...
5. `medquad_ans_0000875-1` (score=30.4129) - Rheumatoid arthritis is a disease that causes chronic abnormal inflammation, primarily affecting the joints. The most common signs and symptoms are pain, swelling, and stiffness of the joints. Small joints in the hand...

**Answer**:

The symptoms of rheumatoid arthritis include pain and stiffness in the joints, warmth, redness, swelling around the joints, and a general feeling of being sick, tired, and sometimes feverish. Rheumatoid arthritis generally occurs in a symmetrical pattern, meaning if one knee or hand is affected, the other is also likely to be affected [medquad_ans_0000061-3__dup3].

**Supported citations**: `medquad_ans_0000061-3__dup3`
**Unsupported citations**: _none_

### Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Scores**: context_relevance=1.000, groundedness=0.990, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0002796-3` (score=28.8914) - Can Henoch-Schonlein purpura be inherited? The cause of Henoch-Schonlein purpura is currently unknown. Some evidence suggests that genetic predisposition may contribute to the development of this disease in some cases...
2. `medquad_ans_0002796-4` (score=27.0803) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...
3. `medquad_ans_0002796-1` (score=22.0530) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is no...
4. `medquad_ans_0002796-2` (score=14.9825) - What are the signs and symptoms of Henoch-Schonlein purpura? The Human Phenotype Ontology provides the following list of signs and symptoms for Henoch-Schonlein purpura. If the information is available, the table belo...
5. `medquad_ans_0000980-2` (score=14.9754) - The precise incidence of thrombotic thrombocytopenic purpura is unknown. Researchers estimate that, depending on geographic location, the condition affects 1.7 to 11 per million people each year in the United States. ...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is not fully understood, it may develop as an immune response to an infection. HSP is usually seen in children, but it may affect people of any age. Most cases go away on their own without treatment. [medquad_ans_0002796-1]

**Supported citations**: `medquad_ans_0002796-1`
**Unsupported citations**: _none_

### Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Scores**: context_relevance=1.000, groundedness=0.967, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0003162-1` (score=54.0162) - Hyper IgM syndrome is a type of primary immunodeficiency syndrome. Primary immunodeficiency occurs when part of a persons immune system is missing or does not work correctly. The bodies of people with primary immunode...
2. `medquad_ans_0001052-5` (score=48.6628) - These resources address the diagnosis or management of X-linked hyper IgM syndrome: - Gene Review: Gene Review: X-Linked Hyper IgM Syndrome - Genetic Testing Registry: Immunodeficiency with hyper IgM type 1 - MedlineP...
3. `medquad_ans_0003166-1` (score=45.9049) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information ...
4. `medquad_ans_0003163-1` (score=44.1232) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information ...
5. `medquad_ans_0003162-3` (score=42.5700) - What causes hyper IgM syndrome? A flawed gene (or genes) in T-cells (a type of white blood cell that is part of the immune system and helps the body fight diseases or harmful substances) is responsible for hyper IgM s...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [medquad_ans_0003166-1].

**Supported citations**: `medquad_ans_0003166-1`
**Unsupported citations**: _none_

### Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.960, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0000896-3` (score=30.4424) - Senior-Lken syndrome can be caused by mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopic, finger...
2. `medquad_ans_0000896-1` (score=24.7886) - Senior-Lken syndrome is a rare disorder characterized by the combination of two specific features: a kidney condition called nephronophthisis and an eye condition known as Leber congenital amaurosis. Nephronophthisis ...
3. `medquad_ans_0000667-3` (score=24.1068) - The causes of Moebius syndrome are unknown, although the condition probably results from a combination of environmental and genetic factors. Researchers are working to identify and describe specific genes related to t...
4. `medquad_ans_0003520-3` (score=23.9825) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. Th...
5. `medquad_ans_0000971-3` (score=23.8818) - The causes of task-specific focal dystonia are unknown, although the disorder likely results from a combination of genetic and environmental factors. Certain genetic changes probably increase the likelihood of develop...

**Answer**:

Genetic changes related to Senior-Lken syndrome include mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia, which are involved in signaling pathways between cells. Mutations in these genes likely lead to problems with the structure and function of cilia, disrupting important chemical signaling pathways within cells [medquad_ans_0000896-3]. However, it remains unclear how these defects specifically lead to nephronophthisis and Leber congenital amaurosis. In some individuals, no mutations in these five known genes are identified, and the genetic cause of the disorder is therefore unknown [medquad_ans_0000896-3].

**Supported citations**: `medquad_ans_0000896-3`
**Unsupported citations**: _none_

### Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.760, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000610-1` (score=19.6220) - Lymphedema-distichiasis syndrome is a condition that affects the normal function of the lymphatic system, which is a part of the circulatory and immune systems. The lymphatic system produces and transports fluids and ...
2. `medquad_ans_0000610-5` (score=17.0022) - These resources address the diagnosis or management of lymphedema-distichiasis syndrome: - Gene Review: Gene Review: Lymphedema-Distichiasis Syndrome - Genetic Testing Registry: Distichiasis-lymphedema syndrome - Medl...
3. `medquad_ans_0003700-1` (score=15.3204) - Lymphedema distichiasis syndrome is a condition that affects the normal function of the lymphatic system (part of the immune system that produces and transports fluids and immune cells throughout the body). People wit...
4. `medquad_ans_0000610-2` (score=15.1435) - The prevalence of lymphedema-distichiasis syndrome is unknown. Because the extra eyelashes can be overlooked during a medical examination, researchers believe that some people with this condition may be misdiagnosed a...
5. `medquad_ans_0003700-2` (score=14.7903) - What are the signs and symptoms of Lymphedema-distichiasis syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lymphedema-distichiasis syndrome. If the information is available...

**Answer**:

The provided context does not specify the number of people affected by lymphedema-distichiasis syndrome. [medquad_ans_0000610-1] and [medquad_ans_0003700-1] describe the condition's symptoms, causes, and associated health issues but do not provide population statistics or prevalence data. Therefore, the number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000610-1`, `medquad_ans_0003700-1`
**Unsupported citations**: _none_

### Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Scores**: context_relevance=1.000, groundedness=0.948, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0005761-3` (score=15.6166) - How is spinocerebellar ataxia type 11 inherited? SCA11 is inherited in an autosomal dominant manner. The rate of de novo mutations is not known. Each child of an individual with SCA11 has a 50% chance of inheriting th...
2. `medquad_ans_0001483-5` (score=14.8002) - Is genetic testing available for congenital bilateral absence of the vas deferens (CBAVD)? GeneTests lists the names of laboratories that are performing genetic testing for CBAVD. To view the contact information for t...
3. `medquad_ans_0002046-5` (score=14.7166) - Is genetic testing available for EEC syndrome? It is estimated that greater than 90% of cases of EEC syndrome are caused by mutations in the TP63 gene. The remainder are suspected to be caused by different mutations i...
4. `medquad_ans_0005415-4` (score=14.6905) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...
5. `medquad_ans_0002853-4` (score=14.6807) - Is genetic testing available for hereditary sensory neuropathy type 1? At least four genes responsible for hereditary sensory neuropathy type 1 (HSN1) have been found: HSN1A (the most common form) is associated with m...

**Answer**:

Yes, information about genetic testing for spinocerebellar ataxia type 11 is available. Genetic testing of adults who do not have any symptoms but are at-risk of having inherited the mutation is possible. However, testing is not useful in predicting age of onset, severity, type of symptoms, or rate of progression in individuals who do not have any symptoms. The affected family member should be tested first to confirm the molecular diagnosis in the family. For detailed information on testing, inheritance, and genetic counseling, including a comprehensive review of spinocerebellar ataxia type 11, the GeneReviews website provides current, evidence-based information [medquad_ans_0005761-3]. Additionally, the Genetic Testing Registry (GTR) offers information about genetic tests, though it is intended for healthcare providers and researchers [medquad_ans_0005761-3].

**Supported citations**: `medquad_ans_0005761-3`
**Unsupported citations**: _none_

### Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.972, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000772-1` (score=30.2945) - Parkes Weber syndrome is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxygen-rich blood from the heart to the body's va...
2. `medquad_ans_0000149-5` (score=29.0808) - These resources address the diagnosis or management of CM-AVM: - Gene Review: Gene Review: RASA1-Related Disorders - Genetic Testing Registry: Capillary malformation-arteriovenous malformation These resources from Med...
3. `medquad_ans_0000149-1` (score=27.8187) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxy...
4. `medquad_ans_0005994-1` (score=19.3478) - Systemic capillary leak syndrome is a condition in which fluid and proteins leak out of tiny blood vessels and flow into surrounding tissues, resulting in dangerously low blood pressure. Attacks frequently last for se...
5. `medquad_ans_0000654-1` (score=18.6871) - Microcephaly-capillary malformation syndrome is an inherited disorder characterized by an abnormally small head size (microcephaly) and abnormalities of small blood vessels in the skin called capillaries (capillary ma...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. It is characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin. These malformations appear as multiple small, round, pink or red spots on the skin, commonly on the face [medquad_ans_0000149-1]. CM-AVM is also associated with arteriovenous malformations (AVMs), which are abnormal connections between arteries and veins. The syndrome shares a genetic cause with some cases of Parkes Weber syndrome [medquad_ans_0000772-1]. Resources for diagnosis and management of CM-AVM include genetic testing, diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care

**Supported citations**: `medquad_ans_0000149-1`, `medquad_ans_0000772-1`
**Unsupported citations**: _none_

### Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Scores**: context_relevance=1.000, groundedness=0.716, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003571-1` (score=26.1597) - What are the signs and symptoms of Leukoencephalopathy - dystonia - motor neuropathy ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoencephalopathy - dystonia - motor neuropat...
2. `medquad_ans_0000141-2__dup1` (score=24.1642) - What are the signs and symptoms of ADCY5-related dyskinesia ? The Human Phenotype Ontology provides the following list of signs and symptoms for ADCY5-related dyskinesia . If the information is available, the table be...
3. `medquad_ans_0001388-1` (score=23.9516) - What are the signs and symptoms of COASY Protein-Associated Neurodegeneration ? The Human Phenotype Ontology provides the following list of signs and symptoms for COASY Protein-Associated Neurodegeneration . If the in...
4. `medquad_ans_0001754-1` (score=23.8106) - What are the signs and symptoms of Deafness, dystonia, and cerebral hypomyelination ? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness, dystonia, and cerebral hypomyelination...
5. `medquad_ans_0001999-1` (score=23.3055) - What are the signs and symptoms of Dystonia 6, torsion? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 6, torsion. If the information is available, the table below includes...

**Answer**:

The provided context does not contain information about the symptoms of Dystonia 16. [medquad_ans_0003571-1] discusses signs and symptoms of Leukoencephalopathy - dystonia - motor neuropathy, and [medquad_ans_0000141-2__dup1] lists symptoms of ADCY5-related dyskinesia, but neither addresses Dystonia 16 specifically. Therefore, the requested information is not available in the context.

**Supported citations**: `medquad_ans_0003571-1`, `medquad_ans_0000141-2__dup1`
**Unsupported citations**: _none_

### Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Scores**: context_relevance=0.857, groundedness=0.980, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0005548-5` (score=28.6382) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not tr...
2. `medquad_ans_0005548-3` (score=26.3743) - What causes septo-optic dysplasia? In most cases of septo-optic dysplasia, the cause of the disorder is unknown. Researchers suspect that a combination of genetic and environmental factors may play a role in causing t...
3. `medquad_ans_0005548-1` (score=26.1477) - Septo-optic dysplasia is a disorder of early brain development. The signs and symptoms vary from person to person; however, underdevelopment (hypoplasia) of the optic nerve, abnormal formation of structures along the ...
4. `medquad_ans_0000899-1` (score=25.4381) - Septo-optic dysplasia is a disorder of early brain development. Although its signs and symptoms vary, this condition is traditionally defined by three characteristic features: underdevelopment (hypoplasia) of the opti...
5. `medquad_ans_0005548-2` (score=24.7351) - What are the signs and symptoms of Septo-optic dysplasia? Symptoms may include blindness in one or both eyes, pupil dilation in response to light, nystagmus (a rapid, involuntary to-and-fro movement of the eyes), inwa...

**Answer**:

Treatment for septo-optic dysplasia is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not treatable, and vision, physical, and occupational therapies may be required [medquad_ans_0005548-5].

**Supported citations**: `medquad_ans_0005548-5`
**Unsupported citations**: _none_

### Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Scores**: context_relevance=0.800, groundedness=0.963, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000208-14` (score=53.6710) - - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. - The prostate is a walnut-shaped gland that is part of the male reproductive system. - The...
2. `medquad_ans_0000208-7` (score=52.0692) - The complications of benign prostatic hyperplasia may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney da...
3. `medquad_ans_0000208-1__dup4` (score=49.7165) - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. Benign prostatic hyperplasia is also called benign prostatic hypertrophy or benign prostatic ...
4. `medquad_ans_0000208-11` (score=48.4505) - The complications of benign prostatic hyperplasia treatment depend on the type of treatment. Medications Medications used to treat benign prostatic hyperplasia may have side effects that sometimes can be serious. Men ...
5. `medquad_ans_0000208-4__dup2` (score=48.4071) - Benign prostatic hyperplasia is the most common prostate problem for men older than age 50. In 2010, as many as 14 million men in the United States had lower urinary tract symptoms suggestive of benign prostatic hyper...

**Answer**:

The complications of benign prostatic hyperplasia may include acute urinary retention, chronic or long-lasting urinary retention, blood in the urine, urinary tract infections (UTIs), bladder damage, kidney damage, and bladder stones [medquad_ans_0000208-14]. Most men with benign prostatic hyperplasia do not develop these complications, however, kidney damage in particular can be a serious health threat when it occurs [medquad_ans_0000208-7].

**Supported citations**: `medquad_ans_0000208-14`, `medquad_ans_0000208-7`
**Unsupported citations**: _none_

### Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Scores**: context_relevance=0.889, groundedness=0.780, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000001-3__dup1` (score=47.9145) - When the absence of the septum pellucidum is part of septo-optic dysplasia, the prognosis varies according to the presence and severity of associated symptoms. By itself, absence of the septum pellucidum is not life-t...
2. `medquad_ans_0000001-1__dup1` (score=33.5398) - The septum pellucidum (SP) is a thin membrane located at the midline of the brain between the two cerebral hemispheres, or halves of the brain.. It is connected to the corpus callosum -- a collection of nerve fibers t...
3. `medquad_ans_0000096-1__dup2` (score=31.9663) - Septo-optic dysplasia (SOD) is a rare disorder characterized by abnormal development of the optic disk, pituitary deficiencies, and often agenesis (absence) of the septum pellucidum (the part of the brain that separat...
4. `medquad_ans_0002968-1` (score=30.7836) - What are the signs and symptoms of Hydrolethalus syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Hydrolethalus syndrome. If the information is available, the table below in...
5. `medquad_ans_0003726-1` (score=30.1732) - What are the signs and symptoms of Macrocephaly-capillary malformation? The Human Phenotype Ontology provides the following list of signs and symptoms for Macrocephaly-capillary malformation. If the information is ava...

**Answer**:

The provided context does not mention any treatments for the absence of the septum pellucidum. While it describes the condition's association with septo-optic dysplasia and related symptoms, including hormonal deficiencies, vision problems, and seizures, no specific treatments are outlined. Therefore, based on the given information, the treatments for absence of the septum pellucidum cannot be determined. [medquad_ans_0000001-3\_\_dup1][medquad_ans_0000001-1__dup1][medquad_ans_0000096-1__dup2]

**Supported citations**: `medquad_ans_0000001-3__dup1`, `medquad_ans_0000001-1__dup1`, `medquad_ans_0000096-1__dup2`
**Unsupported citations**: _none_

### Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Scores**: context_relevance=1.000, groundedness=0.511, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002322-3` (score=41.4820) - What causes febrile ulceronecrotic Mucha-Habermann disease? The cause of FUMHD is not known (idiopathic). A hypersensitivity to an infectious agent is suggested to be the main cause. Single cases of people with FUMHD ...
2. `medquad_ans_0002322-2` (score=37.1605) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (...
3. `medquad_ans_0002322-1` (score=34.6273) - Febrile ulceronecrotic Mucha-Habermann disease (FUMHD) is a rare and severe form of pityriasis lichenoides et varioliformis acuta (PLEVA). PLEVA is characterized by skin lesions that ulcerate, breakdown, form open sor...
4. `medquad_ans_0002322-4` (score=34.5000) - How is febrile ulceronecrotic Mucha-Habermann disease definitively diagnosed? FUMHD is diagnosed based upon the clinical symptoms in the patient, with confirmation by skin biopsy. Skin biopsy findings suggestive of FU...
5. `medquad_ans_0002322-5` (score=30.7265) - How is febrile ulceronecrotic Mucha-Habermann disease (FUMHD) treated? It is important that FUMHD is diagnosed and treated as soon as possible. While a number of treatments have been tried, it is hard to asses the ben...

**Answer**:

The symptoms of febrile ulceronecrotic Mucha-Habermann disease (FUMHD) include red scaly skin lesions (papules) that ulcerate, break down, form open sores, and then develop a red-brown crust (i.e., PLEVA). These lesions can rapidly progress to large, destructive ulcers with extensive, painful loss of skin tissue. The skin lesions may become infected, leading to pus and a putrid odor. FUMHD is often associated with high fever (up to 104°F), which may be persistent or recurrent. Other symptoms may include feeling ill, sore throat, congestion, muscle soreness or pain, joint pain, diarrhea, central nervous system symptoms, abdominal pain, enlarged spleen, arthritis, megaloblastic anemia, interstitial pneumonitis, lymphocytic (viral) myocarditis, and sepsis. FUMHD can be

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Scores**: context_relevance=1.000, groundedness=0.944, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0002833-4` (score=40.8990) - How is hereditary leiomyomatosis and renal cell cancer inherited? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the ...
2. `medquad_ans_0000465-5` (score=38.7152) - These resources address the diagnosis or management of HLRCC: - Gene Review: Gene Review: Hereditary Leiomyomatosis and Renal Cell Cancer - Genetic Testing Registry: Hereditary leiomyomatosis and renal cell cancer - M...
3. `medquad_ans_0002833-3` (score=35.9609) - What causes hereditary leiomyomatosis and renal cell cancer? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is caused by changes (mutations) in the FH gene. This gene gives the body instructions for making an...
4. `medquad_ans_0002833-1` (score=35.7295) - Hereditary leiomyomatosis and renal cell cancer (HLRCC) is a condition that causes benign tumors of smooth muscle tissue in the skin (cutaneous leiomyomas) and in the uterus in females (uterine leiomyomas, or fibroids...
5. `medquad_ans_0002833-5` (score=35.3003) - How might hereditary leiomyomatosis and renal cell cancer be treated? Skin growths (cutaneous leiomyomas) associated with hereditary leiomyomatosis and renal cell cancer (HLRCC) should be examined by a dermatologist. ...

**Answer**:

Yes, hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the gene in each cell is enough to cause symptoms of the condition [medquad_ans_0002833-4]. In some cases, an affected person inherits the mutated copy of the gene from an affected parent, while in other cases, the mutation occurs for the first time in the individual due to a new mutation [medquad_ans_0002833-4].

**Supported citations**: `medquad_ans_0002833-4`
**Unsupported citations**: _none_

### Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000389-1` (score=30.3156) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which al...
2. `medquad_ans_0000389-5` (score=27.0852) - These resources address the diagnosis or management of frontometaphyseal dysplasia: - Gene Review: Gene Review: Otopalatodigital Spectrum Disorders - Genetic Testing Registry: Frontometaphyseal dysplasia These resourc...
3. `medquad_ans_0000389-3` (score=24.8542) - Mutations in the FLNA gene cause frontometaphyseal dysplasia. The FLNA gene provides instructions for producing the protein filamin A, which helps build the network of protein filaments (cytoskeleton) that gives struc...
4. `medquad_ans_0002946-1` (score=21.5434) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...
5. `medquad_ans_0002062-1` (score=21.2726) - What are the signs and symptoms of Ehlers-Danlos syndrome, spondylocheirodysplastic type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, spondylocheirodyspl...

**Answer**:

Symptoms of frontometaphyseal dysplasia include joint deformities called contractures that restrict joint movement, bowed limbs, an abnormal curvature of the spine (scoliosis), and abnormalities of the fingers and hands. Characteristic facial features may include prominent brow ridges, wide-set and downward-slanting eyes, a very small lower jaw and chin (micrognathia), and small, missing, or misaligned teeth. Some affected individuals have hearing loss. Additionally, individuals may experience obstruction of the ducts between the kidneys and bladder (ureters), heart defects, or constrictions in the passages leading from the windpipe to the lungs (bronchi) that can cause breathing problems. Males generally have more severe signs and symptoms than females, who may show only the characteristic facial features [medquad_ans_0000389-1].

**Supported citations**: `medquad_ans_0000389-1`
**Unsupported citations**: _none_

### Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Scores**: context_relevance=0.818, groundedness=0.130, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000038-22` (score=37.3971) - During your next health care visit, talk to your provider about your test results and how to manage your kidney disease. Below is a list of questions you may want to ask. Add any questions you think are missing, and m...
2. `medquad_ans_0000035-9` (score=35.8086) - People can manage their diabetes with meal planning, physical activity, and if needed, medications. More information about taking care of type 1 or type 2 diabetes is provided in the NIDDK health topics: - What I need...
3. `medquad_ans_0000036-19` (score=34.9422) - If you are diagnosed with high blood pressure, here are questions to ask your doctor. - Your Blood Pressure Numbers - What is my blood pressure reading in numbers? - What is my goal blood pressure? - Is my blood press...
4. `medquad_ans_0000078-11` (score=33.6951) - Your health care team will help you make a healthy eating plan with food choices that are good for both you and your baby. These choices are good for you to follow throughout pregnancy and after, as you raise your fam...
5. `medquad_ans_0000193-6` (score=33.6352) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need t...

**Answer**:

The provided context does not contain any information about diverticular disease or its treatments. Therefore, it is not possible to answer the question based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.534, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000619-1__dup1` (score=46.1227) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is avail...
2. `medquad_ans_0006066-1` (score=21.4654) - What are the signs and symptoms of Thompson Baraitser syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thompson Baraitser syndrome. If the information is available, the tabl...
3. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
4. `medquad_ans_0005488-1` (score=20.7288) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
5. `medquad_ans_0006607-2` (score=20.5187) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...

**Answer**:

The symptoms of Baraitser Brett Piesowicz syndrome include cerebral calcification, hyperreflexia, hypertonia, microcephaly, seizures, abnormality of movement, cerebral cortical atrophy, cataract, opacification of the corneal stroma, renal insufficiency, anteverted nares, cerebellar hypoplasia, decreased liver function, elevated hepatic transaminases, failure to thrive, hepatomegaly, high palate, increased CSF protein, intellectual disability, profound, jaundice, lissencephaly, long philtrum, low-set ears, microretrognathia, muscular hypotonia of the trunk, nystagmus, pachygyria, petechiae, phenotypic variability, polymicrogyria, sloping forehead, spasticity, splenomegaly, thrombocytopenia, ventriculomeg

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003911-2` (score=26.9454) - How might meningiomas be treated? The treatment varies depending on the location of the meningioma and the symptoms caused by the tumor. Careful observation is sometimes the best course of action for people with a men...
2. `medquad_ans_0003911-1` (score=22.3072) - Meningiomas originate in the meninges, the membranes that surround the brain and spinal cord. Most meningiomas are benign, though a minority of meningiomas can be classified as atypical or malignant. Though rare, mali...
3. `medquad_ans_0003103-4` (score=19.3836) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...
4. `medquad_ans_0003897-1` (score=19.2438) - What are the signs and symptoms of Melanoma astrocytoma syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Melanoma astrocytoma syndrome. If the information is available, the ...
5. `medquad_ans_0000108-2__dup1` (score=19.0184) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The treatment for meningiomas varies depending on the location of the tumor and the symptoms it causes. Careful observation may be the best course of action for some patients. When treatment is necessary, surgery and radiation are the most common options. Radiation may be used if the meningioma cannot be operated on or if it is only partially removed by surgery. It is also used in cases of malignant, atypical, or recurrent tumors. Other treatments that have been tried or are being explored include hydroxyurea, epidermal growth factor receptor inhibitors, platelet-derived growth factor receptor inhibitors, vascular endothelial growth factor inhibitors, immunotherapy to stimulate the immune system, and somatostatin analogs which prevent the release of growth hormones [medquad_ans_0003911-2].

**Supported citations**: `medquad_ans_0003911-2`
**Unsupported citations**: _none_

### Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Scores**: context_relevance=1.000, groundedness=0.857, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000006_7-1` (score=56.8116) - Key Points - Childhood central nervous system (CNS) germ cell tumors form from germ cells. - There are different types of childhood CNS germ cell tumors. - Germinomas - Nongerminomas - The cause of most childhood CNS ...
2. `medquad_ans_0000006_7-2` (score=51.7203) - Key Points - Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and s...
3. `medquad_ans_0000006_7-8` (score=44.5755) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a te...
4. `medquad_ans_0000015_1-1` (score=42.9937) - Key Points - Childhood extracranial germ cell tumors form from germ cells in parts of the body other than the brain. - Childhood extracranial germ cell tumors may be benign or malignant. - There are three types of ext...
5. `medquad_ans_0000006_4-1` (score=38.5532) - Key Points - Central nervous system atypical teratoid/rhabdoid tumor is a disease in which malignant (cancer) cells form in the tissues of the brain. - Certain genetic changes may increase the risk of atypical teratoi...

**Answer**:

The provided context does not specify the treatments for Childhood Central Nervous System Germ Cell Tumors. While it mentions that treatment for children may differ from treatment for adults and references other PDQ summaries for adult treatments, no details about specific treatments for childhood CNS germ cell tumors are given [medquad_ans_0000006_7-1]. Therefore, based on the available information, the treatments are not explicitly stated.

**Supported citations**: `medquad_ans_0000006_7-1`
**Unsupported citations**: _none_

### Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Scores**: context_relevance=1.000, groundedness=0.473, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0004401-3` (score=47.6752) - What causes neutral lipid storage disease with myopathy? Neutral lipid storage disease with myopathy is caused by mutations in the PNPLA2 gene. This gene provides instructions for making an enzyme called adipose trigl...
2. `medquad_ans_0000729-2` (score=46.6249) - Neutral lipid storage disease with myopathy is a rare condition; its incidence is unknown.
3. `medquad_ans_0000729-1` (score=45.8850) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the ...
4. `medquad_ans_0000729-5` (score=43.2656) - These resources address the diagnosis or management of neutral lipid storage disease with myopathy: - Genetic Testing Registry: Neutral lipid storage disease with myopathy - MedlinePlus Encyclopedia: Hypothyroidism - ...
5. `medquad_ans_0004401-4` (score=40.6594) - How is neutral lipid storage disease with myopathy inherited? This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individua...

**Answer**:

Neutral lipid storage disease with myopathy is a rare condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the accumulation of fats in muscle tissue. Other features may include a fatty liver, a weakened and enlarged heart (cardiomyopathy), inflammation of the pancreas (pancreatitis), reduced thyroid activity (hypothyroidism), and type 2 diabetes mellitus (the most common form of diabetes). The condition is caused by mutations in the PNPLA2 gene, which leads to impaired function of the adipose triglyceride lipase (ATGL) enzyme, resulting in the inability to break down triglycerides and their subsequent accumulation in muscle and tissues. It is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations, and each parent carries one copy of the mutated gene without showing

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Scores**: context_relevance=0.800, groundedness=0.966, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0006546-1` (score=29.9453) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. ...
2. `medquad_ans_0006546-2` (score=28.3769) - How might a Zika virus infection be treated? There is no vaccine to prevent Zika virus infections, nor is there a specific medicine to treat Zika. Individuals infected with the Zika virus should get plenty of rest, dr...
3. `medquad_ans_0000981-1__dup1` (score=24.9233) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been repo...
4. `medquad_ans_0000193-1__dup2` (score=21.0789) - Microcephaly is a medical condition in which the circumference of the head is smaller than normal because the brain has not developed properly or has stopped growing. Microcephaly can be present at birth or it may dev...
5. `medquad_ans_0000174-1__dup3` (score=16.2527) - Chikungunya is a virus that spread by the same kinds of mosquitoes that spread dengue and Zika virus. Rarely, it can spread from mother to newborn around the time of birth. It may also possibly spread through infected...

**Answer**:

Zika virus is a virus spread primarily through the bite of an infected mosquito. It can also be transmitted from a pregnant mother to her child during pregnancy or around the time of birth, through sexual contact with an affected male partner, and via blood transfusions. The illness associated with Zika virus infection is usually mild, with common symptoms including fever, rash, joint pain, and conjunctivitis (red eyes). Recent research has suggested a link between Zika virus infection and Guillain-Barre syndrome (GBS), and prenatal infection has been shown to cause adverse pregnancy outcomes such as microcephaly and other serious brain anomalies [medquad_ans_0006546-1]. There is no vaccine or specific medicine to treat Zika virus; management includes rest, hydration, and use of acetaminophen for pain, with avoidance of aspirin and NSAIDs until dengue is ruled out [medquad_ans_00065

**Supported citations**: `medquad_ans_0006546-1`
**Unsupported citations**: _none_

### Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000019-1__dup2` (score=18.2741) - Broken heart syndrome is a condition in which extreme stress can lead to heart muscle failure. The failure is severe, but often short-term. Most people who experience broken heart syndrome think they may be having a h...
2. `medquad_ans_0000019-4__dup2` (score=17.1105) - Broken heart syndrome affects women more often than men. Often, people who experience broken heart syndrome have previously been healthy. Research shows that the traditional risk factors for heart disease may not appl...
3. `medquad_ans_0000019-3__dup2` (score=16.8558) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack...
4. `medquad_ans_0000019-5__dup1` (score=15.8167) - All chest pain should be checked by a doctor. Because symptoms of broken heart syndrome are similar to those of a heart attack, it is important to seek help right away. Your doctor may not be able to diagnose broken h...
5. `medquad_ans_0000019-7` (score=15.0413) - Even though broken heart syndrome may feel like a heart attack, its a very different problem that needs a different type of treatment. The good news is that broken heart syndrome is usually treatable, and most people ...

**Answer**:

The cause of broken heart syndrome is not fully known. In most cases, symptoms are triggered by extreme emotional or physical stress, such as intense grief, anger, or surprise. Researchers think that the stress releases hormones that stun the heart and affect its ability to pump blood to the body. [medquad_ans_0000019-1__dup2]

**Supported citations**: `medquad_ans_0000019-1__dup2`
**Unsupported citations**: _none_

### Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Scores**: context_relevance=0.800, groundedness=0.503, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000448-4` (score=30.3785) - The inheritance pattern of Hashimoto thyroiditis is unclear because many genetic and environmental factors appear to be involved. However, the condition can cluster in families, and having a close relative with Hashim...
2. `medquad_ans_0000448-3` (score=28.8843) - Hashimoto thyroiditis is thought to result from a combination of genetic and environmental factors. Some of these factors have been identified, but many remain unknown. Hashimoto thyroiditis is classified as an autoim...
3. `medquad_ans_0000448-1` (score=28.8395) - Hashimoto thyroiditis is a condition that affects the function of the thyroid, which is a butterfly-shaped gland in the lower neck. The thyroid makes hormones that help regulate a wide variety of critical body functio...
4. `medquad_ans_0000448-2` (score=25.2921) - Hashimoto thyroiditis affects 1 to 2 percent of people in the United States. It occurs more often in women than in men, which may be related to hormonal factors. The condition is the most common cause of thyroid under...
5. `medquad_ans_0000448-5` (score=23.3813) - These resources address the diagnosis or management of Hashimoto thyroiditis: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Hashimoto thyroiditis - National Institute of Diabetes a...

**Answer**:

Hashimoto thyroiditis is a condition that affects the function of the thyroid, a butterfly-shaped gland in the lower neck. The thyroid makes hormones that help regulate a wide variety of critical body functions. It is classified as an autoimmune disorder, in which the immune system attacks the body's own tissues and organs. In people with Hashimoto thyroiditis, white blood cells called lymphocytes accumulate abnormally in the thyroid, leading to damage of thyroid cells. These lymphocytes produce immune system proteins called antibodies that attack and destroy thyroid cells. When too many thyroid cells are damaged or die, the thyroid can no longer produce enough hormones to regulate body functions, resulting in hypothyroidism. However, some individuals with thyroid antibodies never develop hypothyroidism or experience symptoms. Hashimoto thyroiditis has a complex etiology involving both genetic and environmental factors, with genetic variations—such as those in the human leukocyte antigen (HLA) complex and genes involved in immune regulation

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Scores**: context_relevance=1.000, groundedness=0.912, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0006463-1` (score=18.7221) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include...
2. `medquad_ans_0001043-2` (score=18.4481) - Wolman disease is estimated to occur in 1 in 350,000 newborns.
3. `medquad_ans_0001043-1` (score=15.7168) - Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). In affected individuals, harmful amounts of lipids accumulate in the spleen, liver, ...
4. `medquad_ans_0001043-3` (score=15.4675) - Mutations in the LIPA gene cause Wolman disease. The LIPA gene provides instructions for producing an enzyme called lysosomal acid lipase. This enzyme is found in the lysosomes (compartments that digest and recycle ma...
5. `medquad_ans_0006463-2` (score=14.7576) - What are the signs and symptoms of Wolman disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Wolman disease. If the information is available, the table below includes how often...

**Answer**:

Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). It is a type of lysosomal storage disorder caused by mutations in the LIPA gene, which provides instructions for producing an enzyme called lysosomal acid lipase. This enzyme is essential for processing lipids such as cholesteryl esters and triglycerides in lysosomes, where they are broken down and recycled. Mutations in the LIPA gene lead to a shortage of lysosomal acid lipase, resulting in the accumulation of triglycerides, cholesteryl esters, and other fats within cells and tissues. This accumulation, along with malnutrition due to the body's inability to use lipids properly, causes the signs and symptoms of Wolman disease [medquad_ans_0001043-1][medquad_ans_0006463-1][medquad_ans_0

**Supported citations**: `medquad_ans_0001043-1`, `medquad_ans_0006463-1`
**Unsupported citations**: _none_

### Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002753-3` (score=16.9735) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma a...
2. `medquad_ans_0001853-3` (score=13.3144) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
3. `medquad_ans_0002753-1` (score=11.7580) - Hemangiopericytoma is a term used to described a group of tumors that are derived from pericytes, the cells normally arranged along specific types of blood vessels called capillaries and venules. These types of tumors...
4. `medquad_ans_0000108-2__dup1` (score=7.5116) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
5. `medquad_ans_0000549-1__dup1` (score=7.3447) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a possible relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3]. Hemangiopericytoma is a tumor derived from pericytes, which are cells normally arranged along blood vessels [medquad_ans_0002753-1].

**Supported citations**: `medquad_ans_0002753-3`, `medquad_ans_0002753-1`
**Unsupported citations**: _none_

### Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Scores**: context_relevance=1.000, groundedness=0.948, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000593-3` (score=25.6266) - Mutations in at least two genes on the X chromosome are thought to be responsible for Lenz microphthalmia syndrome. Only one of these genes, BCOR, has been identified. The BCOR gene provides instructions for making a ...
2. `medquad_ans_0003520-3` (score=23.9825) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. Th...
3. `medquad_ans_0000971-3` (score=23.8818) - The causes of task-specific focal dystonia are unknown, although the disorder likely results from a combination of genetic and environmental factors. Certain genetic changes probably increase the likelihood of develop...
4. `medquad_ans_0000655-3` (score=23.0892) - Microphthalmia may be caused by changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or...
5. `medquad_ans_0000116-3` (score=22.3630) - The causes of benign essential blepharospasm are unknown, although the disorder likely results from a combination of genetic and environmental factors. Certain genetic changes probably increase the likelihood of devel...

**Answer**:

The genetic changes related to microphthalmia in Lenz microphthalmia syndrome involve mutations in at least two genes on the X chromosome. Only one of these genes, BCOR, has been identified. The BCOR gene provides instructions for making a protein called the BCL6 corepressor, which helps regulate the activity of other genes and plays an important role in early embryonic development. A mutation in the BCOR gene changes the structure of the BCL6 corepressor protein, disrupting normal development of the eyes and several other organs and tissues before birth [medquad_ans_0000593-3]. Researchers are investigating whether Lenz microphthalmia syndrome is caused by a single disorder with different genetic causes or by two distinct disorders due to mutations in different genes [medquad_ans_0000593-3].

**Supported citations**: `medquad_ans_0000593-3`
**Unsupported citations**: _none_

### Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Scores**: context_relevance=0.833, groundedness=0.725, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000028-6__dup1` (score=25.8467) - Because uric acids role in gout is well understood and medications to ease attacks and reduce the risk or severity of future attacks are widely available, gout is one of the mostif not the mostcontrollable forms of ar...
2. `medquad_ans_0000409-1__dup1` (score=24.9802) - Gout is a common, painful form of arthritis. It causes swollen, red, hot and stiff joints. Gout happens when uric acid builds up in your body. Uric acid comes from the breakdown of substances called purines. Purines a...
3. `medquad_ans_0000028-10` (score=24.6837) - These risk factors are associated with gout. - Genetics. Many people with gout have a family history of the disease. - Gender and age. Gout is more common in men than in women and more common in adults than in childre...
4. `medquad_ans_0000028-3__dup3` (score=24.5967) - Attacks Usually Start at Night Gout is a form of arthritis that causes sudden onset of intense pain and swelling in the joints, which also may be warm and red. Gout typically attacks one joint at a time, and the attac...
5. `medquad_ans_0000028-4__dup3` (score=24.5345) - Symptoms Can Be Controlled With proper treatment, most people with gout are able to control their symptoms and live productive lives. The goals for treatment are to ease the pain that comes from sudden attacks, preven...

**Answer**:

The provided context does not mention the symptoms of gout. [medquad_ans_0000028-6__dup1] discusses treatment, research areas, and dietary factors related to gout but does not describe its symptoms. Therefore, based on the given information, the symptoms of gout cannot be answered.

**Supported citations**: `medquad_ans_0000028-6__dup1`
**Unsupported citations**: _none_

### Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Scores**: context_relevance=1.000, groundedness=0.416, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0001853-3` (score=13.3144) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
2. `medquad_ans_0003571-1` (score=13.1141) - What are the signs and symptoms of Leukoencephalopathy - dystonia - motor neuropathy ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoencephalopathy - dystonia - motor neuropat...
3. `medquad_ans_0000971-1` (score=12.3509) - Task-specific focal dystonia is a movement disorder that interferes with the performance of particular tasks, such as writing, playing a musical instrument, or participating in a sport. Dystonias are a group of moveme...
4. `medquad_ans_0000776-1` (score=11.6140) - Partington syndrome is a neurological disorder that causes intellectual disability along with a condition called focal dystonia that particularly affects movement of the hands. Partington syndrome usually occurs in ma...
5. `medquad_ans_0000859-1` (score=11.1509) - Rapid-onset dystonia parkinsonism is a rare movement disorder. "Rapid-onset" refers to the abrupt appearance of signs and symptoms over a period of hours to days. Dystonia is a condition characterized by involuntary, ...

**Answer**:

The provided context does not contain information about the cause of Dystonia 8. Therefore, a definitive answer cannot be given based on the available content. [medquad_ans_0000971-1] discusses task-specific focal dystonia and general characteristics of dystonias but does not specify the cause of Dystonia 8. [doc_id] is not applicable.

**Supported citations**: `medquad_ans_0000971-1`
**Unsupported citations**: `doc_id`

### Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Scores**: context_relevance=0.778, groundedness=0.732, answer_relevance=0.689

**Top docs**:

1. `medquad_ans_0000612-1` (score=32.7692) - Lysinuric protein intolerance is a disorder caused by the body's inability to digest and use certain protein building blocks (amino acids), namely lysine, arginine, and ornithine. Because the body cannot effectively b...
2. `medquad_ans_0003719-1` (score=32.4655) - Lysinuric protein intolerance is a metabolic disorder caused by the body's inability to digest and use the amino acids lysine, arginine, and ornithine. Because the body cannot effectively break down these amino acids,...
3. `medquad_ans_0000612-5` (score=30.1004) - These resources address the diagnosis or management of lysinuric protein intolerance: - Gene Review: Gene Review: Lysinuric Protein Intolerance - Genetic Testing Registry: Lysinuric protein intolerance - MedlinePlus E...
4. `medquad_ans_0000612-3` (score=27.1448) - Mutations in the SLC7A7 gene cause lysinuric protein intolerance. The SLC7A7 gene provides instructions for producing a protein called y+L amino acid transporter 1 (y+LAT-1), which is involved in transporting lysine, ...
5. `medquad_ans_0000612-2` (score=24.6228) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unk...

**Answer**:

The provided context does not specify the number of people affected by lysinuric protein intolerance. [medquad_ans_0000612-1][medquad_ans_0003719-1][medquad_ans_0000612-5] discuss the causes, symptoms, and management of the disorder but do not include population prevalence or affected numbers. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000612-1`, `medquad_ans_0003719-1`, `medquad_ans_0000612-5`
**Unsupported citations**: _none_

### Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.633, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000170-10` (score=29.8343) - - Renal artery stenosis (RAS) is the narrowing of one or both renal arteries. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood vessel that carries blood from the heart ...
2. `medquad_ans_0000170-1__dup3` (score=29.3017) - Renal artery stenosis is the narrowing of one or both renal arteries. Renal means kidney and stenosis means narrowing. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood ...
3. `medquad_ans_0000170-3__dup2` (score=27.4262) - About 90 percent of RAS is caused by atherosclerosisclogging, narrowing, and hardening of the renal arteries.2 In these cases, RAS develops when plaquea sticky substance made up of fat, cholesterol, calcium, and other...
4. `medquad_ans_0004051-1` (score=25.6220) - What are the signs and symptoms of Microphthalmia syndromic 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Microphthalmia syndromic 9. If the information is available, the table ...
5. `medquad_ans_0006446-2` (score=25.2012) - What are the signs and symptoms of Williams syndrome? The signs and symptoms of Williams syndrome can be variable, but the disorder is generally characterized by mild to moderate intellectual disability a distinctive ...

**Answer**:

Renal artery stenosis (RAS) often has no symptoms until it becomes severe. The first symptoms are usually either high blood pressure or decreased kidney function, or both [medquad_ans_0000170-10]. High blood pressure is defined as a systolic pressure of 140 or above or a diastolic pressure of 90 or above [medquad_ans_0000170-1]. RAS is also associated with an increased risk for chronic kidney disease (CKD), coronary artery disease, stroke, and peripheral vascular disease [medquad_ans_0000170-10]. However, RAS is often overlooked as a cause of high blood pressure [medquad_ans_0000170-10].

**Supported citations**: `medquad_ans_0000170-10`
**Unsupported citations**: `medquad_ans_0000170-1`

### Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Scores**: context_relevance=0.800, groundedness=0.862, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000021-1` (score=41.5009) - Adermatoglyphia is the absence of ridges on the skin on the pads of the fingers and toes, as well as on the palms of the hands and soles of the feet. The patterns of these ridges (called dermatoglyphs) form whorls, ar...
2. `medquad_ans_0000712-5` (score=38.4380) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis -...
3. `medquad_ans_0000712-1` (score=37.6901) - Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis (NFJS/DPR) represents a rare type of ectodermal dysplasia, a group of about 150 conditions characterized by abnormal development of ectoder...
4. `medquad_ans_0000871-1` (score=22.4877) - Retinitis pigmentosa is a group of related eye disorders that cause progressive vision loss. These disorders affect the retina, which is the layer of light-sensitive tissue at the back of the eye. In people with retin...
5. `medquad_ans_0000871-3` (score=21.9501) - Mutations in more than 60 genes are known to cause nonsyndromic retinitis pigmentosa. More than 20 of these genes are associated with the autosomal dominant form of the disorder. Mutations in the RHO gene are the most...

**Answer**:

The provided context does not specify any treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis (NFJS/DPR). It mentions resources that address diagnosis and management, such as genetic testing and information on diagnostic tests, drug therapy, genetic counseling, and palliative care, but does not detail specific treatments [medquad_ans_0000712-5]. Therefore, based on the given information, no specific treatments can be identified.

**Supported citations**: `medquad_ans_0000712-5`
**Unsupported citations**: _none_

### Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Scores**: context_relevance=1.000, groundedness=0.959, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002259-1` (score=43.8079) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0000353-3` (score=40.1243) - The various types of familial hyperaldosteronism have different genetic causes. Familial hyperaldosteronism type I is caused by the abnormal joining together (fusion) of two similar genes called CYP11B1 and CYP11B2, w...
3. `medquad_ans_0000353-1` (score=39.8127) - Familial hyperaldosteronism is a group of inherited conditions in which the adrenal glands, which are small glands located on top of each kidney, produce too much of the hormone aldosterone. Aldosterone helps control ...
4. `medquad_ans_0000353-2` (score=33.0754) - The prevalence of familial hyperaldosteronism is unknown. Familial hyperaldosteronism type II appears to be the most common variety. All types of familial hyperaldosteronism combined account for fewer than 1 out of 10...
5. `medquad_ans_0000353-5` (score=30.8790) - These resources address the diagnosis or management of familial hyperaldosteronism: - Genetic Testing Registry: Familial hyperaldosteronism type 1 - Genetic Testing Registry: Familial hyperaldosteronism type 3 - Hormo...

**Answer**:

The signs and symptoms of Familial hyperaldosteronism type III include hypercalciuria, metabolic acidosis, polydipsia, polyuria, adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

**Supported citations**: `medquad_ans_0002259-1`
**Unsupported citations**: _none_

### Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Scores**: context_relevance=1.000, groundedness=0.952, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0005761-3` (score=24.6117) - How is spinocerebellar ataxia type 11 inherited? SCA11 is inherited in an autosomal dominant manner. The rate of de novo mutations is not known. Each child of an individual with SCA11 has a 50% chance of inheriting th...
2. `medquad_ans_0005761-1` (score=23.7297) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal...
3. `medquad_ans_0005777-1` (score=22.9954) - Spinocerebellar ataxia 3 is a rare movement disorder that is characterized by ataxia, clumsiness and weakness in the arms and legs, spasticity, a staggering lurching walk easily mistaken for drunkenness, difficulty wi...
4. `medquad_ans_0005763-1` (score=22.7295) - Spinocerebellar ataxia 13 (SCA13) is a rare sub-type of spinocerebellar ataxias, a group of neurological conditions characterized by degeneration of the brain and spinal cord. Signs and symptoms of SCA13 appear to var...
5. `medquad_ans_0000940-2` (score=21.3847) - The prevalence of SCA3 is unknown. This condition is thought to be the most common type of spinocerebellar ataxia; however, all types of spinocerebellar ataxia are relatively rare.

**Answer**:

Yes, Spinocerebellar ataxia type 11 (SCA11) is inherited in an autosomal dominant manner. Each child of an individual with SCA11 has a 50% chance of inheriting the mutation [medquad_ans_0005761-3].

**Supported citations**: `medquad_ans_0005761-3`
**Unsupported citations**: _none_

### Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Scores**: context_relevance=1.000, groundedness=0.942, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000503-1` (score=66.0754) - Hypochromic microcytic anemia with iron overload is a condition that impairs the normal transport of iron in cells. Iron is an essential component of hemoglobin, which is the substance that red blood cells use to carr...
2. `medquad_ans_0000503-5` (score=61.6987) - These resources address the diagnosis or management of hypochromic microcytic anemia with iron overload: - Genetic Testing Registry: Hypochromic microcytic anemia with iron overload These resources from MedlinePlus of...
3. `medquad_ans_0000503-2` (score=53.5959) - Hypochromic microcytic anemia with iron overload is likely a rare disorder; at least five affected families have been reported in the scientific literature.
4. `medquad_ans_0000503-3` (score=50.8894) - Mutations in the SLC11A2 gene cause hypochromic microcytic anemia with iron overload. The SLC11A2 gene provides instructions for making a protein called divalent metal transporter 1 (DMT1). The DMT1 protein is found i...
5. `medquad_ans_0003040-1` (score=47.9466) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload....

**Answer**:

The symptoms of hypochromic microcytic anemia with iron overload include pale skin (pallor), tiredness (fatigue), and slow growth [medquad_ans_0000503-1].

**Supported citations**: `medquad_ans_0000503-1`
**Unsupported citations**: _none_

### Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Scores**: context_relevance=0.800, groundedness=0.741, answer_relevance=0.760

**Top docs**:

1. `medquad_ans_0000115-2` (score=29.7199) - Benign chronic pemphigus is a rare condition; its prevalence is unknown.
2. `medquad_ans_0000115-1` (score=23.9192) - Benign chronic pemphigus, often called Hailey-Hailey disease, is a rare skin condition that usually appears in early adulthood. The disorder is characterized by red, raw, and blistered areas of skin that occur most of...
3. `medquad_ans_0000115-5` (score=23.3427) - These resources address the diagnosis or management of benign chronic pemphigus: - American Osteopathic College of Dermatology - Genetic Testing Registry: Familial benign pemphigus These resources from MedlinePlus off...
4. `medquad_ans_0004796-2` (score=19.9599) - What causes pemphigus vulgaris? Pemphigus vulgaris is an autoimmune disorder. The immune system produces antibodies against specific proteins in the skin and mucous membranes. These antibodies create a reaction that c...
5. `medquad_ans_0002290-1` (score=19.5637) - Familial pemphigus vulgaris refers to a cluster of pemphigus vulgaris within a family. Pemphigus vulgaris is a rare autoimmune condition that is characterized by blisters and sores on the skin and mucus membranes. Alt...

**Answer**:

The provided context does not explicitly state whether benign chronic pemphigus is inherited. While it mentions genetic testing resources and genetic counseling in relation to benign chronic pemphigus [medquad_ans_0000115-5], and notes that genetic factors are linked to pemphigus vulgaris [medquad_ans_0004796-2], there is no direct information confirming inheritance in benign chronic pemphigus. Therefore, based on the given context, the inheritance status of benign chronic pemphigus cannot be determined.

Answer: Insufficient information to determine if benign chronic pemphigus is inherited. [medquad_ans_0000115-5]

**Supported citations**: `medquad_ans_0000115-5`, `medquad_ans_0004796-2`
**Unsupported citations**: _none_

### Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.900, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000915-1__dup1` (score=20.9265) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
2. `medquad_ans_0000916-1__dup1` (score=20.9265) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
3. `medquad_ans_0000916-5__dup1` (score=20.7116) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene...
4. `medquad_ans_0000916-4__dup1` (score=20.0161) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is...
5. `medquad_ans_0000145-3` (score=18.2021) - Mutations in the TGFB1 gene cause Camurati-Engelmann disease. The TGFB1 gene provides instructions for producing a protein called transforming growth factor beta-1 (TGF-1). The TGF-1 protein helps control the growth a...

**Answer**:

Yes, Camurati-Engelmann disease is inherited in an autosomal dominant fashion [medquad_ans_0000915-1__dup1].

**Supported citations**: `medquad_ans_0000915-1__dup1`
**Unsupported citations**: _none_

### Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.771, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000483-5` (score=34.3607) - These resources address the diagnosis or management of histiocytosis-lymphadenopathy plus syndrome: - Genetic Testing Registry: Histiocytosis-lymphadenopathy plus syndrome These resources from MedlinePlus offer inform...
2. `medquad_ans_0002895-1` (score=30.6375) - Histiocytosis-lymphadenopathy plus syndrome is a group of conditions with overlapping signs and symptoms that affect many parts of the body. This group of disorders includes H syndrome, pigmented hypertrichosis with i...
3. `medquad_ans_0000483-3` (score=30.4189) - Histiocytosis-lymphadenopathy plus syndrome is caused by mutations in the SLC29A3 gene, which provides instructions for making a protein called equilibrative nucleoside transporter 3 (ENT3). ENT3 belongs to a family o...
4. `medquad_ans_0000483-1` (score=28.4286) - Histiocytosis-lymphadenopathy plus syndrome (also known as SLC29A3 spectrum disorder) is a group of conditions with overlapping signs and symptoms that affect many parts of the body. This group of disorders includes H...
5. `medquad_ans_0002895-2` (score=28.4008) - What are the signs and symptoms of Histiocytosis-lymphadenopathy plus syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Histiocytosis-lymphadenopathy plus syndrome. If the in...

**Answer**:

The provided context does not specify the exact genetic changes related to histiocytosis-lymphadenopathy plus syndrome. It mentions that the conditions in the spectrum share a genetic cause but does not detail the specific genetic mutations or alterations [medquad_ans_0002895-1]. Therefore, based on the given information, the genetic changes are not described.

**Supported citations**: `medquad_ans_0002895-1`
**Unsupported citations**: _none_

### Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.956, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004456-1` (score=21.9178) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
2. `medquad_ans_0004450-1` (score=21.9178) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0004451-1` (score=21.9178) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004452-1` (score=21.9178) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004454-1` (score=21.9178) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

Yes, Noonan syndrome is typically inherited in an autosomal dominant manner, but many cases are due to a new mutation and are not inherited from an affected parent [medquad_ans_0004456-1].

**Supported citations**: `medquad_ans_0004456-1`
**Unsupported citations**: _none_

### Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Scores**: context_relevance=0.625, groundedness=0.710, answer_relevance=0.725

**Top docs**:

1. `medquad_ans_0001093-1` (score=40.4421) - 3-methylglutaconyl-CoA hydratase deficiency is an inherited condition that causes neurological problems. Beginning in infancy to early childhood, children with this condition often have delayed development of mental a...
2. `medquad_ans_0001093-3` (score=36.8501) - Mutations in the AUH gene cause 3-methylglutaconyl-CoA hydratase deficiency. This gene provides instructions for producing 3-methylglutaconyl-CoA hydratase, an enzyme that is involved in breaking down a protein buildi...
3. `medquad_ans_0001093-2` (score=35.4234) - 3-methylglutaconyl-CoA hydratase deficiency is a rare disorder; at least 20 cases have been reported in the scientific literature.
4. `medquad_ans_0001093-5` (score=27.3559) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Aci...
5. `medquad_ans_0000396-5` (score=20.7793) - These resources address the diagnosis or management of fumarase deficiency: - Gene Review: Gene Review: Fumarate Hydratase Deficiency - Genetic Testing Registry: Fumarase deficiency These resources from MedlinePlus of...

**Answer**:

The provided context does not mention any treatments for 3-methylglutaconyl-CoA hydratase deficiency. [medquad_ans_0001093-1] and [medquad_ans_0001093-3] describe the condition's symptoms, genetic cause, and biochemical mechanisms but do not address therapeutic interventions. Therefore, based on the given information, no treatments can be identified.

**Supported citations**: `medquad_ans_0001093-1`, `medquad_ans_0001093-3`
**Unsupported citations**: _none_

### Example 61 (`0003000-1`)

**Question**: What is (are) Hyperkalemic periodic paralysis ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000494-5` (score=39.4509) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis ...
2. `medquad_ans_0000494-2` (score=29.2575) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
3. `medquad_ans_0000494-1` (score=27.4874) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to...
4. `medquad_ans_0003000-2` (score=26.8753) - What are the signs and symptoms of Hyperkalemic periodic paralysis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hyperkalemic periodic paralysis. If the information is available, ...
5. `medquad_ans_0000126-1__dup1` (score=26.3574) - Familial periodic paralyses are a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. They are characterized by episodes in which the affect...

**Answer**:

Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms and legs. Episodes tend to increase in frequency until mid-adulthood, after which they occur less frequently. Factors that can trigger attacks include rest after exercise, potassium-rich foods such as bananas and potatoes, stress, fatigue, alcohol, pregnancy, exposure to cold temperatures, certain medications, and periods without food (fasting). Muscle strength usually returns to normal between attacks, although many affected people continue to experience mild stiffness (myotonia), particularly in muscles of the face and hands. Most people with hyperkalemic periodic paralysis have increased levels of potassium in their blood (hyperkalemia) during attacks. Hyperkalemia results when the weak or paralyzed muscles release potassium ions into the bloodstream. In other cases, attacks are associated with normal

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Scores**: context_relevance=0.900, groundedness=0.848, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000225-1` (score=49.7405) - Congenital fibrosis of the extraocular muscles is a disorder that affects the muscles that surround the eyes. These muscles control eye movement and the position of the eyes (for example, looking straight ahead). Cong...
2. `medquad_ans_0000225-5` (score=47.1877) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of...
3. `medquad_ans_0000225-4` (score=39.1936) - The different types of congenital fibrosis of the extraocular muscles have different patterns of inheritance. CFEOM1 and CFEOM3 are inherited in an autosomal dominant pattern, which means one copy of the altered gene ...
4. `medquad_ans_0006242-1` (score=37.6676) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
5. `medquad_ans_0000225-3` (score=36.1205) - CFEOM1 and rare cases of CFEOM3 result from mutations in the KIF21A gene. This gene provides instructions for making a protein called a kinesin, which is essential for the transport of materials within cells. Research...

**Answer**:

The provided context does not specify any treatments for congenital fibrosis of the extraocular muscles. It mentions resources related to diagnosis and management, such as genetic testing, diagnostic tests, genetic counseling, surgery and rehabilitation, drug therapy, and palliative care [medquad_ans_0000225-5], but does not detail specific treatments. Therefore, based on the given information, the treatments cannot be determined.

**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

### Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Scores**: context_relevance=0.800, groundedness=0.955, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000075-3__dup1` (score=16.9650) - The prognosis for individuals with chorea varies depending on the type of chorea and the associated disease. Huntington's disease is a progressive, and ultimately, fatal disease. Syndenham's chorea is treatable and cu...
2. `medquad_ans_0000491-1__dup2` (score=15.6997) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms...
3. `medquad_ans_0000075-2__dup2` (score=15.3246) - There is no standard course of treatment for chorea. Treatment depends on the type of chorea and the associated disease. Treatment for Huntington's disease is supportive, while treatment for Syndenham's chorea usually...
4. `medquad_ans_0000489-5` (score=14.2838) - These resources address the diagnosis or management of Huntington disease: - Gene Review: Gene Review: Huntington Disease - Genetic Testing Registry: Huntington's chorea - Huntington's Disease Society of America: HD C...
5. `medquad_ans_0000013-9__dup1` (score=14.1915) - A heritable disease is caused by a mutation, or harmful change, in a gene inherited from a parent. Genes are small structures in your body's cells that determine how you look and tell your body how to work. Examples o...

**Answer**:

Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms may include uncontrolled movements, clumsiness, and balance problems. Later, HD can lead to loss of the ability to walk, talk, and swallow, and some individuals may stop recognizing family members, while others remain aware of their environment and can express emotions. If one of your parents has Huntington's disease, you have a 50 percent chance of inheriting it. There is no cure; medicines can help manage symptoms but cannot slow or stop the disease. A blood test can determine if a person has the HD gene and will develop the disease, and genetic counseling can assist in understanding the risks and benefits of testing [medquad_ans_0000491-1__dup2].

**Supported citations**: `medquad_ans_0000491-1__dup2`
**Unsupported citations**: _none_

### Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.289

**Top docs**:

1. `medquad_ans_0000416-3` (score=40.6392) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
2. `medquad_ans_0000416-1` (score=37.3723) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with ...
3. `medquad_ans_0000416-2` (score=33.2877) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
4. `medquad_ans_0002601-1` (score=32.9157) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...
5. `medquad_ans_0000416-5` (score=32.4661) - These resources address the diagnosis or management of glutamate formiminotransferase deficiency: - Baby's First Test - Genetic Testing Registry: Glutamate formiminotransferase deficiency These resources from MedlineP...

**Answer**:

Approximately 20 affected individuals have been identified [medquad_ans_0002601-1].

**Supported citations**: `medquad_ans_0002601-1`
**Unsupported citations**: _none_

### Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.678, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0004453-1` (score=22.7583) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
2. `medquad_ans_0004456-1` (score=22.7583) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0004451-1` (score=22.7583) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004454-1` (score=22.7583) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004452-1` (score=22.7583) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

The provided context does not specify the number of people affected by Noonan syndrome. [medquad_ans_0004453-1][medquad_ans_0004456-1][medquad_ans_0004451-1][medquad_ans_0004454-1] describe the features, genetic causes, and inheritance pattern of Noonan syndrome but do not include epidemiological data on prevalence or affected population size. Therefore, the number of people affected by Noonan syndrome cannot be determined from the given information.

**Supported citations**: `medquad_ans_0004453-1`, `medquad_ans_0004456-1`, `medquad_ans_0004451-1`, `medquad_ans_0004454-1`
**Unsupported citations**: _none_

### Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Scores**: context_relevance=0.857, groundedness=0.989, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000339-5__dup1` (score=44.4780) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on t...
2. `medquad_ans_0000339-6` (score=44.4662) - A lice-killing lotion containing 1% permethrin or a mousse containing pyrethrins and piperonyl butoxide can be used to treat pubic ("crab") lice. These products are available over-the-counter without a prescription at...
3. `medquad_ans_0000339-7` (score=42.4513) - Pubic ("crab") lice most commonly are spread directly from person to person by sexual contact. Pubic lice very rarely may be spread by clothing, bedding, or a toilet seat. The following are steps that can be taken to ...
4. `medquad_ans_0000339-2__dup1` (score=41.2368) - Pubic ("crab") lice infestation is found worldwide and occurs in all races and ethnic groups and in all levels of society. Pubic lice usually are spread through sexual contact and are most common in adults. Occasional...
5. `medquad_ans_0000339-1__dup3` (score=40.9255) - Also called crab lice or "crabs," pubic lice are parasitic insects found primarily in the pubic or genital area of humans. Pubic lice infestation is found worldwide and occurs in all races, ethnic groups, and levels o...

**Answer**:

Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp). Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

### Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.956, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005571-2` (score=28.5414) - What are the signs and symptoms of Shapiro syndrome? Shapiro syndrome generally consists of three findings: spontaneous periodic hypothermia, excessive sweating, and agenesis of the corpus callosum. However, there has...
2. `medquad_ans_0005571-3` (score=26.8063) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
3. `medquad_ans_0005571-1` (score=22.5721) - Shapiro syndrome is a rare disease affecting about 50 people worldwide that is typically characterized by recurrent episodes of excessive sweating and hypothermia and the agenesis of the corpus callosum. The duration ...
4. `medquad_ans_0003103-4` (score=21.4654) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...
5. `medquad_ans_0000108-2__dup1` (score=19.0184) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The following treatments have been attempted for Shapiro syndrome: anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, or sympathectomy. However, responses to these treatments vary among individuals [medquad_ans_0005571-3].

**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

### Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.533, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002359-1` (score=20.4999) - FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. People wi...
2. `medquad_ans_0000373-2` (score=20.4425) - The prevalence of FG syndrome is unknown, although several hundred cases have been reported worldwide. Researchers suspect that FG syndrome may be overdiagnosed because many of its signs and symptoms are also seen wit...
3. `medquad_ans_0000373-5` (score=19.0604) - These resources address the diagnosis or management of FG syndrome: - Gene Review: Gene Review: MED12-Related Disorders - Genetic Testing Registry: FG syndrome - Genetic Testing Registry: FG syndrome 2 - Genetic Testi...
4. `medquad_ans_0000373-1` (score=18.3608) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects ...
5. `medquad_ans_0000373-3` (score=18.2594) - Researchers have identified changes in five regions of the X chromosome that are linked to FG syndrome in affected families. Mutations in a gene called MED12, which is located in one of these regions, appear to be the...

**Answer**:

FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. It is characterized by intellectual disability ranging from mild to severe, hypotonia, constipation and/or anal anomalies, a distinctive facial appearance, broad thumbs and great toes, relative macrocephaly, and abnormalities of the corpus callosum. Medical problems such as heart defects, seizures, undescended testicle, and inguinal hernia have also been reported. Mutations in the MED12 gene are the most common cause, leading to FG syndrome 1, and other genes involved include FLNA (FGS2), CASK (FGS4), UPF3B (FGS6), and BRWD3 (FGS7). FG syndrome is inherited in an X-linked recessive pattern [medquad_ans_0002

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.954, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000225-1` (score=24.4994) - Congenital fibrosis of the extraocular muscles is a disorder that affects the muscles that surround the eyes. These muscles control eye movement and the position of the eyes (for example, looking straight ahead). Cong...
2. `medquad_ans_0006242-1` (score=24.4863) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
3. `medquad_ans_0000225-4` (score=22.5489) - The different types of congenital fibrosis of the extraocular muscles have different patterns of inheritance. CFEOM1 and CFEOM3 are inherited in an autosomal dominant pattern, which means one copy of the altered gene ...
4. `medquad_ans_0000225-2` (score=21.5007) - CFEOM1 is the most common form of congenital fibrosis of the extraocular muscles, affecting at least 1 in 230,000 people. CFEOM1 and CFEOM3 have been reported worldwide, whereas CFEOM2 has been seen in only a few fami...
5. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The symptoms of Tukel syndrome include congenital fibrosis of the extraocular muscles, nonprogressive restrictive external ophthalmoplegia, restrictive external ophthalmoplegia, ptosis, postaxial oligodactyly, carpal bone aplasia, carpal synostosis, compensatory chin elevation, and syndactyly. These symptoms are based on the Human Phenotype Ontology data for Tukel syndrome [medquad_ans_0006242-1].

**Supported citations**: `medquad_ans_0006242-1`
**Unsupported citations**: _none_

### Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Scores**: context_relevance=1.000, groundedness=0.709, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000199-1__dup1` (score=21.3214) - The mucolipidoses (ML) are a group of inherited metabolic diseases that affect the bodys ability to carry out the normal turnover of various materials within cells. In ML, abnormal amounts of carbohydrates and fatty m...
2. `medquad_ans_0003103-4` (score=19.3836) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...
3. `medquad_ans_0000108-2__dup1` (score=19.0184) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
4. `medquad_ans_0000549-1__dup1` (score=18.7510) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...
5. `medquad_ans_0001853-3` (score=18.7075) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

The provided context does not contain information about treatments for Mucolipidoses. Therefore, it is not possible to answer the question based on the given text. [medquad_ans_0000199-1__dup1] discusses the causes, symptoms, and genetic basis of Mucolipidoses but does not mention any available treatments.

**Supported citations**: `medquad_ans_0000199-1__dup1`
**Unsupported citations**: _none_

### Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000829-5` (score=49.6132) - These resources address the diagnosis or management of progressive familial intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Progressive familial intrahepatic cholest...
2. `medquad_ans_0005084-2` (score=43.0918) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis...
3. `medquad_ans_0000118-1` (score=40.5406) - Benign recurrent intrahepatic cholestasis (BRIC) is characterized by episodes of liver dysfunction called cholestasis. During these episodes, the liver cells have a reduced ability to release a digestive fluid called ...
4. `medquad_ans_0005084-1` (score=37.2170) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching...
5. `medquad_ans_0001157-1` (score=36.2740) - What are the signs and symptoms of Cholestasis, progressive familial intrahepatic 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Cholestasis, progressive familial intrahepatic 4....

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 include autosomal recessive inheritance, cirrhosis, conjugated hyperbilirubinemia, death in childhood, diarrhea, elevated alkaline phosphatase, failure to thrive, fat malabsorption, hepatocellular carcinoma, hepatomegaly, infantile onset, intermittent jaundice, intrahepatic cholestasis, pruritus, short stature, and splenomegaly [medquad_ans_0005084-2].

**Supported citations**: `medquad_ans_0005084-2`
**Unsupported citations**: _none_

### Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Scores**: context_relevance=0.800, groundedness=0.761, answer_relevance=0.960

**Top docs**:

1. `medquad_ans_0000070-7` (score=32.3095) - Urinary incontinence means a person leaks urine by accident. Urinary incontinence is a common bladder problem as people age. Women are more likely than men to leak urine. If this problem is happening to you, there is ...
2. `medquad_ans_0000070-1__dup5` (score=28.2695) - Urinary incontinence means a person leaks urine by accident. Urinary incontinence is a common bladder problem as people age. Women are more likely than men to leak urine. If this problem is happening to you, there is ...
3. `medquad_ans_0000070-9` (score=27.5295) - Leaking urine can happen for many reasons. Sometimes urinary incontinence happens for a little while. Short periods of leaking urine can happen because of - urinary tract infections - constipation - some medicines. ur...
4. `medquad_ans_0000070-8` (score=25.7386) - There are different types of urinary incontinence. Stress urinary incontinence happens when urine leaks as pressure is put on the bladder, for example, during exercise, coughing, sneezing, laughing, or lifting heavy o...
5. `medquad_ans_0000211-6` (score=25.1658) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he i...

**Answer**:

The provided context does not specify how many people are affected by urinary incontinence in men. While it states that women are more likely than men to leak urine, no quantitative data or statistics regarding the number of men affected are given [medquad_ans_0000070-7][medquad_ans_0000070-1__dup5]. Therefore, the exact number of affected men cannot be determined from the available information.

**Supported citations**: `medquad_ans_0000070-7`, `medquad_ans_0000070-1__dup5`
**Unsupported citations**: _none_

### Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005415-4` (score=22.4043) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...
2. `medquad_ans_0005415-1` (score=22.2335) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
3. `medquad_ans_0000882-1` (score=20.3987) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
4. `medquad_ans_0005415-3` (score=20.3634) - How is Robinow syndrome inherited? Robinow syndrome may be inherited in an autosomal recessive or autosomal dominant manner. Autosomal recessive (AR) inheritance means both copies of the responsible gene in each cell ...
5. `medquad_ans_0000882-3` (score=19.4220) - Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene. This gene provides instructions for making a protein whose function is not well understood, although it is involved in chemical signaling p...

**Answer**:

Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Two major types have been described: autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome. These types are distinguished by the severity of their signs and symptoms and by their pattern of inheritance. Autosomal recessive Robinow syndrome is characterized by skeletal abnormalities such as shortening of the long bones in the arms and legs, abnormally short fingers and toes (brachydactyly), wedge-shaped spinal bones (hemivertebrae) leading to kyphoscoliosis, fused or missing ribs, short stature, and distinctive facial features. Other features may include underdeveloped genitalia, dental problems, kidney or heart defects, or delayed development. This form is caused by mutations in the ROR2 gene. Autosomal dominant Robinow syndrome causes milder, but similar, features with rarely spine and rib abnormalities and less

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 74 (`0000551-1`)

**Question**: What is (are) juvenile primary lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.927, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003331-3` (score=37.6174) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...
2. `medquad_ans_0003331-4` (score=36.3958) - How is juvenile primary lateral sclerosis inherited? Juvenile primary lateral sclerosis is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of a...
3. `medquad_ans_0003331-1` (score=35.5593) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
4. `medquad_ans_0000551-2` (score=34.2017) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.
5. `medquad_ans_0000551-1` (score=32.4375) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. The features of this disorder are caused by damage to moto...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness (spasticity) of muscles in the arms, legs, and face. It is caused by mutations in the ALS2 gene, which leads to the production of an unstable or nonfunctional protein called alsin. This protein is abundant in motor neurons, and its dysfunction results in damage to these neurons, which control muscle movement. Symptoms begin in early childhood and progress slowly over many years, with early signs including clumsiness, muscle weakness, and spasticity in the legs, and difficulty with balance. As the condition advances, spasticity spreads to the arms and hands, and individuals may develop slurred speech, drooling, difficulty swallowing, and an inability to walk. The disorder is inherited in an autosomal recessive pattern, meaning both copies of the ALS2 gene in each cell must have mutations [medquad_ans_0003331-1]

**Supported citations**: `medquad_ans_0003331-1`
**Unsupported citations**: _none_

### Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Scores**: context_relevance=1.000, groundedness=0.901, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002934-3` (score=14.9767) - What causes human T-cell leukemia virus, type 1? Human T-cell leukemia virus, type 1 (HTLV-1) occurs when a person is infected by the human T-cell leukemia retrovirus. HTLV-1 is spread by blood transfusions, sexual co...
2. `medquad_ans_0002935-3` (score=14.6413) - What causes human T-cell leukemia virus, type 2? Human T-cell leukemia virus, type 2 (HTLV-2) occurs when a person is infected by the human T-cell leukemia retrovirus. HTLV-2 is spread by blood transfusions, sexual co...
3. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
4. `medquad_ans_0004252-2` (score=13.5026) - What causes myelodysplastic/myeloproliferative disease? In most cases, the cause of myelodysplastic/myeloproliferative disease is unknown, and there is limited information regarding potential causes. No specific genet...
5. `medquad_ans_0000001_5-1` (score=13.0152) - Key Points - Hairy cell leukemia is a type of cancer in which the bone marrow makes too many lymphocytes (a type of white blood cell). - Leukemia may affect red blood cells, white blood cells, and platelets. - Gender ...

**Answer**:

The provided context does not define "leukemia" directly. However, it mentions several types of leukemia, such as adult T-cell leukemia (ATL), chronic myelomonocytic leukemia (CMML), and juvenile myelomonocytic leukemia (JMML), which are associated with human T-cell leukemia virus type 1 (HTLV-1) or HTLV-2, or with environmental and genetic factors.

Specifically:

- Adult T-cell leukemia (ATL) is a type of leukemia caused by HTLV-1 infection [medquad_ans_0002934-3].
- Chronic myelomonocytic leukemia (CMML) is a myelodysplastic/myeloproliferative disease with an unknown cause, though exposure to carcinogens, radiation, and cytotoxic agents has been associated [medquad_ans_0004252-2].
- Juvenile my

**Supported citations**: `medquad_ans_0002934-3`, `medquad_ans_0004252-2`
**Unsupported citations**: _none_

### Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Scores**: context_relevance=1.000, groundedness=0.926, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002485-1` (score=17.4281) - Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. Gangliocytomas may occur in all age groups but most often occur in people between the ages of 10 and 30. The most common si...
2. `medquad_ans_0002485-2` (score=14.8553) - What are the signs and symptoms of gangliocytomas? Signs and symptoms caused by the presence of a gangliocytoma can vary depending on the tumor's location. Seizures are the most common symptom. Other symptoms may incl...
3. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
4. `medquad_ans_0000108-2__dup1` (score=11.5977) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
5. `medquad_ans_0000549-1__dup1` (score=11.3985) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...

**Answer**:

Gangliocytoma is a rare type of central nervous system (CNS) tumor composed of mature neurons. It may occur in all age groups but most commonly affects individuals between the ages of 10 and 30. The most frequent site is the temporal lobe of the brain, although it can arise elsewhere in the CNS, including the cerebellum, brainstem, floor of the third ventricle, and spinal cord. Gangliocytomas are often associated with epilepsy and may present with signs and symptoms such as seizures (most common), increased brain pressure, endocrine disorders, and focal symptoms. They are generally slow-growing and non-malignant. Treatment typically involves surgical removal of the tumor [medquad_ans_0002485-1].

**Supported citations**: `medquad_ans_0002485-1`
**Unsupported citations**: _none_

### Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.988, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000130-1__dup1` (score=33.4458) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of ...
2. `medquad_ans_0000130-2__dup1` (score=29.8160) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportiv...
3. `medquad_ans_0000130-3__dup1` (score=27.6155) - The prognosis for most individuals with Miller Fisher syndrome is good. In most cases, recovery begins within 2 to 4 weeks of the onset of symptoms, and may be almost complete within 6 months. Some individuals are lef...
4. `medquad_ans_0000443-2` (score=25.2028) - The prevalence of Guillain-Barr syndrome is estimated to be 6 to 40 cases per 1 million people. The occurrence of the different types of Guillain-Barr syndrome varies across regions. AIDP is the most common type in No...
5. `medquad_ans_0000130-4__dup1` (score=25.0202) - The NINDS supports research aimed at discovering new ways to diagnose, treat, and, ultimately, cure neuropathies such as Miller Fisher syndrome.

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, absence of tendon reflexes, generalized muscle weakness, and may be preceded by a viral illness. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder [medquad_ans_0000130-1__dup1].

**Supported citations**: `medquad_ans_0000130-1__dup1`
**Unsupported citations**: _none_

### Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006570-3` (score=27.4992) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location desig...
2. `medquad_ans_0001084-1` (score=24.0288) - 22q11.2 deletion syndrome (which is also known by several other names, listed below) is a disorder caused by the deletion of a small piece of chromosome 22. The deletion occurs near the middle of the chromosome at a l...
3. `medquad_ans_0006570-4` (score=22.4404) - Is 22q11.2 deletion syndrome inherited? Most cases of 22q11.2 deletion syndrome are not inherited from a parent and are caused by a random error during the formation of egg or sperm cells, or during early fetal develo...
4. `medquad_ans_0001084-4` (score=22.2696) - The inheritance of 22q11.2 deletion syndrome is considered autosomal dominant because a deletion in one copy of chromosome 22 in each cell is sufficient to cause the condition. Most cases of 22q11.2 deletion syndrome ...
5. `medquad_ans_0001084-5` (score=22.1327) - These resources address the diagnosis or management of 22q11.2 deletion syndrome: - Gene Review: Gene Review: 22q11.2 Deletion Syndrome - Genetic Testing Registry: Asymmetric crying face association - Genetic Testing ...

**Answer**:

22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location designated q11.2. Most people with 22q11.2 deletion syndrome are missing a piece of the chromosome that contains about 30 to 40 genes, many of which have not been well characterized. Some affected people have smaller deletions. The deletion of a particular gene, TBX1, is probably responsible for many of the syndrome's characteristic signs (such as heart defects, a cleft palate, distinctive facial features, hearing loss, and low calcium levels). Loss of this gene may also contribute to behavioral problems. The loss of another gene, COMT, may also cause increased risk of behavioral problems and mental illness in affected people. The other genes that are deleted likely contribute to the various features of

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000062-1__dup5` (score=15.3456) - Shingles is a painful rash that develops on one side of the face or body. The rash forms blisters that typically scab over in 7 to 10 days and clear up within 2 to 4 weeks. Most commonly, the rash occurs in a single s...
2. `medquad_ans_0000062-3__dup2` (score=15.2200) - A Vaccine for Adults 60 and Older In May 2006, the U.S. Food and Drug Administration approved a vaccine (Zostavax) to prevent shingles in people age 60 and older. The vaccine is designed to boost the immune system and...
3. `medquad_ans_0000812-1__dup2` (score=14.3933) - Shingles is a disease caused by the varicella-zoster virus - the same virus that causes chickenpox. After you have chickenpox, the virus stays in your body. It may not cause problems for many years. As you get older, ...
4. `medquad_ans_0000148-1__dup2` (score=13.6222) - Shingles (herpes zoster) is an outbreak of rash or blisters on the skin that is caused by the same virus that causes chickenpox the varicella-zoster virus. The first sign of shingles is often burning or tingling pain ...
5. `medquad_ans_0000062-25` (score=13.5555) - Yes. In May 2006, the U.S. Food and Drug Administration approved a vaccine to prevent shingles in people age 60 and older. The vaccine, called Zostavax, is designed to boost the immune system and protect older adults ...

**Answer**:

Shingles is caused by a reactivation of the chickenpox virus. It is distinctive because it affects only one side of the body. The virus that causes shingles is a herpes virus (Another name for shingles is herpes zoster). Once you are infected with this kind of virus, it remains in your body for life. It stays inactive until a period when your immunity is down [medquad_ans_0000062-1__dup5].

**Supported citations**: `medquad_ans_0000062-1__dup5`
**Unsupported citations**: _none_

### Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.853, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0002456-1` (score=44.9671) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the ta...
2. `medquad_ans_0000392-1` (score=25.7965) - Fryns syndrome is a condition that affects the development of many parts of the body. The features of this disorder vary widely among affected individuals and overlap with the signs and symptoms of several other disor...
3. `medquad_ans_0000392-5` (score=25.1567) - These resources address the diagnosis or management of Fryns syndrome: - Children's Hospital of Philadelphia: Treatment of Congenital Diaphragmatic Hernia - Gene Review: Gene Review: Fryns Syndrome - Genetic Testing R...
4. `medquad_ans_0000392-2` (score=25.1141) - The worldwide incidence of Fryns syndrome is unknown. More than 50 affected individuals have been reported in the medical literature. Studies suggest that Fryns syndrome occurs in 1.3 to 10 percent of all cases of con...
5. `medquad_ans_0002458-3` (score=25.0575) - How is Fryns syndrome inherited? Although the exact cause of Fryns syndrome is not currently known (and no disease-causing gene has yet been identified), it is thought to be genetic because it tends to "run in familie...

**Answer**:

The symptoms of Fryns Hofkens Fabry syndrome include ulnar deviation of the finger (seen in 90% of patients), distal ulnar hypoplasia, dysplastic radii, hypoplasia of the radius, mesomelic arm shortening, radial bowing, and ulnar deviation of the hand. These signs and symptoms are based on data from the Human Phenotype Ontology, which compiles information from sources such as Orphanet [medquad_ans_0002456-1].

Note: There appears to be a confusion in the name "Fryns Hofkens Fabry syndrome" as Fryns syndrome and Fabry syndrome are distinct conditions. The provided context refers to symptoms associated with Fryns syndrome and mentions "Fryns Hofkens Fabry syndrome" without clear evidence that this is a recognized or valid syndrome. The symptoms listed are more consistent with Fryns

**Supported citations**: `medquad_ans_0002456-1`
**Unsupported citations**: _none_

### Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Scores**: context_relevance=0.900, groundedness=0.871, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000030_1-7` (score=26.4074) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000007_2-6` (score=26.0332) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000009_2-8` (score=25.1102) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000032_1-8` (score=24.9566) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000006_4-7` (score=24.7930) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not contain specific information about research or clinical trials being conducted for osteoporosis. While it mentions chemotherapy is given to reduce back pain from osteoporosis or compression fractures of the spine and bisphosphonate therapy is used to slow bone loss and reduce bone pain, there is no detail on ongoing clinical trials or research specifically targeting osteoporosis. [medquad_ans_0000030_1-7]

**Supported citations**: `medquad_ans_0000030_1-7`
**Unsupported citations**: _none_

### Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Scores**: context_relevance=1.000, groundedness=0.890, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000430-1__dup1` (score=21.4204) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. ...
2. `medquad_ans_0000006-18` (score=14.2797) - You can help your doctor make a diagnosis by writing down key information about your dizziness or balance problem beforehand and giving the information to your doctor during the visit. Write down answers to these ques...
3. `medquad_ans_0000286-1__dup2` (score=13.9997) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...
4. `medquad_ans_0000925-1__dup1` (score=13.2892) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
5. `medquad_ans_0000057-22` (score=12.8434) - You can help your doctor make a diagnosis by writing down important information about your problem beforehand and giving the information to your doctor during your visit. Write down answers to the following questions....

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. These scams often promise quick fixes, miracle cures, ancient remedies, secret ingredients, or scientific breakthroughs for common health issues such as weight loss, memory loss, sexual performance, and joint pain. They target individuals with serious conditions like cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. Red flags include claims of a miracle cure, quick fix, ancient remedy, secret ingredient, or scientific breakthrough. To protect yourself, it is advised to consult a doctor or healthcare professional before taking unproven treatments, especially when using prescription drugs. The Food and Drug Administration (FDA) is mentioned as a source of information on this topic [medquad_ans_0000430-1__dup1].

**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

### Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Scores**: context_relevance=0.917, groundedness=0.833, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000061-6__dup1` (score=37.9371) - People with diabetes can lower their risk of sexual and urologic problems by keeping their blood glucose, blood pressure, and cholesterol levels close to the target numbers their health care provider recommends. Being...
2. `medquad_ans_0000015-8` (score=32.3325) - Diabetes cannot be cured, but it can be managed. Managing blood glucose (blood sugar) as well as blood pressure and cholesterol is the best defense against the serious complications of diabetes. Know What To Do Every ...
3. `medquad_ans_0000054-1__dup4` (score=29.6927) - Diabetes problems are health problems that can happen when you have diabetes. If your diabetes is not under control, you will have too much glucose\*, also called sugar, in your blood. Having too much glucose in your b...
4. `medquad_ans_0000038-2__dup2` (score=28.9640) - Risk Factors Diabetes and high blood pressure are the two leading causes of kidney disease. Both diabetes and high blood pressure damage the small blood vessels in your kidneys and can cause kidney disease -- without ...
5. `medquad_ans_0000274-1__dup2` (score=28.9261) - Summary : If you have diabetes, your body cannot make or properly use insulin. This leads to high blood glucose, or blood sugar, levels. Healthy eating helps keep your blood sugar in your target range. It is a critica...

**Answer**:

The provided context does not mention the symptoms of "Prevent diabetes problems: Keep your eyes healthy." Therefore, no answer can be given based on the available information. [medquad_ans_0000061-6__dup1] and [medquad_ans_0000015-8] discuss general diabetes management and prevention of sexual and urologic problems, as well as blood glucose, blood pressure, and cholesterol control, but do not address eye health symptoms.

**Supported citations**: `medquad_ans_0000061-6__dup1`, `medquad_ans_0000015-8`
**Unsupported citations**: _none_

### Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.948, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000001-4` (score=20.6047) - Aarskog-Scott syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chro...
2. `medquad_ans_0000001-1` (score=19.8667) - Aarskog-Scott syndrome is a genetic disorder that affects the development of many parts of the body. This condition mainly affects males, although females may have mild features of the syndrome. People with Aarskog-Sc...
3. `medquad_ans_0000001-2` (score=19.8297) - Aarskog-Scott syndrome is believed to be a rare disorder; however, its prevalence is unknown because mildly affected people are often not diagnosed.
4. `medquad_ans_0000001-3` (score=19.7995) - Mutations in the FGD1 gene cause some cases of Aarskog-Scott syndrome. The FGD1 gene provides instructions for making a protein that turns on (activates) another protein called Cdc42, which transmits signals that are ...
5. `medquad_ans_0000001-5` (score=14.5667) - These resources address the diagnosis or management of Aarskog-Scott syndrome: - Genetic Testing Registry: Aarskog syndrome - MedlinePlus Encyclopedia: Aarskog syndrome These resources from MedlinePlus offer informati...

**Answer**:

Yes, Aarskog-Scott syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition, FGD1, is located on the X chromosome. In males, who have only one X chromosome, one altered copy of the gene is sufficient to cause the condition. In females, who have two X chromosomes, a mutation would need to occur in both copies of the gene to cause the syndrome, making it less common in females [medquad_ans_0000001-4]. This pattern of inheritance explains why males are affected more frequently than females [medquad_ans_0000001-4].

**Supported citations**: `medquad_ans_0000001-4`
**Unsupported citations**: _none_

### Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Scores**: context_relevance=0.900, groundedness=0.980, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0006525-1` (score=37.9564) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin. People with X-linked sideroblastic anemia have mature red blood cells that...
2. `medquad_ans_0001063-1` (score=37.6924) - X-linked sideroblastic anemia and ataxia is a rare condition characterized by a blood disorder called sideroblastic anemia and movement problems known as ataxia. This condition occurs only in males. Sideroblastic anem...
3. `medquad_ans_0001062-1` (score=34.6954) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin, which is the protein that carries oxygen in the blood. People with X-linke...
4. `medquad_ans_0001063-4` (score=33.3107) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
5. `medquad_ans_0001063-5` (score=33.2418) - These resources address the diagnosis or management of X-linked sideroblastic anemia and ataxia: - Gene Review: Gene Review: X-Linked Sideroblastic Anemia and Ataxia - Genetic Testing Registry: Anemia sideroblastic an...

**Answer**:

The genetic changes related to X-linked sideroblastic anemia are mutations in the ALAS2 gene. In rare cases, mutations are also found in both the HFE gene and the ALAS2 gene, resulting in a more severe form of the disorder [medquad_ans_0006525-1].

**Supported citations**: `medquad_ans_0006525-1`
**Unsupported citations**: _none_

### Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.504, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000602-3` (score=33.6750) - The various forms of limb-girdle muscular dystrophy are caused by mutations in many different genes. These genes provide instructions for making proteins that are involved in muscle maintenance and repair. Some of the...
2. `medquad_ans_0003604-1` (score=28.9127) - Limb-girdle muscular dystrophy type 2A (LGMD2A) is an autosomal recessive limb-girdle muscular dystrophy characterized by progressive, symmetrical weakness of the proximal limb and girdle muscles (mainly those around ...
3. `medquad_ans_0000602-5` (score=27.8496) - These resources address the diagnosis or management of limb-girdle muscular dystrophy: - Cleveland Clinic - Gene Review: Gene Review: Limb-Girdle Muscular Dystrophy Overview - Genetic Testing Registry: Limb-girdle mus...
4. `medquad_ans_0003610-1` (score=27.2669) - Limb-girdle muscular dystrophy type 2C (LGMD2C) is a condition that affects the muscles and is caused by mutations in the gamma-sarcoglycan gene. This condition belongs to a group of muscle disorders called limb-girdl...
5. `medquad_ans_0003610-3` (score=27.0626) - What treatment is available for limb-girdle muscular dystrophy? There is no specific treatment for limb-girdle muscular dystrophy. Management of the condition is based on the person's symptoms and subtype (if known). ...

**Answer**:

Limb-girdle muscular dystrophy is a group of genetic disorders characterized by progressive weakness of the proximal limb and girdle muscles, primarily those around the hips and shoulders, due to mutations in various genes. These mutations affect proteins involved in muscle maintenance and repair, including those that assemble into larger protein complexes to maintain muscle integrity and enable muscle contraction, or those involved in cell signaling, membrane repair, or removal of toxic wastes from muscle cells. The disorder is classified based on inheritance pattern and genetic cause, with types such as limb-girdle muscular dystrophy type 1B (caused by LMNA gene mutations), type 1C (caused by CAV3 gene mutations), type 2A (caused by CAPN3 gene mutations), and others resulting from mutations in genes like DYSF, SGCA/SGCB/SGCG/SGCD, TTN, ANO5, and others. Type 2

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Scores**: context_relevance=1.000, groundedness=0.883, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000016-2__dup2` (score=20.7390) - Who Is at Risk? All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes ...
2. `medquad_ans_0000016-6` (score=19.9558) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
3. `medquad_ans_0000016-1__dup3` (score=19.5355) - Can Cause Vision Loss, Blindness Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina in the back of the eye. A ...
4. `medquad_ans_0000275-1__dup1` (score=18.8475) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
5. `medquad_ans_0000343-1__dup1` (score=18.0776) - Some eye problems are minor and don't last long. But some can lead to a permanent loss of vision. Common eye problems include - Refractive errors - Cataracts - clouded lenses - Glaucoma - a disorder caused by damage t...

**Answer**:

Diabetic eye problems include diabetic retinopathy, cataract, and glaucoma. Diabetic retinopathy occurs when diabetes damages the tiny blood vessels inside the retina, leading to potential vision loss or blindness. It can progress through stages, with fragile abnormal blood vessels leaking blood into the eye (proliferative retinopathy) or fluid leaking into the macula (macular edema), both of which can blur vision. People with diabetes are at increased risk for these conditions, and a comprehensive dilated eye exam is recommended at least once a year to detect and manage them [medquad_ans_0000016-2\_\_dup2][medquad_ans_0000016-6][medquad_ans_0000016-1__dup3].

**Supported citations**: `medquad_ans_0000016-2__dup2`, `medquad_ans_0000016-6`, `medquad_ans_0000016-1__dup3`
**Unsupported citations**: _none_

### Example 88 (`0006449-1`)

**Question**: What is (are) Wilson disease ?

**Scores**: context_relevance=1.000, groundedness=0.461, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000133-13` (score=18.8697) - - Wilson disease is a genetic disease that prevents the body from removing extra copper. - Normally, the liver filters extra copper and releases it into bile. In Wilson disease, the liver does not filter copper correc...
2. `medquad_ans_0006449-1` (score=17.7172) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver,...
3. `medquad_ans_0006449-3` (score=17.6792) - What causes Wilson disease? Wilson disease is caused by changes (mutations) in the ATP7B gene. This gene encodes a protein that plays an important role in the transport of copper from the liver to the rest of the body...
4. `medquad_ans_0006449-5` (score=16.7185) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lif...
5. `medquad_ans_0001037-1` (score=16.3776) - Wilson disease is an inherited disorder in which excessive amounts of copper accumulate in the body, particularly in the liver, brain, and eyes. The signs and symptoms of Wilson disease usually first appear between th...

**Answer**:

Wilson disease is a rare inherited disorder characterized by the accumulation of copper in the body due to mutations in the ATP7B gene. This gene encodes a protein responsible for transporting copper from the liver to the rest of the body and removing excess copper from the body. When the ATP7B gene is mutated, the protein does not function properly, leading to copper buildup in tissues and organs such as the liver, brain, and eyes. High levels of copper are toxic and can cause damage to these organs, resulting in signs and symptoms including chronic liver disease, central nervous system abnormalities, and psychiatric disturbances. Wilson disease is inherited in an autosomal recessive manner, meaning a child must inherit the mutation from both parents to be at increased risk. Although there is no cure, treatment aims to reduce and control copper levels in the body through medications, dietary changes (such as avoiding high-copper foods like shellfish, liver, mushrooms, nuts, and chocolate), and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 89 (`0001503-2`)

**Question**: What are the symptoms of Congenital dyserythropoietic anemia type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.802, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000307-1` (score=38.5423) - Dyserythropoietic anemia and thrombocytopenia is a condition that affects blood cells and primarily occurs in males. A main feature of this condition is a type of anemia called dyserythropoietic anemia, which is chara...
2. `medquad_ans_0001504-1` (score=36.9221) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital dyserythropoietic anemia type 3. If the info...
3. `medquad_ans_0000223-5` (score=35.6135) - These resources address the diagnosis or management of CDA: - Gene Review: Gene Review: Congenital Dyserythropoietic Anemia Type I - Genetic Testing Registry: Congenital dyserythropoietic anemia, type I - Genetic Test...
4. `medquad_ans_0001503-2` (score=32.7263) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to abs...
5. `medquad_ans_0001503-3` (score=32.1705) - How might congenital dyserythropoietic anemia (CDA) type 2 be treated? The goal of CDA type 2 treatment is to address and prevent complications from anemia and iron overload. Most people with CDA type 2 develop iron o...

**Answer**:

The provided context does not contain information about the symptoms of Congenital dyserythropoietic anemia type 2. Therefore, a specific answer cannot be generated based on the given text. [medquad_ans_0000307-1]

**Supported citations**: `medquad_ans_0000307-1`
**Unsupported citations**: _none_

### Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.757, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
2. `medquad_ans_0000038-22` (score=12.7524) - During your next health care visit, talk to your provider about your test results and how to manage your kidney disease. Below is a list of questions you may want to ask. Add any questions you think are missing, and m...
3. `medquad_ans_0000108-2__dup1` (score=12.3473) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
4. `medquad_ans_0000549-1__dup1` (score=12.1314) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...
5. `medquad_ans_0000136-12` (score=12.0403) - You and your doctor will work together to choose a treatment that's best for you. The publications of the NIDDK Kidney Failure Series can help you learn about the specific issues you will face. Booklets - What I need ...

**Answer**:

The provided context does not define or describe kidney disease explicitly. It discusses questions to ask during a healthcare visit related to kidney health, test results, and management, but does not offer a definition or explanation of kidney disease itself. Therefore, based on the given information, it is not possible to answer the question about what kidney disease is.

[medquad_ans_0000038-22]

**Supported citations**: `medquad_ans_0000038-22`
**Unsupported citations**: _none_

### Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Scores**: context_relevance=0.857, groundedness=0.923, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000036_1-6` (score=29.2047) - Key Points - There are different types of treatment for patients with prostate cancer. - Seven types of standard treatment are used: - Watchful waiting or active surveillance - Surgery - Radiation therapy and radiopha...
2. `medquad_ans_0000058-23` (score=28.6670) - There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But ...
3. `medquad_ans_0000036_1-7` (score=27.5852) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000750-1__dup1` (score=27.4654) - The prostate is the gland below a man's bladder that produces fluid for semen. Prostate cancer is common among older men. It is rare in men younger than 40. Risk factors for developing prostate cancer include being ov...
5. `medquad_ans_0000036_2-3` (score=27.1905) - Key Points - Avoiding risk factors and increasing protective factors may help prevent cancer. - The following risk factors may increase the risk of prostate cancer: - Age - Family history of prostate cancer - Race - H...

**Answer**:

The treatments for prostate cancer include: watchful waiting or active surveillance, surgery, radiation therapy and radiopharmaceutical therapy, hormone therapy, chemotherapy, biologic therapy, bisphosphonate therapy, and treatments for bone pain caused by bone metastases or hormone therapy. Additionally, new treatments such as cryosurgery, high-intensity focused ultrasound therapy, and proton beam radiation therapy are being tested in clinical trials. Patients may also consider participating in clinical trials, which are research studies aimed at improving current treatments or obtaining information on new treatments [medquad_ans_0000036_1-6].

**Supported citations**: `medquad_ans_0000036_1-6`
**Unsupported citations**: _none_

### Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Scores**: context_relevance=0.857, groundedness=0.507, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000036-4__dup3` (score=22.8984) - Insurance companies sell private health insurance plans. Two types of private health insurance are - Group health insurance. People may be eligible to purchase group health insurance through their employer or union or...
2. `medquad_ans_0000036-16` (score=20.2687) - - Diabetes management and treatment is expensive. Many people who have diabetes need help paying for their care. For those who qualify, a variety of government and nongovernment programs can help cover health care exp...
3. `medquad_ans_0000036-2__dup3` (score=20.0851) - Health insurance helps pay for medical care, including the cost of diabetes care. Health insurance options include the following: - private health insurance, which includes group and individual health insurance - gove...
4. `medquad_ans_0000005-19` (score=19.8440) - Most insurance plans, including health maintenance organizations (HMOs), will cover treatment for anxiety disorders. Check with your insurance company and find out. If you dont have insurance, the Health and Human Ser...
5. `medquad_ans_0000362-1__dup2` (score=18.9586) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...

**Answer**:

Yes, I have information about health insurance. Insurance companies sell private health insurance plans, which include group health insurance and individual health insurance. Group health insurance is available through employers, unions, family members' employers, or other organizations like professional or alumni groups. Individual health insurance allows people to purchase coverage for themselves and their families, and information about such plans can be found on HealthCare.gov, which also provides a search function called the Health Insurance Marketplace to find options by state. Depending on income and family size, some individuals may qualify for lower-cost premiums through the Marketplace. People can select or change individual plans during the annual open enrollment period, and life events may allow enrollment outside this period. Employers may have a waiting period of no more than 90 days before employees can enroll, and certain health plans like HMOs may have an affiliation period of no longer than three months. The ACA expanded access to preventive services, such as diabetes screening for adults

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 93 (`0003860-1`)

**Question**: What is (are) MECP2 duplication syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.936, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003860-3` (score=32.3159) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
2. `medquad_ans_0003860-5` (score=30.3941) - How is MECP2 duplication syndrome diagnosed? A diagnosis of MECP2 duplication syndrome is often suspected based on the presence of characteristic signs and symptoms. Genetic testing can then be ordered to confirm the ...
3. `medquad_ans_0003860-4` (score=30.2420) - Is MECP2 duplication syndrome inherited? MECP2 duplication syndrome is inherited in an X-linked manner. A condition is considered X-linked if the genetic change that causes the condition is located on the X chromosome...
4. `medquad_ans_0000635-4` (score=29.9726) - MECP2 duplication syndrome is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome...
5. `medquad_ans_0003860-1` (score=29.8683) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...

**Answer**:

MECP2 duplication syndrome is a genetic disorder caused by an extra copy (duplication) of the MECP2 gene in each cell, typically located on the long (q) arm of the X chromosome [medquad_ans_0003860-3]. This duplication results in the production of excess MECP2 protein, which disrupts the normal regulation of other genes in the brain, leading to irregular brain activity and the associated signs and symptoms of the condition [medquad_ans_0003860-3]. The syndrome is inherited in an X-linked manner; in males, a duplication of the MECP2 gene is sufficient to cause the disorder, while in females, a duplication of one copy of the gene usually does not cause symptoms due to random X-inactivation, although skewed X-inactivation may inactivate the abnormal chromosome and prevent disease expression [medquad_ans_0003860-4

**Supported citations**: `medquad_ans_0003860-3`
**Unsupported citations**: _none_

### Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Scores**: context_relevance=0.833, groundedness=0.825, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000028_2-1` (score=23.6087) - Key Points - Melanoma is a disease in which malignant (cancer) cells form in melanocytes (cells that color the skin). - There are different types of cancer that start in the skin. - Melanoma can occur anywhere on the ...
2. `medquad_ans_0005046-1` (score=22.4797) - Primary melanoma of the gastrointestinal (GI) tract refers to a melanoma starting in the stomach, intestines, salivary glands, mouth, esophagus, liver, pancreas, gallbladder, or rectum. Melanoma is a disease in which ...
3. `medquad_ans_0003897-1` (score=22.4654) - What are the signs and symptoms of Melanoma astrocytoma syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Melanoma astrocytoma syndrome. If the information is available, the ...
4. `medquad_ans_0000063-8` (score=21.7026) - Many Areas of Research Scientists are constantly searching for new ways to detect skin cancer, assess risk, and predict patient outcomes. They are interested in finding new treatments and new ways to deliver drugs and...
5. `medquad_ans_0000028_5-1` (score=21.6958) - Key Points - Skin cancer is a disease in which malignant (cancer) cells form in the tissues of the skin. - Nonmelanoma skin cancer is the most common cancer in the United States. - Being exposed to ultraviolet radiati...

**Answer**:

The provided context does not contain specific information about the outlook for melanoma, such as prognosis or survival rates. While it mentions that certain factors affect prognosis and treatment options [medquad_ans_0000028_2-1], no details on the actual outlook or chance of recovery are given. Therefore, based on the available information, the outlook for melanoma cannot be determined.

**Supported citations**: `medquad_ans_0000028_2-1`
**Unsupported citations**: _none_

### Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.720, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000132-2` (score=23.6754) - Boomerang dysplasia is a rare disorder; its exact prevalence is unknown. Approximately 10 affected individuals have been identified.
2. `medquad_ans_0000132-1` (score=22.9272) - Boomerang dysplasia is a disorder that affects the development of bones throughout the body. Affected individuals are born with inward- and upward-turning feet (clubfeet) and dislocations of the hips, knees, and elbow...
3. `medquad_ans_0000132-5` (score=20.1204) - These resources address the diagnosis or management of boomerang dysplasia: - Gene Review: Gene Review: FLNB-Related Disorders - Genetic Testing Registry: Boomerang dysplasia These resources from MedlinePlus offer inf...
4. `medquad_ans_0000132-3` (score=19.1079) - Mutations in the FLNB gene cause boomerang dysplasia. The FLNB gene provides instructions for making a protein called filamin B. This protein helps build the network of protein filaments (cytoskeleton) that gives stru...
5. `medquad_ans_0001853-3` (score=14.3676) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

The provided context does not explicitly state whether boomerang dysplasia is inherited. While it mentions that mutations in the FLNB gene cause the disorder and that filamin B plays a role in skeletal development, there is no information about inheritance patterns such as autosomal dominant, recessive, or X-linked inheritance. Therefore, based on the given context, the inheritance of boomerang dysplasia cannot be determined.

Answer: Insufficient information to determine if boomerang dysplasia is inherited. [medquad_ans_0000132-3]

**Supported citations**: `medquad_ans_0000132-3`
**Unsupported citations**: _none_

### Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.920

**Top docs**:

1. `medquad_ans_0000466-1` (score=39.4767) - Hereditary multiple osteochondromas is a condition in which people develop multiple benign (noncancerous) bone tumors called osteochondromas. The number of osteochondromas and the bones on which they are located vary ...
2. `medquad_ans_0002838-1` (score=38.4471) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called...
3. `medquad_ans_0000466-3` (score=34.6212) - Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and the EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. The two exosto...
4. `medquad_ans_0003520-3` (score=31.1801) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. Th...
5. `medquad_ans_0000466-5` (score=30.4406) - These resources address the diagnosis or management of hereditary multiple osteochondromas: - Gene Review: Gene Review: Hereditary Multiple Osteochondromas - Genetic Testing Registry: Multiple congenital exostosis - G...

**Answer**:

The genetic changes related to hereditary multiple osteochondromas are mutations in the EXT1 and EXT2 genes [medquad_ans_0002838-1].

**Supported citations**: `medquad_ans_0002838-1`
**Unsupported citations**: _none_

### Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.732, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000746-5` (score=17.6284) - These resources address the diagnosis or management of oculofaciocardiodental syndrome: - Genetic Testing Registry: Oculofaciocardiodental syndrome These resources from MedlinePlus offer information about the diagnosi...
2. `medquad_ans_0000746-1` (score=17.1493) - Oculofaciocardiodental (OFCD) syndrome is a condition that affects the development of the eyes (oculo-), facial features (facio-), heart (cardio-) and teeth (dental). This condition occurs only in females. The eye abn...
3. `medquad_ans_0004510-1` (score=14.4505) - Oculofaciocardiodental syndrome is a genetic syndrome that affects the eyes, heart, face, and teeth. Common signs and symptoms include abnormally small deep-set eyes, cataracts, long narrow face, a broad nasal tip tha...
4. `medquad_ans_0004510-2` (score=12.3666) - What are the signs and symptoms of Oculofaciocardiodental syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Oculofaciocardiodental syndrome. If the information is available, ...
5. `medquad_ans_0005488-1` (score=12.3337) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...

**Answer**:

The provided context does not specify the number of people affected by oculofaciocardiodental syndrome. [medquad_ans_0000746-5] discusses resources for diagnosis and management, [medquad_ans_0000746-1] and [medquad_ans_0004510-1] describe the signs and symptoms, and [medquad_ans_0004510-2] lists symptoms with frequency but does not provide population statistics. Therefore, the exact number of affected individuals is not available in the context.

**Supported citations**: `medquad_ans_0000746-5`, `medquad_ans_0000746-1`, `medquad_ans_0004510-1`, `medquad_ans_0004510-2`
**Unsupported citations**: _none_

### Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Scores**: context_relevance=1.000, groundedness=0.994, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001350-3` (score=18.0284) - What causes citrullinemia type I? Citrullinemia type I is caused by mutations in the ASS1 gene. This gene provides instructions for making an enzyme, argininosuccinate synthetase 1, that is responsible for the third s...
2. `medquad_ans_0001350-5` (score=17.9929) - What happens when citrullinemia type I is not treated? Untreated individuals with the severe form of citrullinemia type I have hyperammonemia (plasma ammonia concentration 1000-3000 mol/L). Without prompt intervention...
3. `medquad_ans_0000189-2` (score=17.5105) - Type I citrullinemia is the most common form of the disorder, affecting about 1 in 57,000 people worldwide. Type II citrullinemia is found primarily in the Japanese population, where it occurs in an estimated 1 in 100...
4. `medquad_ans_0001350-4` (score=17.5105) - How is citrullinemia type I inherited? Citrullinemia type I is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosom...
5. `medquad_ans_0000189-1` (score=17.2248) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused b...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused by mutations in different genes. Type I citrullinemia (also known as classic citrullinemia) usually becomes evident in the first few days of life. Affected infants typically appear normal at birth, but as ammonia builds up in the body they experience a progressive lack of energy (lethargy), poor feeding, vomiting, seizures, and loss of consciousness. These medical problems are life-threatening in many cases. Less commonly, a milder form of type I citrullinemia can develop later in childhood or adulthood [medquad_ans_0000189-1]. Type I citrullinemia is caused by mutations in the ASS1 gene, which provides instructions for making an enzyme,

**Supported citations**: `medquad_ans_0000189-1`
**Unsupported citations**: _none_

### Example 99 (`0000718-1__dup1`)

**Question**: What is (are) Bilateral frontal polymicrogyria ?

**Scores**: context_relevance=0.833, groundedness=0.991, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000718-1__dup1` (score=32.5362) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
2. `medquad_ans_0000722-5__dup1` (score=27.6401) - Is genetic testing available for bilateral perisylvian polymicrogyria? Genetic testing is not available for bilateral perisylvian polymicrogyria because the underlying genetic cause is unknown. How is bilateral perisy...
3. `medquad_ans_0000720-1__dup1` (score=24.8723) - Bilateral generalized polymicrogyria is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). This is the most widespread form of polymicrogyria and typically affects the enti...
4. `medquad_ans_0000722-4__dup1` (score=24.6125) - Is bilateral perisylvian polymicrogyria inherited? In most cases, bilateral perisylvian polymicrogyria (BPP) occurs sporadically in people with no family history of the condition. Rarely, more than one family member m...
5. `medquad_ans_0000719-1__dup1` (score=24.5984) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides ...

**Answer**:

Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area located behind the Sylvius fissure or the area located behind the Rolando sulcus. Signs and symptoms include delayed motor and language milestones; spastic (stiffness) hemiparesis (weakness in one side of the body) or quadriparesis (weakness in all four limbs of the body); and mild to moderate intellectual disability. Seizures may also be present. The cause for the frontal form of polymicrogyria is still not known [medquad_ans_0000718-1__dup1].

**Supported citations**: `medquad_ans_0000718-1__dup1`
**Unsupported citations**: _none_

### Example 100 (`0000944-5`)

**Question**: What are the treatments for spondylocostal dysostosis ?

**Scores**: context_relevance=0.714, groundedness=0.971, answer_relevance=0.486

**Top docs**:

1. `medquad_ans_0000944-3` (score=35.4023) - Mutations in at least four genes are known to cause spondylocostal dysostosis: Mutations in the DLL3 gene cause spondylocostal dysostosis type 1; mutations in the MESP2 gene cause spondylocostal dysostosis type 2; mut...
2. `medquad_ans_0005820-1` (score=33.8724) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
3. `medquad_ans_0005822-1` (score=33.8724) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
4. `medquad_ans_0005821-1` (score=33.8724) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
5. `medquad_ans_0005819-1` (score=33.8724) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...

**Answer**:

Treatment for spondylocostal dysostosis is symptomatic and supportive and may include respiratory support and surgery to correct inguinal hernia and scoliosis [medquad_ans_0005820-1].

**Supported citations**: `medquad_ans_0005820-1`
**Unsupported citations**: _none_

### Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.948, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000786-1` (score=29.4573) - Perry syndrome is a progressive brain disease that is characterized by four major features: a pattern of movement abnormalities known as parkinsonism, psychiatric changes, weight loss, and abnormally slow breathing (h...
2. `medquad_ans_0000786-5` (score=25.9851) - These resources address the diagnosis or management of Perry syndrome: - Gene Review: Gene Review: Perry Syndrome - Genetic Testing Registry: Perry syndrome - MedlinePlus Encyclopedia: Major Depression - MedlinePlus E...
3. `medquad_ans_0004825-1` (score=23.9391) - What are the signs and symptoms of Perry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Perry syndrome. If the information is available, the table below includes how often...
4. `medquad_ans_0000786-3` (score=23.7182) - Perry syndrome results from mutations in the DCTN1 gene. This gene provides instructions for making a protein called dynactin-1, which is involved in the transport of materials within cells. To move materials, dynacti...
5. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The symptoms of Perry syndrome include parkinsonism (characterized by bradykinesia, stiffness, and tremors), psychiatric changes such as depression, apathy, withdrawal from friends and family, and suicidal thoughts, significant unexplained weight loss, and abnormally slow breathing (hypoventilation) [medquad_ans_0000786-1]. Hypoventilation often occurs at night and can lead to respiratory failure. Other reported symptoms include sleep disturbance, dysarthria, and developmental regression [medquad_ans_0004825-1]. These signs and symptoms typically appear in a person's forties or fifties, with parkinsonism and psychiatric changes being the earliest features [medquad_ans_0000786-1].

**Supported citations**: `medquad_ans_0000786-1`, `medquad_ans_0004825-1`
**Unsupported citations**: _none_

### Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Scores**: context_relevance=0.714, groundedness=0.924, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0001028-2` (score=34.6219) - The prevalence of warfarin sensitivity is unknown. However, it appears to be more common in people who are older, those with lower body weights, and individuals of Asian ancestry. Of the approximately 2 million people...
2. `medquad_ans_0001028-1` (score=34.1122) - Warfarin sensitivity is a condition in which individuals have a low tolerance for the drug warfarin. Warfarin is an anticoagulant, which means that it thins the blood, preventing blood clots from forming. Warfarin is ...
3. `medquad_ans_0006398-1` (score=32.7236) - Warfarin sensitivity is a condition that is characterized by a reduced tolerance for a "blood-thinning" medication called warfarin. Warfarin is an anticoagulant that is often prescribed to people who are at an increas...
4. `medquad_ans_0001028-4` (score=30.3225) - The polymorphisms associated with this condition are inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to result in warfarin sensitivity. However, differen...
5. `medquad_ans_0001028-3` (score=28.2463) - Many genes are involved in the metabolism of warfarin and in determining the drug's effects in the body. Certain common changes (polymorphisms) in the CYP2C9 and VKORC1 genes account for 30 percent of the variation in...

**Answer**:

The most common symptom of warfarin sensitivity is excessive internal bleeding, which often occurs when individuals with warfarin sensitivity are given too much of the medication. This can manifest as abnormal bleeding in the brain, gastrointestinal tract, or other tissues [medquad_ans_0001028-2]. [medquad_ans_0006398-1]

**Supported citations**: `medquad_ans_0001028-2`, `medquad_ans_0006398-1`
**Unsupported citations**: _none_

### Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.315, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0004262-5` (score=21.7656) - How might Myhre syndrome be treated? Treatment of this condition is symptomatic and supportive. Children with Myhre syndrome may require management by a team of specialists, including pediatricians, speech pathologist...
2. `medquad_ans_0004262-1` (score=20.3509) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, a...
3. `medquad_ans_0004262-3` (score=19.7467) - What causes Myhre syndrome? Myhre syndrome is caused by mutations in the SMAD4 gene. This gene provides instructions for making a protein involved in transmitting chemical signals from the cell surface to the nucleus....
4. `medquad_ans_0000701-1` (score=19.0277) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most...
5. `medquad_ans_0000701-2` (score=17.8201) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.

**Answer**:

Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement of the muscles (muscle hypertrophy), and/or joint stiffness. Myhre syndrome is caused by mutations in the SMAD4 gene, which provides instructions for making a protein involved in transmitting chemical signals from the cell surface to the nucleus through the transforming growth factor beta (TGF-) pathway. These mutations impair the ability of the SMAD4 protein to bind properly with other proteins in the signaling pathway or result in an abnormally stable SMAD4 protein that remains active longer, leading to abnormal signaling and affecting development across multiple body systems [medquad_ans_0004262-1, medquad_ans_0004262-3]. The condition is inherited in an autosomal dominant pattern, with most cases resulting from

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0004262-1, medquad_ans_0004262-3`

### Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Scores**: context_relevance=0.857, groundedness=0.961, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0006057-4` (score=34.1353) - Is thanatophoric dysplasia inherited? Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell causes the condition. However, almost all cases of tha...
2. `medquad_ans_0006057-1` (score=33.2535) - Thanatophoric dysplasia is a severe skeletal disorder characterized by extremely short limbs and folds of extra skin on the arms and legs. Other features of this condition include a narrow chest, short ribs, underdeve...
3. `medquad_ans_0000976-1` (score=33.1496) - Thanatophoric dysplasia is a severe skeletal disorder characterized by extremely short limbs and folds of extra (redundant) skin on the arms and legs. Other features of this condition include a narrow chest, short rib...
4. `medquad_ans_0000976-4` (score=32.2029) - Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell is sufficient to cause the condition. Virtually all cases of thanatophoric dysplasia are cau...
5. `medquad_ans_0006057-3` (score=31.5585) - What causes thanatophoric dysplasia? Thanatophoric dysplasia is caused by mutations in the FGFR3 gene. This gene provides instructions for making a protein that is involved in the development and maintenance of bone a...

**Answer**:

The symptoms of thanatophoric dysplasia include extremely short limbs, folds of extra skin on the arms and legs, a narrow chest, short ribs, underdeveloped lungs, an enlarged head with a large forehead, and prominent, wide-spaced eyes. Two major forms are described: type I, which has curved thigh bones and flattened bones of the spine (platyspondyly), and type II, which has straight thigh bones and a moderate to severe skull abnormality called a cloverleaf skull [medquad_ans_0006057-1][medquad_ans_0000976-1][medquad_ans_0000976-4].

**Supported citations**: `medquad_ans_0006057-1`, `medquad_ans_0000976-1`, `medquad_ans_0000976-4`
**Unsupported citations**: _none_

### Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.901, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000381-3` (score=32.1562) - Mutations in the FMR1 gene cause fragile X syndrome. The FMR1 gene provides instructions for making a protein called FMRP. This protein helps regulate the production of other proteins and plays a role in the developme...
2. `medquad_ans_0002417-3` (score=31.4066) - What causes fragile X syndrome? Mutations (changes) in the FMR1 gene cause fragile X syndrome (FXS). This gene carries instructions to make a protein called the fragile X mental retardation 1 protein. The FMR1 gene co...
3. `medquad_ans_0000381-5` (score=29.0618) - These resources address the diagnosis or management of fragile X syndrome: - Gene Review: Gene Review: FMR1-Related Disorders - GeneFacts: Fragile X Syndrome: Diagnosis - GeneFacts: Fragile X Syndrome: Management - Ge...
4. `medquad_ans_0002417-2` (score=28.2177) - What are the signs and symptoms of Fragile X syndrome? Fragile X syndrome is characterized by developmental problems including intellectual disability and delayed speech and language development. Males are usually mor...
5. `medquad_ans_0000381-1` (score=27.9578) - Fragile X syndrome is a genetic condition that causes a range of developmental problems including learning disabilities and cognitive impairment. Usually, males are more severely affected by this disorder than females...

**Answer**:

The symptoms of Fragile X syndrome include disrupted nervous system functions due to the loss or deficiency of the FMRP protein, which is essential for regulating the production of other proteins and for the development of synapses. Synapses are critical for relaying nerve impulses, and their disruption leads to the signs and symptoms of fragile X syndrome. Specific manifestations may include intellectual disability, physical features such as prominent ears, emotional problems like anxiety or depression, learning disabilities, and autistic-like behavior [medquad_ans_0000381-3]. Additionally, individuals with a premutation may have mild versions of these physical features and emotional issues, though most are intellectually normal [medquad_ans_0002417-3].

**Supported citations**: `medquad_ans_0000381-3`, `medquad_ans_0002417-3`
**Unsupported citations**: _none_

### Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Scores**: context_relevance=0.750, groundedness=0.550, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000889-4__dup1` (score=18.2990) - How is CADASIL inherited? CADASIL is inherited in an autosomal dominant manner. This means that having a mutation in only one copy of the responsible gene in each cell is enough to cause CADASIL. In most cases, an aff...
2. `medquad_ans_0000056-1__dup1` (score=17.7570) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease that occurs when the thickening of blood vessel walls blocks the fl...
3. `medquad_ans_0000889-1__dup1` (score=17.1819) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited disease of the blood vessels that occurs when the thickening of blood vessel walls blocks the flow ...
4. `medquad_ans_0000889-5__dup1` (score=17.1529) - How might CADASIL be treated? There is currently no treatment for CADASIL that is proven to be effective. While antiplatelet treatment is often used, it is also not proven to be useful. Migraine should be treated both...
5. `medquad_ans_0000166-2` (score=17.1052) - CADASIL is likely a rare condition; however, its prevalence is unknown.

**Answer**:

CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease that occurs when the thickening of blood vessel walls blocks the flow of blood to the brain. The disease primarily affects small blood vessels in the white matter of the brain. A mutation in the Notch3 gene alters the muscular walls in these small arteries. CADASIL is characterized by migraine headaches and multiple strokes progressing to dementia. Other symptoms include cognitive deterioration, seizures, vision problems, and psychiatric problems such as severe depression and changes in behavior and personality. Individuals may also be at higher risk of heart attack. Symptoms and disease onset vary widely, with signs typically appearing in the mid-30s. Some individuals may not show signs of the disease until later in life. CADASIL is an autosomal dominant inheritance disorder, meaning that one parent carries and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Scores**: context_relevance=0.833, groundedness=0.550, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000688-2` (score=32.0715) - Multiple endocrine neoplasia type 1 affects about 1 in 30,000 people; multiple endocrine neoplasia type 2 affects an estimated 1 in 35,000 people. Among the subtypes of type 2, type 2A is the most common form, followe...
2. `medquad_ans_0000688-1` (score=32.0609) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate...
3. `medquad_ans_0004187-3` (score=31.1474) - How is multiple endocrine neoplasia type 2A inherited? Multiple endocrine neoplasia type 2A (MEN 2A) is inherited in an autosomal dominant pattern. A person with MEN 2A often inherits the altered RET gene from one par...
4. `medquad_ans_0000688-5` (score=29.9518) - These resources address the diagnosis or management of multiple endocrine neoplasia: - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 1 - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 2 - Ge...
5. `medquad_ans_0000688-3` (score=29.1020) - Mutations in the MEN1, RET, and CDKN1B genes can cause multiple endocrine neoplasia. Mutations in the MEN1 gene cause multiple endocrine neoplasia type 1. This gene provides instructions for producing a protein called...

**Answer**:

Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cells and tissues throughout the body. Multiple endocrine neoplasia typically involves tumors (neoplasia) in at least two endocrine glands; tumors can also develop in other organs and tissues. These growths can be noncancerous (benign) or cancerous (malignant). If the tumors become cancerous, the condition can be life-threatening. The major forms of multiple endocrine neoplasia are called type 1, type 2, and type 4. These types are distinguished by the genes involved, the types of hormones made, and the characteristic signs and symptoms. Many different types of tumors are associated with multiple endocrine neoplasia. Type 1 frequently involves tumors of the

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Scores**: context_relevance=1.000, groundedness=0.992, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000678-5` (score=22.2014) - These resources address the diagnosis or management of mucopolysaccharidosis type I: - Baby's First Test - Gene Review: Gene Review: Mucopolysaccharidosis Type I - Genetic Testing Registry: Mucopolysaccharidosis type ...
2. `medquad_ans_0004155-4` (score=18.3327) - How is mucopolysaccharidosis I (MPS I) inherited? MPS I is inherited from both parents in an autosomal recessive pattern.
3. `medquad_ans_0004155-5` (score=16.5212) - What treatment is available for mucopolysaccharidosis I (MPS I)? The two main treatments for MPS I are enzyme replacement therapy (ERT) and bone marrow transplant. Both of these treatments work by replacing the missin...
4. `medquad_ans_0004155-3` (score=16.0252) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycos...
5. `medquad_ans_0004155-1` (score=15.4727) - Mucopolysaccharidosis I (MPS I) is a condition that affects many parts of the body. It is a progressively debilitating disorder; however, the rate of progression varies among affected individuals. MPS I is caused by m...

**Answer**:

Mutations in the IDUA gene cause mucopolysaccharidosis type I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs). Mutations in the IDUA gene reduce or completely eliminate the function of the IDUA enzyme. The lack of IDUA enzyme activity leads to the accumulation of GAGs within cells, specifically inside the lysosomes. Lysosomes are compartments in the cell that digest and recycle different types of molecules. Conditions that cause molecules to build up inside the lysosomes, including MPS I, are called lysosomal storage disorders. The accumulation of GAGs increases the size of the lysosomes, which is why many tissues and organs are enlarged in this disorder [medquad_ans_0004155-3].

**Supported citations**: `medquad_ans_0004155-3`
**Unsupported citations**: _none_

### Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Scores**: context_relevance=0.900, groundedness=0.963, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000825-1` (score=39.8095) - Primary spontaneous pneumothorax is an abnormal accumulation of air in the space between the lungs and the chest cavity (called the pleural space) that can result in the partial or complete collapse of a lung. This ty...
2. `medquad_ans_0005065-1` (score=38.9713) - Primary spontaneous pneumothorax is an abnormal accumulation of air in the pleural space (the space between the lungs and the chest cavity) that can result in the partial or complete collapse of a lung. It is called p...
3. `medquad_ans_0000825-3` (score=33.0456) - Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition. The FLCN gene provides instructions for making a protein called follicu...
4. `medquad_ans_0000825-4` (score=31.8544) - When this condition is caused by mutations in the FLCN gene, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, ...
5. `medquad_ans_0005065-2` (score=30.8079) - What are the signs and symptoms of Primary spontaneous pneumothorax? The Human Phenotype Ontology provides the following list of signs and symptoms for Primary spontaneous pneumothorax. If the information is available...

**Answer**:

In rare cases, primary spontaneous pneumothorax can be caused by mutations in the FLCN gene. [medquad_ans_0005065-1]

**Supported citations**: `medquad_ans_0005065-1`
**Unsupported citations**: _none_

### Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Scores**: context_relevance=1.000, groundedness=0.938, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
2. `medquad_ans_0000261-3__dup2` (score=12.9963) - Although stroke is a disease of the brain, it can affect the entire body. A common disability that results from stroke is complete paralysis on one side of the body, called hemiplegia. A related disability that is not...
3. `medquad_ans_0000066-13` (score=12.3353) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blo...
4. `medquad_ans_0000066-1__dup4` (score=12.2871) - Stroke -- A Serious Event A stroke is serious, just like a heart attack. Each year in the United States, approximately 795,000 people have a stroke. About 610,000 of these are first or new strokes. On average, one Ame...
5. `medquad_ans_0000066-5__dup2` (score=12.2154) - Stroke is preventable and treatable. A better understanding of the causes of stroke has helped people make lifestyle changes that have cut the stroke death rate nearly in half in the last two decades. Preventing Strok...

**Answer**:

Stroke is a serious event, comparable to a heart attack, and occurs when blood flow to part of the brain is interrupted or reduced. There are two main types: ischemic stroke, which accounts for approximately 80 percent of all strokes and is caused by a blood clot blocking or plugging a blood vessel in the brain, and hemorrhagic stroke, which is caused by a blood vessel in the brain breaking and bleeding into the brain [medquad_ans_0000066-13]. Stroke is the fourth leading cause of death in the United States, with about 795,000 people experiencing a stroke each year, of which 610,000 are first or new strokes. On average, one American dies from stroke every four minutes. Nearly three-quarters of all strokes occur in people over the age of 65, and the risk of stroke more than doubles each decade between the ages of

**Supported citations**: `medquad_ans_0000066-13`
**Unsupported citations**: _none_

### Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006276-3` (score=25.3021) - What causes uncombable hair syndrome? The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been s...
2. `medquad_ans_0006276-5` (score=23.3447) - How is uncombable hair syndrome diagnosed? A diagnosis of uncombable hair syndrome (UHS) is made by observing the characteristic symptoms of the condition, as well observing the hair shaft under a special microscope. ...
3. `medquad_ans_0006276-4` (score=22.3744) - Is uncombable hair syndrome inherited? Uncombable hair syndrome (UHS) is thought to be inherited in an autosomal dominant manner with reduced penetrance. Autosomal dominant means that having a change (mutation) in onl...
4. `medquad_ans_0006276-6` (score=21.6874) - How might uncombable hair syndrome be treated? There is no definitive treatment for uncombable hair syndrome, but the condition usually improves or resolves on its own with the onset of puberty. Gentle hair care is ge...
5. `medquad_ans_0006276-2` (score=21.5340) - What are the signs and symptoms of Uncombable hair syndrome? Uncombable hair syndrome (UHS) may first become apparent any time between the ages of 3 months and 12 years. It only affects the scalp hair. The quantity of...

**Answer**:

The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. This abnormal shape is thought to result from premature keratinization (development of keratin) of the inner root sheath, which forms the channel for the growing hair. The inner root sheath conforms to the abnormal outline of the hair shaft, forming an irregular, rigid tube that alters the shape of the emerging hair. While the condition is assumed to be autosomal dominant and thus due to changes (mutations) in a gene, no responsible gene has been identified [medquad_ans_0006276-3].

**Supported citations**: `medquad_ans_0006276-3`
**Unsupported citations**: _none_

### Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.837, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000887-3` (score=21.0147) - The genetic causes of Russell-Silver syndrome are complex. The disorder often results from the abnormal regulation of certain genes that control growth. Research has focused on genes located in particular regions of c...
2. `medquad_ans_0005440-4` (score=20.5351) - Is Russell-Silver syndrome inherited? Most cases of Russell-Silver syndrome (RSS) are sporadic (not inherited), which means they occur in people with no family history of RSS. Less commonly, Russell-Silver syndrome is...
3. `medquad_ans_0000887-1` (score=20.3553) - Russell-Silver syndrome is a growth disorder characterized by slow growth before and after birth. Babies with this condition have a low birth weight and often fail to grow and gain weight at the expected rate (failure...
4. `medquad_ans_0000887-2` (score=18.9816) - The exact incidence of Russell-Silver syndrome is unknown, but the condition is estimated to affect 1 in 75,000 to 100,000 people.
5. `medquad_ans_0000887-5` (score=18.6038) - These resources address the diagnosis or management of Russell-Silver syndrome: - Gene Review: Gene Review: Russell-Silver Syndrome - Genetic Testing Registry: Russell-Silver syndrome - MedlinePlus Encyclopedia: Russe...

**Answer**:

Russell-Silver syndrome is not typically inherited in a classic Mendelian pattern. The genetic causes are complex and often result from abnormalities in gene regulation rather than direct inheritance of mutated genes. Most cases are due to epigenetic changes, such as abnormal methylation of genes like H19 and IGF2 on chromosome 11, or maternal uniparental disomy (UPD) of chromosome 7, which occurs sporadically rather than being passed down from parents. In at least 40 percent of cases, the cause is unknown, suggesting that many cases arise from de novo (new) genetic or epigenetic changes rather than inheritance [medquad_ans_0000887-3]. Therefore, Russell-Silver syndrome is not generally inherited.

**Supported citations**: `medquad_ans_0000887-3`
**Unsupported citations**: _none_

### Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Scores**: context_relevance=1.000, groundedness=0.973, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=19.6866) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0000094-6` (score=16.1873) - Treatments for obesity hypoventilation syndrome (OHS) include breathing support, weight loss, and medicines. The goals of treating OHS may include: Supporting and aiding your breathing Achieving major weight loss Trea...
3. `medquad_ans_0000043-2__dup4` (score=15.8954) - Many people have no visible signs or symptoms of diabetes. Symptoms can also be so mild that you might not notice them. More than 5 million people in the United States have type 2 diabetes and do not know it. - increa...
4. `medquad_ans_0000022-4__dup3` (score=15.8673) - The signs and symptoms of type 2 diabetes can be so mild that you might not even notice them. Nearly 7 million people in the United States have type 2 diabetes and dont know they have the disease. Many have no signs o...
5. `medquad_ans_0000095-7` (score=15.4371) - Successful weight-loss treatments include setting goals and making lifestyle changes, such as eating fewer calories and being physically active. Medicines and weight-loss surgery also are options for some people if li...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. It often limits the amount of food you can take in and some types affect how you digest and absorb nutrients. All types of surgery carry risks and complications, such as infections, hernias, and blood clots. Many people lose weight quickly, but some regain weight later; however, following diet and exercise recommendations can help keep most of the weight off. Ongoing medical follow-up is required for the rest of your life [medquad_ans_0000969-1__dup2].

**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

### Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.152, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000889-3` (score=27.2914) - Mutations in the TWIST1 gene cause Saethre-Chotzen syndrome. The TWIST1 gene provides instructions for making a protein that plays an important role in early development. This protein is a transcription factor, which ...
2. `medquad_ans_0000889-1` (score=24.9747) - Saethre-Chotzen syndrome is a genetic condition characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the ...
3. `medquad_ans_0000889-5` (score=22.3365) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Sae...
4. `medquad_ans_0003103-4` (score=21.4654) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...
5. `medquad_ans_0000108-2__dup1` (score=19.0184) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The provided context does not mention any treatments for Saethre-Chotzen syndrome. Therefore, based on the available information, it is not possible to determine the treatments for this condition. [medquad_ans_0000889-3, medquad_ans_0000889-1]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000889-3, medquad_ans_0000889-1`

### Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=0.833, groundedness=0.856, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000512-1` (score=29.5273) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effect...
2. `medquad_ans_0000512-2` (score=29.2026) - Idiopathic pulmonary fibrosis has an estimated prevalence of 13 to 20 per 100,000 people worldwide. About 100,000 people are affected in the United States, and 30,000 to 40,000 new cases are diagnosed each year. Famil...
3. `medquad_ans_0000512-3` (score=27.9543) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of devel...
4. `medquad_ans_0000512-4` (score=25.7661) - Most cases of idiopathic pulmonary fibrosis are sporadic; they occur in people with no history of the disorder in their family. Familial pulmonary fibrosis appears to have an autosomal dominant pattern of inheritance....
5. `medquad_ans_0000760-1__dup2` (score=25.2392) - Summary : Pulmonary rehabilitation (rehab) is a medically supervised program to help people who have chronic breathing problems, including - COPD (chronic obstructive pulmonary disease) - Sarcoidosis - Idiopathic pulm...

**Answer**:

Idiopathic pulmonary fibrosis is a chronic, progressive lung disease characterized by the buildup of scar tissue (fibrosis) in the lungs, which impairs the lungs' ability to transport oxygen into the bloodstream effectively [medquad_ans_0000512-1]. The disease typically affects individuals between the ages of 50 and 70 and is associated with common symptoms such as shortness of breath, a persistent dry, hacking cough, loss of appetite, and gradual weight loss. Some patients may also exhibit clubbing of the fingers and toes due to oxygen deficiency [medquad_ans_0000512-1]. The exact cause is unknown, but it is believed to result from a combination of genetic and environmental factors, with genetic changes potentially increasing risk and environmental exposures acting as triggers [medquad_ans_0000512-3]. The disease is generally sporadic, occurring in only one

**Supported citations**: `medquad_ans_0000512-1`, `medquad_ans_0000512-3`
**Unsupported citations**: _none_

### Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Scores**: context_relevance=0.917, groundedness=0.843, answer_relevance=0.783

**Top docs**:

1. `medquad_ans_0000213-3__dup1` (score=32.7257) - Early identification of and treatment for individuals with neuroleptic malignant syndrome improves outcome. If clinically indicated, a low potency neuroleptic can be reintroduced very slowly when the individual recove...
2. `medquad_ans_0004379-1` (score=29.7108) - Neuroleptic malignant syndrome is a rare neurological condition that is caused by an adverse reaction to neuroleptic (tranquilizer) or antipsychotic drugs. These drugs are commonly prescribed for the treatment of schi...
3. `medquad_ans_0000019_3-6` (score=27.3488) - Key Points - There are different types of treatment for patients with ovarian low malignant potential tumor. - Two types of standard treatment are used: - Surgery - Chemotherapy - New types of treatment are being test...
4. `medquad_ans_0000015_1-8` (score=26.4375) - Key Points - There are different types of treatment for children with extracranial germ cell tumors. - Children with extracranial germ cell tumors should have their treatment planned by a team of health care providers...
5. `medquad_ans_0000009_2-7` (score=26.4121) - Key Points - There are different types of treatment for patients with osteosarcoma or malignant fibrous histiocytoma (MFH) of bone. - Children with osteosarcoma or MFH should have their treatment planned by a team of ...

**Answer**:

The provided context does not mention any specific research or clinical trials being conducted for Neuroleptic Malignant Syndrome. [medquad_ans_0000213-3__dup1] and [medquad_ans_0004379-1] discuss the diagnosis, treatment, and risk factors of the condition but do not reference ongoing research or clinical trials. Therefore, based on the given information, no details about research or clinical trials for Neuroleptic Malignant Syndrome are available.

**Supported citations**: `medquad_ans_0000213-3__dup1`, `medquad_ans_0004379-1`
**Unsupported citations**: _none_

### Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000160-11` (score=27.1992) - The following changes in diet may help prevent UTIs and kidney stone formation: - Drinking plenty of water and other liquids can help flush bacteria from the urinary tract and dilute urine so kidney stones cannot form...
2. `medquad_ans_0000203-11` (score=27.0501) - - A kidney stone is a solid piece of material that forms in a kidney when substances that are normally found in the urine become highly concentrated. - Kidney stones are one of the most common disorders of the urinary...
3. `medquad_ans_0000016-1` (score=26.3551) - Adenine phosphoribosyltransferase (APRT) deficiency is an inherited condition that affects the kidneys and urinary tract. The most common feature of this condition is recurrent kidney stones; urinary tract stones are ...
4. `medquad_ans_0000221-8` (score=25.4679) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other he...
5. `medquad_ans_0000203-10` (score=24.7088) - The first step in preventing kidney stones is to understand what is causing the stones to form. The health care provider may ask the person to try to catch the kidney stone as it passes, so it can be sent to a lab for...

**Answer**:

To diagnose kidney stones in adults, a health care provider will perform a physical exam and take a medical history. The provider may also perform urine, blood, and imaging tests to complete the diagnosis [medquad_ans_0000203-11].

**Supported citations**: `medquad_ans_0000203-11`
**Unsupported citations**: _none_

### Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000041_1-1` (score=19.7315) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of ...
2. `medquad_ans_0000041_1-5` (score=19.7240) - Key Points - After urethral cancer has been diagnosed, tests are done to find out if cancer cells have spread within the urethra or to other parts of the body. - There are three ways that cancer spreads in the body. -...
3. `medquad_ans_0000041_1-6` (score=16.8243) - Key Points - There are different types of treatment for patients with urethral cancer. - Four types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Active surveillance - New types of tre...
4. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0000929-1__dup1` (score=13.2981) - The urethra is the tube that allows urine to pass out of the body. In men, it's a long tube that runs through the penis. It also carries semen in men. In women, it's short and is just above the vagina. Urethral proble...

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. The urethra is the tube that carries urine from the bladder to outside the body. In women, the urethra is about 1 inch long and is just above the vagina. In men, the urethra is about 8 inches long, and goes through the prostate gland and the penis to the outside of the body. In men, the urethra also carries semen. Urethral cancer is a rare cancer that occurs more often in men than in women. There are different types of urethral cancer that begin in cells that line the urethra, named for the types of cells that become malignant: squamous cell carcinoma, transitional cell carcinoma, and adenocarcinoma. Squamous cell carcinoma is the most common type and forms in cells near the bladder in women and in the lining

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Scores**: context_relevance=1.000, groundedness=0.900, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004725-1` (score=30.3014) - Paramyotonia congenita is an inherited condition that affects muscles used for movement (skeletal muscles), mainly in the face, neck, arms, and hands. Symptoms begin in infancy or early childhood and include episodes ...
2. `medquad_ans_0000771-1` (score=27.2701) - Paramyotonia congenita is a disorder that affects muscles used for movement (skeletal muscles). Beginning in infancy or early childhood, people with this condition experience bouts of sustained muscle tensing (myotoni...
3. `medquad_ans_0000771-2` (score=26.8566) - Paramyotonia congenita is an uncommon disorder; it is estimated to affect fewer than 1 in 100,000 people.
4. `medquad_ans_0000771-5` (score=25.5669) - These resources address the diagnosis or management of paramyotonia congenita: - Genetic Testing Registry: Paramyotonia congenita of von Eulenburg - Periodic Paralysis International: How is Periodic Paralysis Diagnose...
5. `medquad_ans_0000378-1__dup1` (score=16.0394) - Anonychia congenita is an extremely rare nail disorder characterized by the complete absence (anonychia) or abnormally developed fingernails and toenails. Affected individuals usually do not have hair, teeth, or bone ...

**Answer**:

Yes, paramyotonia congenita is inherited in an autosomal dominant pattern [medquad_ans_0004725-1].

**Supported citations**: `medquad_ans_0004725-1`
**Unsupported citations**: _none_

### Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.974, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000125-3` (score=31.8755) - Mutations in the BTD gene cause biotinidase deficiency. The BTD gene provides instructions for making an enzyme called biotinidase. This enzyme recycles biotin, a B vitamin found in foods such as liver, egg yolks, and...
2. `medquad_ans_0000125-5` (score=28.4220) - These resources address the diagnosis or management of biotinidase deficiency: - Baby's First Test - Gene Review: Gene Review: Biotinidase Deficiency - Genetic Testing Registry: Biotinidase deficiency - MedlinePlus En...
3. `medquad_ans_0000735-2__dup1` (score=28.2125) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound bio...
4. `medquad_ans_0000735-1__dup1` (score=27.9788) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. The disorder may become apparent in the first few months of life, or later in childhood. The more severe form ...
5. `medquad_ans_0000125-1` (score=26.9611) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. If this condition is not recognized and treated, its signs and symptoms typically appear within the first few ...

**Answer**:

The signs and symptoms of biotinidase deficiency typically appear within the first few months of life and may include seizures, weak muscle tone (hypotonia), breathing problems, delayed development, hearing loss, eye abnormalities and loss of vision, problems with movement and balance (ataxia), skin rashes, hair loss (alopecia), and a fungal infection called candidiasis. In partial biotinidase deficiency, which is a milder form, affected children may experience hypotonia, skin rashes, and hair loss, but these problems may appear only during illness, infection, or other times of stress on the body [medquad_ans_0000125-3][medquad_ans_0000735-2__dup1].

**Supported citations**: `medquad_ans_0000125-3`, `medquad_ans_0000735-2__dup1`
**Unsupported citations**: _none_

### Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Scores**: context_relevance=0.857, groundedness=0.350, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000019-1__dup3` (score=16.6234) - Summary : Air pollution is a mixture of solid particles and gases in the air. Car emissions, chemicals from factories, dust, pollen and mold spores may be suspended as particles. Ozone, a gas, is a major part of air p...
2. `medquad_ans_0000460-1__dup1` (score=15.1018) - Your liver is the largest organ inside your body. It helps your body digest food, store energy, and remove poisons. Hepatitis is an inflammation of the liver. One type, hepatitis A, is caused by the hepatitis A virus ...
3. `medquad_ans_0000179-6` (score=14.8227) - Peritoneal dialysis uses the lining of your belly to filter your blood inside your body. You can do peritoneal dialysis at home because it uses your body to filter. A doctor will place a soft tube called a catheter in...
4. `medquad_ans_0000021-6` (score=14.7754) - Your doctor usually will diagnose bronchitis based on your signs and symptoms. He or she may ask questions about your cough, such as how long you've had it, what you're coughing up, and how much you cough. Your doctor...
5. `medquad_ans_0000066-1__dup3` (score=14.4864) - Asthma is a chronic disease that affects your airways. Your airways are tubes that carry air in and out of your lungs. If you have asthma, the inside walls of your airways become sore and swollen. In the United States...

**Answer**:

No, there is no information about water pollution in the provided context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Scores**: context_relevance=1.000, groundedness=0.971, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000132-3__dup1` (score=30.7563) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is...
2. `medquad_ans_0000132-1__dup1` (score=28.1970) - Foot drop describes the inability to raise the front part of the foot due to weakness or paralysis of the muscles that lift the foot. As a result, individuals with foot drop scuff their toes along the ground or bend t...
3. `medquad_ans_0005812-1` (score=26.3660) - Split hand split foot nystagmus is a rare congenital syndrome characterized by split hand and split foot deformity and eye abnormalities, especially nystagmus. It is thought to have an autosomal dominant mode of inher...
4. `medquad_ans_0001088-1__dup1` (score=24.9365) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
5. `medquad_ans_0006496-1` (score=23.1503) - X-linked Charcot-Marie-Tooth disease type 5 (CMTX5) is a neurological condition characterized by peripheral neuropathy, early-onset bilateral profound sensorineural hearing loss, and optic neuropathy leading to visual...

**Answer**:

The outlook for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1]. Foot drop can be temporary or permanent, depending on the underlying cause, which may include neurodegenerative disorders, motor neuron disorders, nerve root injury, peripheral nerve disorders, or muscle disorders [medquad_ans_0000132-1__dup1].

**Supported citations**: `medquad_ans_0000132-3__dup1`, `medquad_ans_0000132-1__dup1`
**Unsupported citations**: _none_

### Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Scores**: context_relevance=1.000, groundedness=0.981, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000916-5__dup1` (score=20.7116) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene...
2. `medquad_ans_0000916-3__dup1` (score=19.4267) - What causes Camurati-Engelmann disease? Mutations in the TGFB1 gene cause Camurati-Engelmann disease. The TGFB1 gene provides instructions for producing a protein called transforming growth factor beta-1 (TGF-1). The ...
3. `medquad_ans_0000916-1__dup1` (score=19.0005) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
4. `medquad_ans_0000915-1__dup1` (score=19.0005) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
5. `medquad_ans_0000145-3` (score=18.2021) - Mutations in the TGFB1 gene cause Camurati-Engelmann disease. The TGFB1 gene provides instructions for producing a protein called transforming growth factor beta-1 (TGF-1). The TGF-1 protein helps control the growth a...

**Answer**:

Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are also affected. The thickened bones can lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. The age at which affected individuals first experience symptoms varies greatly; however, most people with this condition develop pain or weakness by adolescence. The disease is caused by mutations in the TGFB1 gene, which results in the production of a transforming growth factor beta-1 (TGF-1) protein that is always active, leading to increased bone density and decreased body fat and muscle tissue [medquad_ans_0000916-3__dup1]. Diagnosis is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing,

**Supported citations**: `medquad_ans_0000916-3__dup1`
**Unsupported citations**: _none_

### Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Scores**: context_relevance=1.000, groundedness=0.941, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000316-4` (score=38.9779) - Emery-Dreifuss muscular dystrophy can have several different patterns of inheritance. When this condition is caused by mutations in the EMD gene, it is inherited in an X-linked recessive pattern. A condition is consid...
2. `medquad_ans_0002080-2` (score=37.8223) - What are the signs and symptoms of Emery-Dreifuss muscular dystrophy, X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Emery-Dreifuss muscular dystrophy, X-linked. If the in...
3. `medquad_ans_0000316-5` (score=35.4433) - These resources address the diagnosis or management of Emery-Dreifuss muscular dystrophy: - Gene Review: Gene Review: Emery-Dreifuss Muscular Dystrophy - Genetic Testing Registry: Emery-Dreifuss muscular dystrophy - G...
4. `medquad_ans_0000316-3` (score=34.7645) - Mutations in the EMD and LMNA genes cause Emery-Dreifuss muscular dystrophy. The EMD and LMNA genes provide instructions for making proteins that are components of the nuclear envelope, which surrounds the nucleus in ...
5. `medquad_ans_0000316-1` (score=32.8376) - Emery-Dreifuss muscular dystrophy is a condition that chiefly affects muscles used for movement (skeletal muscles) and heart (cardiac) muscle. Among the earliest features of this disorder are joint deformities called ...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked include abnormality of the neck, Achilles tendon contracture, atrioventricular block, childhood onset, and decreased cervical spine flexion due to contractures of posterior cervicals [medquad_ans_0002080-2].

**Supported citations**: `medquad_ans_0002080-2`
**Unsupported citations**: _none_

### Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.746, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000072-3` (score=32.3537) - Arterial tortuosity syndrome is caused by mutations in the SLC2A10 gene. This gene provides instructions for making a protein called GLUT10. The level of GLUT10 appears to be involved in the regulation of a process ca...
2. `medquad_ans_0000072-1` (score=31.9017) - Arterial tortuosity syndrome is a disorder that affects connective tissue. Connective tissue provides strength and flexibility to structures throughout the body, including blood vessels, skin, joints, and the gastroin...
3. `medquad_ans_0000072-5` (score=30.7670) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins Mc...
4. `medquad_ans_0000072-2` (score=26.7053) - Arterial tortuosity syndrome is a rare disorder; its prevalence is unknown. About 100 cases have been reported in the medical literature.
5. `medquad_ans_0003659-1` (score=24.3787) - What are the signs and symptoms of Loeys-Dietz syndrome type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Loeys-Dietz syndrome type 4. If the information is available, the tabl...

**Answer**:

The provided context does not mention any treatments for arterial tortuosity syndrome. Therefore, based on the available information, it is not possible to determine the treatments for this condition. [medquad_ans_0000072-3] [medquad_ans_0000072-1]

**Supported citations**: `medquad_ans_0000072-3`, `medquad_ans_0000072-1`
**Unsupported citations**: _none_

### Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Scores**: context_relevance=0.875, groundedness=0.862, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0005142-1` (score=28.0417) - Pseudohypoparathyroidism type 1C is a genetic disorder that is very similar to hypoparathyroidism (parathyroid hormone levels are too low). However, pseudohypoparathyroidism is caused by no response to parathyroid hor...
2. `medquad_ans_0000173-1` (score=27.6040) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
3. `medquad_ans_0000173-5` (score=27.2539) - These resources address the diagnosis or management of Charcot-Marie-Tooth disease: - Gene Review: Gene Review: Charcot-Marie-Tooth Hereditary Neuropathy Overview - Gene Review: Gene Review: Charcot-Marie-Tooth Neurop...
4. `medquad_ans_0001088-1__dup1` (score=26.6913) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
5. `medquad_ans_0001104-5` (score=25.8793) - Is genetic testing available for Charcot-Marie-Tooth disease type 2F? Yes. GeneTests lists the names of laboratories that are performing clincial genetic testing for Charcot-Marie-Tooth disease type 2F. To view the co...

**Answer**:

The provided context does not contain information about the symptoms of Charcot-Marie-Tooth disease type 1C. While it describes general symptoms of Charcot-Marie-Tooth disease, such as balance difficulties, clumsiness, muscle weakness in the feet, foot abnormalities (e.g., high arches, flat feet, curled toes), difficulty walking, and decreased sensitivity to touch, heat, and cold, it does not specify symptoms related to type 1C. Therefore, based on the given context, the symptoms of Charcot-Marie-Tooth disease type 1C cannot be determined. [medquad_ans_0000173-1]

**Supported citations**: `medquad_ans_0000173-1`
**Unsupported citations**: _none_

### Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.908, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000870-5` (score=70.3494) - These resources address the diagnosis or management of RAMSVPS: - Calgary Retina Consultants: Retinal Arterial Macroaneurysm - Genetic Testing Registry: Retinal arterial macroaneurysm with supravalvular pulmonic steno...
2. `medquad_ans_0005321-1` (score=66.3030) - What are the signs and symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal arterial macroaneurysm...
3. `medquad_ans_0000870-1` (score=62.0273) - Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) is a disorder that affects blood vessels in the eyes and heart. The condition generally becomes apparent in infancy or childhood. RAMSVPS d...
4. `medquad_ans_0000962-1` (score=31.1499) - Supravalvular aortic stenosis (SVAS) is a heart defect that develops before birth. This defect is a narrowing (stenosis) of the large blood vessel that carries blood from the heart to the rest of the body (the aorta)....
5. `medquad_ans_0001036-1` (score=30.5342) - Williams syndrome is a developmental disorder that affects many parts of the body. This condition is characterized by mild to moderate intellectual disability or learning problems, unique personality characteristics, ...

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) include exudative retinal detachment and pulmonic stenosis [medquad_ans_0005321-1]. These signs and symptoms are part of the clinical features associated with the condition, as documented in the Human Phenotype Ontology [medquad_ans_0005321-1].

**Supported citations**: `medquad_ans_0005321-1`
**Unsupported citations**: _none_

### Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Scores**: context_relevance=1.000, groundedness=0.959, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001838-1` (score=31.4759) - Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. Symptoms have varied greatly among the few reported cases. Dicarboxylic aminoaciduria is...
2. `medquad_ans_0001838-2` (score=22.9980) - What are the signs and symptoms of Dicarboxylic aminoaciduria? There are no common signs or symptoms of dicarboxylic aminoaciduria. Hypoglycemia, developmental and neurological abnormalities, and obsessive compulsive ...
3. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
4. `medquad_ans_0002659-2` (score=12.8509) - What are the signs and symptoms of GRACILE syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for GRACILE syndrome. If the information is available, the table below includes how o...
5. `medquad_ans_0000108-2__dup1` (score=11.5977) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. It is caused by mutations in the SLC1A1 gene and inherited in an autosomal recessive fashion. There are no common signs or symptoms; however, in individual cases, hypoglycemia, developmental and neurological abnormalities, and obsessive compulsive tendencies have been reported. Some individuals have been diagnosed with virtually no signs or symptoms [medquad_ans_0001838-1][medquad_ans_0001838-2].

**Supported citations**: `medquad_ans_0001838-1`, `medquad_ans_0001838-2`
**Unsupported citations**: _none_

### Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Scores**: context_relevance=1.000, groundedness=0.956, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0005860-3` (score=33.4218) - Is spondylothoracic dysostosis genetic? Yes. Spondylothoracic dysostosis is caused by mutations in the MESP2 gene. It is inherited in an autosomal recessive fashion.
2. `medquad_ans_0005860-1` (score=27.7832) - Spondylothoracic dysostosis is a rare condition that affects the bones of the spine and the ribs. Signs and symptoms are generally present at birth and may include short-trunk dwarfism (a short body with normal length...
3. `medquad_ans_0000949-1` (score=25.8177) - Spondylothoracic dysostosis is a condition characterized by the malformation of the bones of the spine and ribs. The bones of the spine (vertebrae) do not develop properly, which causes them to be misshapen and abnorm...
4. `medquad_ans_0005860-4` (score=25.0647) - What treatment is available for spondylothoracic dysostosis? Many infants born with spondylothoracic dysostosis have difficulty breathing due to their small, malformed chests, and therefore are prone to repeated respi...
5. `medquad_ans_0000949-2` (score=24.7454) - Spondylothoracic dysostosis affects about one in 200,000 people worldwide. However, it is much more common in people of Puerto Rican ancestry, affecting approximately one in 12,000 people.

**Answer**:

Yes, spondylothoracic dysostosis is inherited in an autosomal recessive manner. This is due to mutations in the MESP2 gene [medquad_ans_0005860-3]. The condition is caused by changes (mutations) in the MESP2 gene and is inherited in an autosomal recessive fashion [medquad_ans_0005860-1].

**Supported citations**: `medquad_ans_0005860-3`, `medquad_ans_0005860-1`
**Unsupported citations**: _none_

### Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.490, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0003296-1` (score=36.0163) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. I...
2. `medquad_ans_0000542-5` (score=33.1477) - These resources address the diagnosis or management of Jervell and Lange-Nielsen syndrome: - Gene Review: Gene Review: Jervell and Lange-Nielsen Syndrome - Genetic Testing Registry: Jervell and Lange-Nielsen syndrome ...
3. `medquad_ans_0000542-3` (score=31.4399) - Mutations in the KCNE1 and KCNQ1 genes cause Jervell and Lange-Nielsen syndrome. The KCNE1 and KCNQ1 genes provide instructions for making proteins that work together to form a channel across cell membranes. These cha...
4. `medquad_ans_0000542-2` (score=29.7946) - Jervell and Lange-Nielsen syndrome is uncommon; it affects an estimated 1.6 to 6 per 1 million people worldwide. This condition has a higher prevalence in Denmark, where it affects at least 1 in 200,000 people.
5. `medquad_ans_0000542-1` (score=27.8666) - Jervell and Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). This disorder is a form of long QT syndrome, which is a heart ...

**Answer**:

Jervell Lange-Nielsen syndrome is a form of long QT syndrome that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). It is inherited in an autosomal recessive fashion and is caused by mutations in the KCNE1 and KCNQ1 genes, which lead to altered function or assembly of potassium channels. These changes disrupt potassium ion flow in the inner ear and cardiac muscle, resulting in hearing loss and irregular heartbeats. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. About 90 percent of cases are due to mutations in the KCNQ1 gene, with the remaining cases caused by KCNE1 mutations. The condition is uncommon, affecting an estimated 1.6 to 6 per 1 million people worldwide, with a higher prevalence in Denmark [medquad_ans_0003296-1, medquad_ans

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Scores**: context_relevance=1.000, groundedness=0.788, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000218-5` (score=51.4548) - These resources address the diagnosis or management of CCFDN: - Gene Review: Gene Review: Congenital Cataracts, Facial Dysmorphism, and Neuropathy - Genetic Testing Registry: Congenital Cataracts, Facial Dysmorphism, ...
2. `medquad_ans_0000218-1` (score=30.4083) - Congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is a rare disorder that affects several parts of the body. It is characterized by a clouding of the lens of the eyes at birth (congenital cataracts) and...
3. `medquad_ans_0003571-1` (score=21.9750) - What are the signs and symptoms of Leukoencephalopathy - dystonia - motor neuropathy ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoencephalopathy - dystonia - motor neuropat...
4. `medquad_ans_0000475-1` (score=19.8445) - Hereditary sensory neuropathy type IA is a condition characterized by nerve abnormalities in the legs and feet (peripheral neuropathy). Many people with this condition experience prickling or tingling sensations (pare...
5. `medquad_ans_0000395-1` (score=19.6507) - Fukuyama congenital muscular dystrophy is an inherited condition that predominantly affects the muscles, brain, and eyes. Congenital muscular dystrophies are a group of genetic conditions that cause muscle weakness an...

**Answer**:

The provided context does not specify the number of people affected by congenital cataracts, facial dysmorphism, and neuropathy (CCFDN). [medquad_ans_0000218-1] describes the condition's features and complications but does not include population statistics or prevalence data. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000218-1`
**Unsupported citations**: _none_

### Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Scores**: context_relevance=0.769, groundedness=0.960, answer_relevance=0.892

**Top docs**:

1. `medquad_ans_0000040-1` (score=48.4411) - Alpha thalassemia X-linked intellectual disability syndrome is an inherited disorder that affects many parts of the body. This condition occurs almost exclusively in males. Males with alpha thalassemia X-linked intell...
2. `medquad_ans_0000040-3` (score=44.4172) - Alpha thalassemia X-linked intellectual disability syndrome results from mutations in the ATRX gene. This gene provides instructions for making a protein that plays an essential role in normal development. Although th...
3. `medquad_ans_0000040-5` (score=42.0282) - These resources address the diagnosis or management of alpha thalassemia X-linked intellectual disability syndrome: - Gene Review: Gene Review: Alpha-Thalassemia X-Linked Intellectual Disability Syndrome - Genetic Tes...
4. `medquad_ans_0000040-2` (score=39.7164) - Alpha thalassemia X-linked intellectual disability syndrome appears to be a rare condition, although its exact prevalence is unknown. More than 200 affected individuals have been reported.
5. `medquad_ans_0000040-4` (score=36.9721) - This condition is inherited in an X-linked recessive pattern. The ATRX gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of the g...

**Answer**:

The genetic changes related to alpha thalassemia X-linked intellectual disability syndrome are mutations in the ATRX gene. These mutations change the structure of the ATRX protein, which likely prevents it from effectively regulating the expression of other genes, including HBA1 and HBA2, which are necessary for normal hemoglobin production [medquad_ans_0000040-3].

**Supported citations**: `medquad_ans_0000040-3`
**Unsupported citations**: _none_

### Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002622-1` (score=36.0599) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepa...
2. `medquad_ans_0002620-3` (score=34.9505) - What causes glycogen storage disease type 4? Glycogen storage disease type 4 (GSD 4) is caused by mutations in the GBE1 gene. The GBE1 gene normally provides instructions for making the glycogen branching enzyme. This...
3. `medquad_ans_0002615-2` (score=33.9245) - What are the signs and symptoms of Glycogen storage disease type 13? Glycogen storage disease type 13 causes muscle pain (myalgia). Individuals with GSD13 also experience exercise intolerance, which means they have di...
4. `medquad_ans_0000425-5` (score=33.4160) - These resources address the diagnosis or management of glycogen storage disease type IX: - Gene Review: Gene Review: Phosphorylase Kinase Deficiency - Genetic Testing Registry: Glycogen storage disease IXb - Genetic T...
5. `medquad_ans_0000423-5` (score=32.7924) - These resources address the diagnosis or management of glycogen storage disease type III: - Gene Review: Gene Review: Glycogen Storage Disease Type III - Genetic Testing Registry: Glycogen storage disease type III The...

**Answer**:

The symptoms of Glycogen storage disease type 6 include low blood sugar (hypoglycemia), an enlarged liver (hepatomegaly), and an increase in the amount of lactic acid in the blood (lactic acidosis), particularly when an individual does not eat for a long time. Symptoms improve significantly as individuals with this condition get older [medquad_ans_0002622-1].

**Supported citations**: `medquad_ans_0002622-1`
**Unsupported citations**: _none_

### Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Scores**: context_relevance=1.000, groundedness=0.546, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0001853-3` (score=23.9952) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
2. `medquad_ans_0004342-3` (score=17.3930) - What causes nephrogenic diabetes insipidus? Nephrogenic diabetes insipidus can be either acquired or hereditary. The acquired form can result from chronic kidney disease, certain medications (such as lithium), low lev...
3. `medquad_ans_0004434-2` (score=16.2732) - What causes nonalcoholic steatohepatitis? The underlying cause of NASH remains unclear. It most often occurs in persons who are middle-aged and overweight or obese. Many patients with NASH have elevated blood lipids, ...
4. `medquad_ans_0006157-3` (score=15.8830) - What causes transposition of the great arteries (TGA)? The exact cause of TGA remains unknown. Some possible associated risk factors that have been proposed include gestational diabetes mellitus, maternal exposure to ...
5. `medquad_ans_0000126-3__dup2` (score=15.6860) - Doctors often don't know what causes tetralogy of Fallot and other congenital heart defects. Some conditions or factors that occur during pregnancy may raise your risk of having a child who has tetralogy of Fallot. Th...

**Answer**:

The provided context does not contain information about the causes of diabetes in general. While it mentions diabetes mellitus as a condition associated with diffuse idiopathic skeletal hyperostosis [medquad_ans_0001853-3] and as a risk factor for nonalcoholic steatohepatitis [medquad_ans_0004434-2], it does not specify the underlying causes of diabetes itself. Therefore, based on the given context, the causes of diabetes cannot be determined. [insufficient context]

**Supported citations**: `medquad_ans_0001853-3`, `medquad_ans_0004434-2`
**Unsupported citations**: `insufficient context`

### Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Scores**: context_relevance=1.000, groundedness=0.803, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000016-6` (score=30.0325) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
2. `medquad_ans_0000016-1__dup3` (score=29.8097) - Can Cause Vision Loss, Blindness Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina in the back of the eye. A ...
3. `medquad_ans_0000016-2__dup2` (score=27.4062) - Who Is at Risk? All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes ...
4. `medquad_ans_0000016-4__dup2` (score=27.0104) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...
5. `medquad_ans_0000007_2-6` (score=26.0332) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any specific research or clinical trials being conducted for diabetic retinopathy. [medquad_ans_0000016-6] and [medquad_ans_0000016-1__dup3] describe the condition, its stages, and associated risks but do not reference ongoing research or clinical trials. Therefore, based on the given information, no such research is cited.

**Supported citations**: `medquad_ans_0000016-6`, `medquad_ans_0000016-1__dup3`
**Unsupported citations**: _none_

### Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Scores**: context_relevance=1.000, groundedness=0.607, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=22.6762) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart an...
2. `medquad_ans_0000854-3__dup1` (score=16.2446) - What causes bronchiolitis obliterans organizing pneumonia (BOOP)? BOOP may be caused by a variety of factors, including viral infections, inhalation of toxic gases, drugs, connective tissue disorders, radiation therap...
3. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
4. `medquad_ans_0000853-1__dup1` (score=13.6809) - Bronchiolitis obliterans is an inflammatory obstruction of the lung's tiniest airways, the bronchioles. The bronchioles may become damaged and inflamed after inhalation of toxic fumes, as a result of respiratory infec...
5. `medquad_ans_0000077-1__dup4` (score=13.6209) - Your back is made of bones, muscles, and other tissues extending from your neck to your pelvis. Back injuries can result from sports injuries, work around the house or in the garden, or a sudden jolt such as a car acc...

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles in the air from fires or toxic fumes. These substances can damage the eyes and respiratory system and may worsen chronic heart and lung diseases. Symptoms may include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, it may worsen [medquad_ans_0000516-1__dup1]. The best prevention is to limit exposure, especially when smoke is detected or fires are nearby [Environmental Protection Agency].

**Supported citations**: `medquad_ans_0000516-1__dup1`
**Unsupported citations**: `Environmental Protection Agency`

### Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.725, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000134-10` (score=22.0736) - - Zollinger-Ellison syndrome is a rare disorder that occurs when one or more tumors form in the pancreas and duodenum. - Experts do not know the exact cause of Zollinger-Ellison syndrome. - About 25 to 30 percent of g...
2. `medquad_ans_0000134-8` (score=17.7605) - A health care provider treats Zollinger-Ellison syndrome with medications to reduce gastric acid secretion and with surgery to remove gastrinomas. A health care provider sometimes uses chemotherapymedications to shrin...
3. `medquad_ans_0000134-1__dup3` (score=17.2375) - Zollinger-Ellison syndrome is a rare disorder that occurs when one or more tumors form in the pancreas and duodenum. The tumors, called gastrinomas, release large amounts of gastrin that cause the stomach to produce l...
4. `medquad_ans_0000134-4__dup2` (score=16.8241) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has ...
5. `medquad_ans_0000134-5__dup1` (score=16.1665) - Zollinger-Ellison syndrome signs and symptoms are similar to those of peptic ulcers. A dull or burning pain felt anywhere between the navel and midchest is the most common symptom of a peptic ulcer. This discomfort us...

**Answer**:

The provided context does not specify the number of people affected by Zollinger-Ellison syndrome. [medquad_ans_0000134-10] and [medquad_ans_0000134-8] discuss the causes, symptoms, diagnosis, and treatment of the condition but do not include population statistics or prevalence rates. Therefore, the exact number of people affected cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000134-10`, `medquad_ans_0000134-8`
**Unsupported citations**: _none_

### Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Scores**: context_relevance=0.714, groundedness=0.662, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000974-3` (score=29.2763) - Tetrahydrobiopterin deficiency can be caused by mutations in one of several genes, including GCH1, PCBD1, PTS, and QDPR. These genes provide instructions for making enzymes that help produce and recycle tetrahydrobiop...
2. `medquad_ans_0000974-2` (score=28.8745) - This condition is rare, affecting an estimated 1 in 500,000 to 1 in 1 million newborns. In most parts of the world, tetrahydrobiopterin deficiency accounts for 1 to 3 percent of all cases of elevated phenylalanine lev...
3. `medquad_ans_0000898-3` (score=27.4640) - Mutations in the SPR gene cause sepiapterin reductase deficiency. The SPR gene provides instructions for making the sepiapterin reductase enzyme. This enzyme is involved in the production of a molecule called tetrahyd...
4. `medquad_ans_0000974-1` (score=25.6991) - Tetrahydrobiopterin deficiency is a rare disorder characterized by a shortage (deficiency) of a molecule called tetrahydrobiopterin or BH4. This condition alters the levels of several substances in the body, including...
5. `medquad_ans_0001860-1` (score=24.3661) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...

**Answer**:

The provided context does not mention any treatments for tetrahydrobiopterin deficiency. Therefore, based on the given information, it is not possible to answer the question about treatments. [medquad_ans_0000974-3][medquad_ans_0000974-2][medquad_ans_0000898-3]

**Supported citations**: `medquad_ans_0000974-3`, `medquad_ans_0000974-2`, `medquad_ans_0000898-3`
**Unsupported citations**: _none_

### Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Scores**: context_relevance=0.875, groundedness=0.756, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000736-1` (score=29.4809) - Nonsyndromic holoprosencephaly is an abnormality of brain development that also affects the head and face. Normally, the brain divides into two halves (hemispheres) during early development. Holoprosencephaly occurs w...
2. `medquad_ans_0000736-3` (score=27.2529) - Mutations in 11 genes have been found to cause nonsyndromic holoprosencephaly. These genes provide instructions for making proteins that are important for normal embryonic development, particularly for determining the...
3. `medquad_ans_0000736-5` (score=22.0998) - These resources address the diagnosis or management of nonsyndromic holoprosencephaly: - Gene Review: Gene Review: Holoprosencephaly Overview - Genetic Testing Registry: Holoprosencephaly 1 - Genetic Testing Registry:...
4. `medquad_ans_0000736-2` (score=20.7061) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
5. `medquad_ans_0000736-4` (score=18.9589) - Nonsyndromic holoprosencephaly is inherited in an autosomal dominant pattern, which means an alteration in one copy of a gene in each cell is usually sufficient to cause the disorder. However, not all people with a ge...

**Answer**:

The provided context does not specify the number of people affected by nonsyndromic holoprosencephaly. [medquad_ans_0000736-1] describes the condition, its features, types, and variability but does not include epidemiological data or prevalence statistics. Therefore, the number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000736-1`
**Unsupported citations**: _none_

### Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Scores**: context_relevance=1.000, groundedness=0.857, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000829-5` (score=45.4915) - These resources address the diagnosis or management of progressive familial intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Progressive familial intrahepatic cholest...
2. `medquad_ans_0000118-1` (score=40.8068) - Benign recurrent intrahepatic cholestasis (BRIC) is characterized by episodes of liver dysfunction called cholestasis. During these episodes, the liver cells have a reduced ability to release a digestive fluid called ...
3. `medquad_ans_0005084-2` (score=39.3676) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis...
4. `medquad_ans_0000525-3` (score=36.9018) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt ex...
5. `medquad_ans_0000829-1` (score=35.5813) - Progressive familial intrahepatic cholestasis (PFIC) is a disorder that causes progressive liver disease, which typically leads to liver failure. In people with PFIC, liver cells are less able to secrete a digestive f...

**Answer**:

The provided context does not specify the exact genetic changes related to progressive familial intrahepatic cholestasis (PFIC). It mentions genetic testing resources for PFIC types 2 and 3 and references a gene review for ATP8B1 deficiency, which is associated with a different condition (likely PFIC type 1 or related to cholestasis), but no direct information on the genetic changes for PFIC is given. Therefore, based on the available context, the genetic changes related to PFIC cannot be determined.

[medquad_ans_0000829-5]

**Supported citations**: `medquad_ans_0000829-5`
**Unsupported citations**: _none_

### Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005678-1` (score=35.8905) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below in...
2. `medquad_ans_0003934-1` (score=21.7093) - What are the signs and symptoms of Mental retardation Smith Fineman Myers type? The Human Phenotype Ontology provides the following list of signs and symptoms for Mental retardation Smith Fineman Myers type. If the in...
3. `medquad_ans_0002946-1` (score=21.5434) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...
4. `medquad_ans_0002062-1` (score=21.2726) - What are the signs and symptoms of Ehlers-Danlos syndrome, spondylocheirodysplastic type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, spondylocheirodyspl...
5. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

### Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Scores**: context_relevance=1.000, groundedness=0.766, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000453-1` (score=37.6477) - Hepatic veno-occlusive disease with immunodeficiency (also called VODI) is a hereditary disorder of the liver and immune system. Its signs and symptoms appear after the first few months of life. Hepatic veno-occlusive...
2. `medquad_ans_0000453-5` (score=35.4004) - These resources address the diagnosis or management of VODI: - Gene Review: Gene Review: Hepatic Veno-Occlusive Disease with Immunodeficiency - Genetic Testing Registry: Hepatic venoocclusive disease with immunodefici...
3. `medquad_ans_0001517-2` (score=20.2058) - What causes congenital hepatic fibrosis? Isolated congenital hepatic fibrosis is rare. Congenital hepatic fibrosis is usually associated with conditions known as hepatorenal fibrocystic diseases (FCD) that can also af...
4. `medquad_ans_0000047-1__dup1` (score=19.6417) - Achondroplasia with severe combined immunodeficiency is an extremely rare type of SCID. The condition is characterized by the classic signs of SCID, including severe and recurrent infections, diarrhea, failure to thri...
5. `medquad_ans_0002804-1` (score=19.2300) - What are the signs and symptoms of Hepatic venoocclusive disease with immunodeficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Hepatic venoocclusive disease with immunodefic...

**Answer**:

Yes, hepatic veno-occlusive disease with immunodeficiency (VODI) is inherited. This is supported by the context, which states that VODI is a "hereditary disorder of the liver and immune system" [medquad_ans_0000453-1]. Additionally, while the specific inheritance pattern is not explicitly detailed in the provided text, the mention of it being a hereditary disorder implies inheritance. [medquad_ans_0000453-1]

**Supported citations**: `medquad_ans_0000453-1`
**Unsupported citations**: _none_

### Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005902-1` (score=29.1410) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how o...
2. `medquad_ans_0000057-22` (score=23.3275) - You can help your doctor make a diagnosis by writing down important information about your problem beforehand and giving the information to your doctor during your visit. Write down answers to the following questions....
3. `medquad_ans_0000036-19` (score=23.0617) - If you are diagnosed with high blood pressure, here are questions to ask your doctor. - Your Blood Pressure Numbers - What is my blood pressure reading in numbers? - What is my goal blood pressure? - Is my blood press...
4. `medquad_ans_0004259-3` (score=22.2493) - What causes MYH7-related scapuloperoneal myopathy? MYH7-related scapuloperoneal myopathy is caused by mutations in the MYH7 gene. This gene, located on chromosome 14q12, provides instructions for making a protein know...
5. `medquad_ans_0000038-22` (score=21.6416) - During your next health care visit, talk to your provider about your test results and how to manage your kidney disease. Below is a list of questions you may want to ask. Add any questions you think are missing, and m...

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. These signs and symptoms are based on the Human Phenotype Ontology, which compiles information from sources such as Orphanet [medquad_ans_0005902-1].

**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

### Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Scores**: context_relevance=0.667, groundedness=0.876, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0003688-1` (score=29.3415) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, h...
2. `medquad_ans_0000159-6` (score=29.2929) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
3. `medquad_ans_0000159-8` (score=28.5238) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed...
4. `medquad_ans_0000159-7` (score=25.5842) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may n...
5. `medquad_ans_0000159-1__dup3` (score=24.3480) - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). SLE is an autoimmune diseasea disorder in which the bodys immune system attacks the bodys own cells and organs. Up to 60 pe...

**Answer**:

Treatment for lupus nephritis involves medications to suppress the immune system to prevent damage to the kidneys. Standard treatment typically includes a corticosteroid, such as prednisone, to reduce inflammation, and an immunosuppressive medication like cyclophosphamide or mycophenolate mofetil, used in combination with prednisone. People with high blood pressure due to lupus nephritis may also need medications to lower blood pressure, which can slow the progression of kidney disease. In some cases, treatment may result in complete or partial control of the condition, with few or no further complications. However, 10 to 30 percent of individuals may still develop kidney failure, requiring dialysis or a kidney transplant. Early diagnosis and treatment are crucial to prevent chronic kidney damage [medquad_ans_0003688-1][medquad_ans_0000159-6][medquad

**Supported citations**: `medquad_ans_0003688-1`, `medquad_ans_0000159-6`
**Unsupported citations**: _none_

### Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.117, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000869-3` (score=33.7192) - Restless legs syndrome likely results from a combination of genetic and environmental factors, many of which are unknown. Studies suggest that restless legs syndrome is related to a shortage (deficiency) of iron in ce...
2. `medquad_ans_0000869-5` (score=32.3235) - These resources address the diagnosis or management of restless legs syndrome: - Agency for Healthcare Research and Quality: Options for Treating Restless Legs Syndrome - Genetic Testing Registry: Restless legs syndro...
3. `medquad_ans_0000869-1` (score=31.7300) - Restless legs syndrome is a neurological condition that causes an irresistible urge to move the legs. The movement is triggered by strange or uncomfortable feelings, often described as crawling, pulling, or itching, d...
4. `medquad_ans_0000824-1__dup2` (score=31.1188) - Is it hard for you to fall asleep or stay asleep through the night? Do you wake up feeling tired or feel very sleepy during the day, even if you have had enough sleep? You might have a sleep disorder. The most common ...
5. `medquad_ans_0005317-1` (score=30.5182) - Restless legs syndrome is a neurological condition that causes an irresistible urge to move the legs. The movement is triggered by strange or uncomfortable feelings, which occur mostly while the affected person is sit...

**Answer**:

The provided context does not include information about the treatments for Restless Legs Syndrome. Therefore, based on the given text, it is not possible to answer the question. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Scores**: context_relevance=1.000, groundedness=0.936, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002573-5` (score=17.3276) - Is genetic testing available for glioblastoma? Genetic testing is not available for many people with glioblastoma since most of these tumors occur sporadically (by chance) and are not caused by a genetic mutation. How...
2. `medquad_ans_0002573-6` (score=16.6799) - How might glioblastoma be treated? Unfortunately, there is no cure for glioblastoma. Treatment is palliative and may include surgery, radiation therapy and/or chemotherapy. The best treatment options for each person d...
3. `medquad_ans_0002573-3` (score=15.8709) - What causes glioblastoma? In most cases, the exact underlying cause of glioblastoma is unknown. In rare cases, they can occur in people with certain genetic syndromes such as neurofibromatosis type 1, Turcot syndrome ...
4. `medquad_ans_0002575-1` (score=15.0258) - Gliomatosis cerebri is a type of brain cancer. It is a variant form of glioblastoma multiforme. It is characterized by scattered and widespread tumor cells that can cause the cerebrum, cerebellum, or brain stem to enl...
5. `medquad_ans_0002573-4` (score=14.2906) - Is glioblastoma inherited? Most glioblastomas are not inherited. They usually occur sporadically in people with no family history of tumors. However, they can rarely occur in people with certain genetic syndromes such...

**Answer**:

Glioblastoma is a type of brain tumor that is typically diagnosed based on a physical exam, characteristic symptoms, and imaging studies such as computed tomography (CT) or magnetic resonance imaging (MRI) [medquad_ans_0002573-5]. It is often caused by unknown factors, with most cases occurring sporadically and not due to genetic mutations. In rare cases, glioblastoma can occur in individuals with certain genetic syndromes, such as neurofibromatosis type 1, Turcot syndrome, and Li Fraumeni syndrome [medquad_ans_0002573-3]. Unfortunately, there is no cure for glioblastoma, and treatment is palliative, often involving surgery, radiation therapy, and/or chemotherapy, depending on tumor size, location, and the patient's age and health [medquad_ans_0002573-6].

**Supported citations**: `medquad_ans_0002573-5`, `medquad_ans_0002573-3`, `medquad_ans_0002573-6`
**Unsupported citations**: _none_

### Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0005496-3` (score=19.1276) - What causes schizencephaly? The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, s...
2. `medquad_ans_0005496-4` (score=15.1365) - Is schizencephaly inherited? Schizencephaly is not thought to be inherited in most cases and it rarely affects more than one person in a family. A few cases of familial schizencephaly have been linked to changes (muta...
3. `medquad_ans_0005496-5` (score=13.9164) - Is genetic testing available for schizencephaly? In rare cases, people affected by schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Genetic testing is available f...
4. `medquad_ans_0001853-3` (score=13.3144) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0005496-6` (score=12.6425) - How might schizencephaly be treated? The best treatment options for people with schizencephaly depend on many factors, including the severity of the condition and the signs and symptoms present. For example, people wi...

**Answer**:

The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, schizencephaly can affect more than one family member, which supports a genetic cause in some cases. Schizencephaly has also been linked to a variety of non-genetic factors, including young maternal age and certain medications and infections that can cause vascular disruptions (disruption of blood flow or blood supply) in a developing baby [medquad_ans_0005496-3].

**Supported citations**: `medquad_ans_0005496-3`
**Unsupported citations**: _none_

### Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Scores**: context_relevance=0.800, groundedness=0.865, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000052-5` (score=41.9346) - These resources address the diagnosis or management of amyotrophic lateral sclerosis: - Gene Review: Gene Review: ALS2-Related Disorders - Gene Review: Gene Review: Amyotrophic Lateral Sclerosis Overview - Gene Review...
2. `medquad_ans_0000316-1__dup1` (score=37.4747) - What are the signs and symptoms of Amyotrophic lateral sclerosis type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyotrophic lateral sclerosis type 6. If the information is a...
3. `medquad_ans_0000309-3__dup1` (score=37.2963) - What causes amyotrophic lateral sclerosis? In approximately 90-95% of cases the cause of amyotrophic lateral sclerosis (ALS) is unknown and is sporadic (occurring in individuals with no history of the condition in the...
4. `medquad_ans_0003319-3` (score=36.3929) - What causes juvenile amyotrophic lateral sclerosis? Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases the underlying gene abnormality...
5. `medquad_ans_0000309-2__dup1` (score=34.3382) - What are the signs and symptoms of Amyotrophic lateral sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyotrophic lateral sclerosis. If the information is available, the ...

**Answer**:

The genetic changes related to amyotrophic lateral sclerosis (ALS) include mutations in specific genes such as C9orf72, TARDBP, and ALS2. These are detailed in gene reviews covering C9orf72-related ALS and frontotemporal dementia, TARDBP-related ALS, and ALS2-related disorders [medquad_ans_0000052-5]. Approximately 5-10% of ALS cases are genetic (familial), with the remaining 90-95% being sporadic with an unknown cause [medquad_ans_0000309-3__dup1]. Genetic testing is available for ALS and ALS type 1, supporting the identification of these genetic alterations [medquad_ans_0000052-5].

**Supported citations**: `medquad_ans_0000052-5`, `medquad_ans_0000309-3__dup1`
**Unsupported citations**: _none_

### Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.725, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0003473-5` (score=26.6719) - How is Laron syndrome diagnosed? A diagnosis of Laron syndrome is often suspected based on the presence of characteristic signs and symptoms. Additional testing can then be ordered to confirm the diagnosis and rule ou...
2. `medquad_ans_0003473-2` (score=24.4239) - What are the signs and symptoms of Laron syndrome? Laron syndrome is a rare condition in which the body is unable to use growth hormone. The primary symptom is short stature. Although affected people are generally clo...
3. `medquad_ans_0000581-1` (score=24.0241) - Laron syndrome is a rare form of short stature that results from the body's inability to use growth hormone, a substance produced by the brain's pituitary gland that helps promote growth. Affected individuals are clos...
4. `medquad_ans_0000581-3` (score=21.6728) - Laron syndrome is caused by mutations in the GHR gene. This gene provides instructions for making a protein called the growth hormone receptor. The receptor is present on the outer membrane of cells throughout the bod...
5. `medquad_ans_0003473-4` (score=21.5917) - Is Laron syndrome inherited? Most cases of Laron syndrome are inherited in an autosomal recessive manner. This means that to be affected, a person must have a mutation in both copies of the responsible gene in each ce...

**Answer**:

The provided context does not mention any treatments for Laron syndrome. Therefore, based on the available information, it is not possible to answer the question. [medquad_ans_0003473-5][medquad_ans_0003473-2]

**Supported citations**: `medquad_ans_0003473-5`, `medquad_ans_0003473-2`
**Unsupported citations**: _none_

### Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Scores**: context_relevance=0.857, groundedness=0.745, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000016-1__dup3` (score=34.5899) - Can Cause Vision Loss, Blindness Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina in the back of the eye. A ...
2. `medquad_ans_0000016-6` (score=30.8949) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
3. `medquad_ans_0000016-2__dup2` (score=30.7022) - Who Is at Risk? All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes ...
4. `medquad_ans_0000016-7` (score=29.9516) - The four stages of diabetic retinopathy are - mild nonproliferative retinopathy - moderate nonproliferative retinopathy - severe nonproliferative retinopathy - proliferative retinopathy mild nonproliferative retinopat...
5. `medquad_ans_0000016-24` (score=28.7312) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even ...

**Answer**:

The provided context does not mention any treatments for diabetic retinopathy. It describes the condition, its stages, and associated eye diseases but does not include information on treatment options [medquad_ans_0000016-1\_\_dup3][medquad_ans_0000016-6].

**Supported citations**: `medquad_ans_0000016-1__dup3`, `medquad_ans_0000016-6`
**Unsupported citations**: _none_

### Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Scores**: context_relevance=1.000, groundedness=0.745, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000116-1__dup4` (score=36.4734) - Microscopic colitis is an inflammation of the colon that a health care provider can see only with a microscope. Inflammation is the bodys normal response to injury, irritation, or infection of tissues. Microscopic col...
2. `medquad_ans_0001421-2` (score=31.8680) - What are the signs and symptoms of collagenous colitis? All individuals with collagenous colitis experience chronic, watery, non-bloody diarrhea which is what typically prompts individuals to seek medical attention. O...
3. `medquad_ans_0000122-1__dup3` (score=30.4877) - Ulcerative colitis is a chronic, or long lasting, disease that causes inflammationirritation or swellingand sores called ulcers on the inner lining of the large intestine. Ulcerative colitis is a chronic inflammatory ...
4. `medquad_ans_0000093-1__dup4` (score=29.6877) - Crohn's disease is a chronic, or long lasting, disease that causes inflammationirritation or swellingin the gastrointestinal (GI) tract. Most commonly, Crohn's affects the small intestine and the beginning of the larg...
5. `medquad_ans_0000116-11` (score=29.6284) - - Microscopic colitis is an inflammation of the colon that a health care provider can see only with a microscope. - The two types of microscopic colitis are collagenous colitis and lymphocytic colitis. - The exact cau...

**Answer**:

The provided context does not contain information about the treatments for collagenous colitis. [medquad_ans_0001421-2] describes the signs and symptoms of collagenous colitis but does not mention any treatments. Therefore, based on the given context, the treatments for collagenous colitis cannot be determined.

**Supported citations**: `medquad_ans_0001421-2`
**Unsupported citations**: _none_

### Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Scores**: context_relevance=0.667, groundedness=0.813, answer_relevance=0.489

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=19.6941) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
2. `medquad_ans_0000808-3` (score=19.1319) - Mutations in the GAA gene cause Pompe disease. The GAA gene provides instructions for producing an enzyme called acid alpha-glucosidase (also known as acid maltase). This enzyme is active in lysosomes, which are struc...
3. `medquad_ans_0000194-3` (score=18.7894) - Cockayne syndrome can result from mutations in either the ERCC6 gene (also known as the CSB gene) or the ERCC8 gene (also known as the CSA gene). These genes provide instructions for making proteins that are involved ...
4. `medquad_ans_0000418-7` (score=18.4935) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause ...
5. `medquad_ans_0000168-3` (score=18.0190) - Mutations in at least three genes, KRIT1 (also known as CCM1), CCM2, and PDCD10 (also known as CCM3), cause familial cerebral cavernous malformations. The precise functions of these genes are not fully understood. Stu...

**Answer**:

To prevent trichinellosis, one should avoid consuming raw or undercooked meat, particularly pork and wild game meat. Proper cooking of meat to safe internal temperatures kills the Trichinella parasite. Additionally, commercial and home freezing of pork can reduce the risk of infection. Public awareness of the dangers of eating raw or undercooked meat products also plays a role in reducing cases. Eating undercooked wild game, especially bear meat, remains a risk factor [medquad_ans_0000418-2__dup1].

**Supported citations**: `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

### Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000536-3` (score=22.7086) - What causes autoimmune hepatitis? Although the exact cause of autoimmune hepatitis is unknown, evidence suggests that liver injury in a patient with autoimmune hepatitis is the result of a cell-mediated immunologic at...
2. `medquad_ans_0000182-11` (score=20.7984) - - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the liver and causes inflammation and damage. - Autoimmune hepatitis is a serious condition that may worsen over time...
3. `medquad_ans_0000182-5__dup1` (score=19.8032) - Autoimmune hepatitis is classified into several types. Type 1 autoimmune hepatitis is the most common form in North America. Type 1 can occur at any age; however, it most often starts in adolescence or young adulthood...
4. `medquad_ans_0000535-3` (score=19.1616) - What causes autoimmune hemolytic anemia? In about half of cases, the cause of autoimmune hemolytic anemia cannot be determined (idiopathic or primary). This condition can also be caused by or occur with another disord...
5. `medquad_ans_0000182-1__dup3` (score=18.7556) - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the normal components, or cells, of the liver and causes inflammation and liver damage. The immune system normally prot...

**Answer**:

The exact cause of autoimmune hepatitis is unknown, but evidence suggests that liver injury is the result of a cell-mediated immunologic attack. This autoimmune attack may be triggered by genetic factors, viral infections, or chemical agents. Autoimmune hepatitis sometimes occurs in relatives of people with autoimmune diseases, further suggesting a genetic cause [medquad_ans_0000536-3].

**Supported citations**: `medquad_ans_0000536-3`
**Unsupported citations**: _none_

### Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000006_3-7` (score=22.6224) - Key Points - The grade of the tumor is used to plan cancer treatment. - Low-grade astrocytomas - High-grade astrocytomas - An MRI is done after surgery. The grade of the tumor is used to plan cancer treatment. Staging...
2. `medquad_ans_0000006_3-1` (score=19.3242) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
3. `medquad_ans_0000324-2__dup1` (score=18.2593) - Are anaplastic astrocytomas inherited? Anaplastic astrocytomas are usually not inherited. These tumors typically occur sporadically, in people with no family history of astrocytomas. In most cases, the exact cause is ...
4. `medquad_ans_0004883-3` (score=16.0374) - What causes pilocytic astrocytoma? The exact underlying cause of pilocytic astrocytomas is currently unknown. Although most are thought to be sporadic (occurring by chance in an affected individual), they are known to...
5. `medquad_ans_0000006_3-9` (score=14.6372) - Key Points - There are different types of treatment for patients with childhood astrocytoma. - Children with astrocytomas should have their treatment planned by a team of health care providers who are experts in treat...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain [medquad_ans_0000006_3-1].

**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

### Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000013_3-1__dup3` (score=39.7455) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia in...
2. `medquad_ans_0000013_3-4__dup2` (score=31.0344) - Treatment of atypical chronic myelogenous leukemia (CML) may include chemotherapy. Check the list of NCI-supported cancer clinical trials that are now accepting patients with atypical chronic myeloid leukemia, BCR-ABL...
3. `medquad_ans_0004252-2` (score=30.3811) - What causes myelodysplastic/myeloproliferative disease? In most cases, the cause of myelodysplastic/myeloproliferative disease is unknown, and there is limited information regarding potential causes. No specific genet...
4. `medquad_ans_0000001_4-6` (score=30.1006) - Key Points - After chronic myelogenous leukemia has been diagnosed, tests are done to find out if the cancer has spread. - Chronic myelogenous leukemia has 3 phases. - Chronic phase - Accelerated phase - Blastic phase...
5. `medquad_ans_0000013_2-1__dup3` (score=29.7297) - Chronic myelogenous leukemia is a disease in which too many white blood cells are made in the bone marrow. See the PDQ summary on Chronic Myelogenous Leukemia Treatment for information on diagnosis, staging, and treat...

**Answer**:

Atypical chronic myelogenous leukemia (aCML) is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. The body tells too many blood stem cells to become granulocytes, and some of these stem cells never become mature white blood cells; these immature cells are called blasts. Over time, the granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. Leukemia cells in atypical CML and CML look alike under a microscope, but in atypical CML, the "Philadelphia chromosome" is not present [medquad_ans_0000013_3-1__dup3]. Additionally, aCML has been associated with cytogenetic (chromosomal) abnormalities in as many as 80% of individuals, although no specific cytogenetic abnormality is unique to the disease [medquad*ans*

**Supported citations**: `medquad_ans_0000013_3-1__dup3`
**Unsupported citations**: _none_

### Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Scores**: context_relevance=1.000, groundedness=0.928, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000996-3` (score=36.4383) - Most cases of trisomy 13 result from having three copies of chromosome 13 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the cha...
2. `medquad_ans_0000996-4` (score=34.0055) - Most cases of trisomy 13 are not inherited and result from random events during the formation of eggs and sperm in healthy parents. An error in cell division called nondisjunction results in a reproductive cell with a...
3. `medquad_ans_0000996-5` (score=29.3497) - These resources address the diagnosis or management of trisomy 13: - Genetic Testing Registry: Complete trisomy 13 syndrome - MedlinePlus Encyclopedia: Trisomy 13 These resources from MedlinePlus offer information abo...
4. `medquad_ans_0000996-1` (score=28.1403) - Trisomy 13, also called Patau syndrome, is a chromosomal condition associated with severe intellectual disability and physical abnormalities in many parts of the body. Individuals with trisomy 13 often have heart defe...
5. `medquad_ans_0000667-3` (score=27.2370) - The causes of Moebius syndrome are unknown, although the condition probably results from a combination of environmental and genetic factors. Researchers are working to identify and describe specific genes related to t...

**Answer**:

The genetic changes related to trisomy 13 include having three copies of chromosome 13 in each cell instead of the usual two copies, which disrupts normal development and causes characteristic features [medquad_ans_0000996-3]. This can occur due to nondisjunction during the formation of reproductive cells (eggs and sperm), resulting in a reproductive cell with an extra copy of chromosome 13 [medquad_ans_0000996-4]. In some cases, trisomy 13 results from a translocation, where a part of chromosome 13 becomes attached to another chromosome during the formation of reproductive cells or early fetal development [medquad_ans_0000996-3]. In rare instances, only part of chromosome 13 is present in three copies, leading to potentially different physical signs and symptoms [medquad_ans_00009

**Supported citations**: `medquad_ans_0000996-3`, `medquad_ans_0000996-4`
**Unsupported citations**: _none_

### Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Scores**: context_relevance=0.714, groundedness=0.033, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000731-2` (score=26.3020) - Niemann-Pick disease types A and B is estimated to affect 1 in 250,000 individuals. Niemann-Pick disease type A occurs more frequently among individuals of Ashkenazi (eastern and central European) Jewish descent than ...
2. `medquad_ans_0000731-3` (score=25.3566) - Niemann-Pick disease types A and B is caused by mutations in the SMPD1 gene. This gene provides instructions for producing an enzyme called acid sphingomyelinase. This enzyme is found in lysosomes, which are compartme...
3. `medquad_ans_0004415-1` (score=25.2169) - Niemann-Pick disease is an inherited condition involving lipid metabolism, which is the breakdown, transport, and use of fats and cholesterol in the body. In people with this condition, abnormal lipid metabolism cause...
4. `medquad_ans_0000218-4__dup1` (score=24.4091) - The National Institute of Neurological Disorders and Stroke (NINDS), a part of the National Institutes of Health (NIH), conducts and supports research about Niemann-Pick disease through research grants to research ins...
5. `medquad_ans_0000731-1` (score=23.8124) - Niemann-Pick disease is a condition that affects many body systems. It has a wide range of symptoms that vary in severity. Niemann-Pick disease is divided into four main types: type A, type B, type C1, and type C2. Th...

**Answer**:

The provided context does not mention any treatments for Niemann-Pick disease. Therefore, based on the available information, it is not possible to answer the question about treatments. [medquad_ans_0000731-2, medquad_ans_0000731-3, medquad_ans_0004415-1]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000731-2, medquad_ans_0000731-3, medquad_ans_0004415-1`

### Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001714-1` (score=43.6462) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial poly...
2. `medquad_ans_0000270-4` (score=36.9709) - Most cases of Dandy-Walker malformation are sporadic, which means they occur in people with no history of the disorder in their family. A small percentage of cases seem to run in families; however, Dandy-Walker malfor...
3. `medquad_ans_0003858-1` (score=36.5964) - What are the signs and symptoms of Meckel syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Meckel syndrome type 3. If the information is available, the table below in...
4. `medquad_ans_0000270-3` (score=36.4404) - Researchers have found mutations in a few genes that are thought to cause Dandy-Walker malformation, but these mutations account for only a small number of cases. Dandy-Walker malformation has also been associated wit...
5. `medquad_ans_0003857-1` (score=36.1655) - What are the signs and symptoms of Meckel syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Meckel syndrome type 2. If the information is available, the table below in...

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, autosomal recessive inheritance, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, and wide nasal bridge [

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003789-4` (score=29.0980) - How is Mantle cell lymphoma diagnosed? Mantle cell lymphoma is diagnosed by a biopsy (surgical removal) of the lymph nodes. If lymph nodes are not easily accessible to be biopsied, a fine needle aspiration may be perf...
2. `medquad_ans_0003789-1` (score=26.5144) - Mantle cell lymphoma (MCL) belongs to a group of diseases known as non-Hodgkins lymphomas (NHL), which are cancers that affect the the lymphatic system (part of the immune system). MCL accounts for 6% of all non-Hodgk...
3. `medquad_ans_0003789-3` (score=21.2257) - What causes Mantle cell lymphoma? Most lymphomas are not inherited, but rather an acquired disease in which the DNAwithin the cells has been damaged. Damage to the DNA occurs by a combination of different factors. Man...
4. `medquad_ans_0003789-2` (score=20.0406) - What are the signs and symptoms of Mantle cell lymphoma? Common symptoms of Mantle cell lymphoma include fatigue, loss of appetite, and enlarged lymph nodes, spleen, and/or liver. Other symptoms may include night swea...
5. `medquad_ans_0003789-5` (score=19.4789) - How might Mantle cell lympoma be treated? Various treatmentsare currently available for Mantle cell lymphomas. Rare cases of early stage mantle cell lymphomas may be treated with radiation therapy. For more advance st...

**Answer**:

Most lymphomas are not inherited but are acquired diseases in which the DNA within the cells has been damaged. Many mantle cell lymphomas are found to be associated with a chromosome translocation. Causes of non-Hodgkin lymphomas (NHL) have also been linked to viral infections including Epstein-Barr virus, HIV, and human herpesvirus 6. Immunodeficiencies and environmental factors such as hair dyes and pesticides may lead to NHLs [medquad_ans_0003789-3].

**Supported citations**: `medquad_ans_0003789-3`
**Unsupported citations**: _none_

### Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.884, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000589-3` (score=20.7864) - Legg-Calv-Perthes disease is usually not caused by genetic factors. The cause in these cases is unknown. In a small percentage of cases, mutations in the COL2A1 gene cause the bone abnormalities characteristic of Legg...
2. `medquad_ans_0000589-1` (score=19.9209) - Legg-Calv-Perthes disease is a bone disorder that affects the hips. Usually, only one hip is involved, but in about 10 percent of cases, both hips are affected. Legg-Calv-Perthes disease begins in childhood, typically...
3. `medquad_ans_0000589-2` (score=17.9418) - The incidence of Legg-Calv-Perthes disease varies by population. The condition is most common in white populations, in which it affects an estimated 1 to 3 in 20,000 children under age 15.
4. `medquad_ans_0001853-3` (score=14.3676) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0000589-5` (score=12.4353) - These resources address the diagnosis or management of Legg-Calv-Perthes disease: - National Osteonecrosis Foundation - Seattle Children's Hospital These resources from MedlinePlus offer information about the diagnosi...

**Answer**:

Legg-Calv-Perthes disease is usually not caused by genetic factors, and the cause in most cases is unknown. In a small percentage of cases, mutations in the COL2A1 gene may be involved, which suggests a possible genetic component in those specific instances. However, overall, the condition is not typically inherited [medquad_ans_0000589-3].

**Supported citations**: `medquad_ans_0000589-3`
**Unsupported citations**: _none_

### Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Scores**: context_relevance=0.800, groundedness=0.521, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000647-3` (score=26.3947) - Most individuals with metachromatic leukodystrophy have mutations in the ARSA gene, which provides instructions for making the enzyme arylsulfatase A. This enzyme is located in cellular structures called lysosomes, wh...
2. `medquad_ans_0000647-1` (score=25.4167) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the sub...
3. `medquad_ans_0003965-3` (score=24.2087) - How is metachromatic leukodystrophy inherited? Metachromatic leukodystrophy is inherited in an autosomal recessive manner. This means that both copies of the disease-causing gene in each cell must have a mutation for ...
4. `medquad_ans_0000647-2` (score=23.1086) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigra...
5. `medquad_ans_0000647-5` (score=21.8056) - These resources address the diagnosis or management of metachromatic leukodystrophy: - Gene Review: Gene Review: Arylsulfatase A Deficiency - Genetic Testing Registry: Metachromatic leukodystrophy - Genetic Testing Re...

**Answer**:

Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells, particularly affecting cells in the nervous system that produce myelin, the substance that insulates and protects nerves. This accumulation leads to progressive destruction of white matter (leukodystrophy) throughout the nervous system, including the brain, spinal cord, and peripheral nerves. The damage results in progressive deterioration of intellectual functions and motor skills, such as the ability to walk, as well as loss of sensation in the extremities (peripheral neuropathy), incontinence, seizures, paralysis, inability to speak, blindness, and hearing loss. Eventually, individuals may lose awareness of their surroundings and become unresponsive. While neurological problems are the primary feature, effects on other organs, such as the gallbladder, have also been reported. The most common form, affecting about 50 to 60 percent of individuals, is the late infant

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005135-1` (score=25.9424) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
2. `medquad_ans_0002946-1` (score=21.5434) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...
3. `medquad_ans_0002062-1` (score=21.2726) - What are the signs and symptoms of Ehlers-Danlos syndrome, spondylocheirodysplastic type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, spondylocheirodyspl...
4. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
5. `medquad_ans_0000549-1__dup1` (score=20.4489) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies (90%), elbow dislocation (90%), hypoplasia of the zygomatic bone (90%), scoliosis (90%), omphalocele (7.5%), hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, and tongue-like lumbar vertebral deformities [medquad_ans_0005135-1].

**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

### Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.502, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000251-3` (score=20.9622) - Cri-du-chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected individuals; studies suggest that...
2. `medquad_ans_0000251-1` (score=19.5109) - Cri-du-chat (cat's cry) syndrome, also known as 5p- (5p minus) syndrome, is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that ...
3. `medquad_ans_0000251-2` (score=19.1314) - Cri-du-chat syndrome occurs in an estimated 1 in 20,000 to 50,000 newborns. This condition is found in people of all ethnic backgrounds.
4. `medquad_ans_0000251-4` (score=18.9568) - Most cases of cri-du-chat syndrome are not inherited. The deletion occurs most often as a random event during the formation of reproductive cells (eggs or sperm) or in early fetal development. Affected people typicall...
5. `medquad_ans_0000251-5` (score=14.0312) - These resources address the diagnosis or management of cri-du-chat syndrome: - Cri du Chat Syndrome Support Group (UK): Diagnosis - Cri du Chat Syndrome Support Group (UK): Therapies - Genetic Testing Registry: 5p par...

**Answer**:

Cri-du-chat syndrome is a chromosomal condition that results when a piece of chromosome 5 is missing, specifically a deletion of the end of the short (p) arm of chromosome 5, written as 5p-. This condition is characterized by a high-pitched cry that sounds like a cat's cry, intellectual disability, delayed development, small head size (microcephaly), low birth weight, weak muscle tone (hypotonia) in infancy, and distinctive facial features such as widely set eyes (hypertelorism), low-set ears, a small jaw, and a rounded face. Some children are born with a heart defect. The severity of intellectual disability and developmental delay tends to be more pronounced with larger deletions. The syndrome occurs in an estimated 1 in 20,000 to 50,000 newborns and affects individuals of all ethnic backgrounds. Most cases are not inherited and occur as a random event

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Scores**: context_relevance=1.000, groundedness=0.991, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000719-2__dup1` (score=41.9090) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmen...
2. `medquad_ans_0000807-3` (score=38.6502) - In most people with polymicrogyria, the cause of the condition is unknown. However, researchers have identified several environmental and genetic factors that can be responsible for the disorder. Environmental causes ...
3. `medquad_ans_0000718-1__dup1` (score=38.4897) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
4. `medquad_ans_0000719-1__dup1` (score=38.3512) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides ...
5. `medquad_ans_0000807-5` (score=37.1632) - These resources address the diagnosis or management of polymicrogyria: - Gene Review: Gene Review: Polymicrogyria Overview - Genetic Testing Registry: Congenital bilateral perisylvian syndrome - Genetic Testing Regist...

**Answer**:

The signs and symptoms of bilateral frontoparietal polymicrogyria may include: moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze (eyes that are not aligned), ataxia, strabismus, increased muscle tone, finger dysmetria (difficulty controlling speed, distance, and/or power of movements), ankle clonus, Babinski sign, broad-based gait, cerebellar hypoplasia, cerebral dysmyelination, esotropia, exotropia, hyperreflexia, hypertonia, hypoplasia of the brainstem, nystagmus, polymicrogyria with an anterior to posterior gradient, and truncal ataxia [medquad_ans_0000719-2__dup1].

**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

### Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.977, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003414-3` (score=51.3651) - What causes Konigsmark Knox Hussels syndrome? Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid ar...
2. `medquad_ans_0003414-4` (score=42.0927) - Is genetic testing available for Konigsmark Knox Hussels syndrome? GeneTests lists the names of laboratories that are performing genetic testing for Konigsmark Knox Hussels syndrome. To view the contact information fo...
3. `medquad_ans_0003414-1` (score=38.3920) - Konigsmark Knox Hussels syndrome is an inherited condition that causes both hearing and vision loss. This condition is characterized by late-onset progressive sensorineural deafness and progressive optic atrophy, whic...
4. `medquad_ans_0003414-2` (score=31.5964) - What are the signs and symptoms of Konigsmark Knox Hussels syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Konigsmark Knox Hussels syndrome. If the information is available...
5. `medquad_ans_0001853-3` (score=13.3144) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid arginine with the amino acid histidine at position 445 in the OPA1 protein, written as Arg445His or R445H [medquad_ans_0003414-3]. This mutation is associated with both hearing and vision loss in affected individuals [medquad_ans_0003414-3]. The condition is inherited in an autosomal dominant fashion [medquad_ans_0003414-1].

**Supported citations**: `medquad_ans_0003414-3`, `medquad_ans_0003414-1`
**Unsupported citations**: _none_

### Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.962, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004584-1` (score=31.4660) - Orofaciodigital syndrome 1 (OFD1), also called orofaciodigital syndrome type 1, is a condition that affects the development of the oral cavity (the mouth and teeth), facial features, and digits (fingers and toes). Thi...
2. `medquad_ans_0004584-3` (score=25.9462) - Is genetic testing available for orofaciodigital syndrome 1 (OFD1)? Genetic testing for orofaciodigital syndrome 1 is clinically available. OFD1 is the only gene currently known to be associated with this condition. T...
3. `medquad_ans_0004586-1` (score=25.8345) - What are the signs and symptoms of Orofaciodigital syndrome 11? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 11. If the information is available, the tabl...
4. `medquad_ans_0000754-5` (score=25.0897) - These resources address the diagnosis or management of oral-facial-digital syndrome: - Gene Review: Gene Review: Oral-Facial-Digital Syndrome Type I - Genetic Testing Registry: Mohr syndrome - Genetic Testing Registry...
5. `medquad_ans_0004589-1` (score=25.0031) - Orofaciodigital syndrome (OFDS) type 2 is a genetic condition that was first described in 1941 by Mohr. OFDS type 2 belongs to a group of disorders called orofaciodigital syndromes (OFDS) characterized by mouth malfor...

**Answer**:

The symptoms of Orofaciodigital syndrome 1 (OFD1) include abnormalities in the oral cavity (such as cleft lip or palate and oral abnormalities), facial features, digits (such as syndactyly or webbing of the fingers or toes), and polycystic kidney disease. [medquad_ans_0004584-1]

**Supported citations**: `medquad_ans_0004584-1`
**Unsupported citations**: _none_

### Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000286-1__dup2` (score=13.5239) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...
2. `medquad_ans_0000130-1__dup3` (score=12.2557) - Summary : If you need a mastectomy, you have a choice about whether or not to have surgery to rebuild the shape of the breast. Instead of breast reconstruction, you could choose to wear a breast form that replaces the...
3. `medquad_ans_0000006-18` (score=11.9680) - You can help your doctor make a diagnosis by writing down key information about your dizziness or balance problem beforehand and giving the information to your doctor during the visit. Write down answers to these ques...
4. `medquad_ans_0000136-9` (score=11.5392) - You can't prevent vasculitis. However, treatment can help prevent or delay the complications of vasculitis. People who have severe vasculitis are treated with prescription medicines. Rarely, surgery may be done. Peopl...
5. `medquad_ans_0003325-3` (score=11.1459) - Is juvenile myoclonic epilepsy inherited? If I have juvenile myoclonic epilepsy, will my children also have it? Juvenile myoclonic epilepsy is an inherited disorder (about a third of patients with this condition have ...

**Answer**:

Yes, there is information about surgery. Specifically, the context mentions that if you need a mastectomy, you have a choice about whether or not to have surgery to rebuild the shape of the breast. Breast reconstruction may be done at the same time as the mastectomy or later, and if radiation therapy is part of the treatment plan, your doctor may suggest waiting until after radiation therapy. Additionally, it is recommended to talk to a plastic surgeon before the mastectomy, even if you plan to have reconstruction later on [medquad_ans_0000130-1__dup3].

**Supported citations**: `medquad_ans_0000130-1__dup3`
**Unsupported citations**: _none_

### Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Scores**: context_relevance=0.875, groundedness=0.914, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0002935-3` (score=36.2811) - What causes human T-cell leukemia virus, type 2? Human T-cell leukemia virus, type 2 (HTLV-2) occurs when a person is infected by the human T-cell leukemia retrovirus. HTLV-2 is spread by blood transfusions, sexual co...
2. `medquad_ans_0002935-4` (score=33.4920) - How is human T-cell leukemia virus, type 2 diagnosed? Human T-cell leukemia virus, type 2 (HTLV-2) is usually diagnosed based on blood tests that detect antibodies to the virus. However, HTLV-2 is often never suspecte...
3. `medquad_ans_0002934-3` (score=33.3345) - What causes human T-cell leukemia virus, type 1? Human T-cell leukemia virus, type 1 (HTLV-1) occurs when a person is infected by the human T-cell leukemia retrovirus. HTLV-1 is spread by blood transfusions, sexual co...
4. `medquad_ans_0002935-5` (score=32.6799) - How might human T-cell leukemia virus, type 2 be treated? No cure or treatment exists for human T-cell leukemia virus, type 2 (HTLV-2). Management is focused on early detection and preventing the spread of HTLV-2 to o...
5. `medquad_ans_0002934-1` (score=32.2913) - Human T-cell leukemia virus, type 1 (HTLV-1) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, some affected people may later de...

**Answer**:

Human T-cell leukemia virus type 2 (HTLV-2) is a retroviral infection that occurs when a person is infected by the human T-cell leukemia retrovirus. It is spread through blood transfusions, sexual contact, and sharing needles, and can also be transmitted from mother to child during birth or breastfeeding. Most people with HTLV-2 do not develop signs or symptoms and remain asymptomatic throughout their lives, although it is unclear why some individuals may develop neurological problems or other medical conditions while others do not [medquad_ans_0002935-3]. There is no cure or treatment for HTLV-2, and management focuses on early detection and preventing transmission through screening blood donors, promoting safe sex, discouraging needle sharing, and reducing mother-to-child transmission by screening pregnant women and avoiding breastfeeding if infected [medquad_ans_0002935-5]. Diagnosis is typically made via blood

**Supported citations**: `medquad_ans_0002935-3`, `medquad_ans_0002935-5`
**Unsupported citations**: _none_

### Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Scores**: context_relevance=0.833, groundedness=0.969, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000522-1__dup1` (score=26.1443) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...
2. `medquad_ans_0001152-1` (score=25.0035) - Children's interstitial and diffuse lung disease (chILD) is not a single condition, but a group of rare lung diseases found in infants, children and adolescents that can range from mild to severe. All types of chILD d...
3. `medquad_ans_0001152-2` (score=24.8473) - How might chILD be treated? There is no single treatment for interstitial lung diseases in children. Different forms of chILD require different treatments and support depending on the condition. The goals of treatment...
4. `medquad_ans_0000038-1__dup2` (score=23.2236) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chroni...
5. `medquad_ans_0000038-5__dup1` (score=22.3722) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...

**Answer**:

Interstitial lung diseases (ILDs) are a large group of diseases that inflame or scar the lungs. The inflammation and scarring, known as pulmonary fibrosis, make it difficult to get enough oxygen. These diseases can be caused by breathing in dust or other particles in the air, such as coal dust, farm dust, asbestos fibers, iron from mines or welding fumes, or silica dust. Other causes include autoimmune diseases or occupational exposures to molds, gases, or fumes. Some types of ILDs have no known cause. Treatment depends on the type of exposure and the stage of the disease and may involve medicines, oxygen therapy, or a lung transplant in severe cases [medquad_ans_0000522-1__dup1]. In children, interstitial and diffuse lung disease (chILD) refers to a group of rare lung diseases affecting infants, children, and adolescents, which decrease the ability to supply oxygen to the body

**Supported citations**: `medquad_ans_0000522-1__dup1`
**Unsupported citations**: _none_

### Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.676, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000645-1` (score=19.3859) - Melnick-Needles syndrome is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which also ...
2. `medquad_ans_0000645-2` (score=19.1861) - Melnick-Needles syndrome is a rare disorder; fewer than 100 cases have been reported worldwide.
3. `medquad_ans_0000645-5` (score=16.3615) - These resources address the diagnosis or management of Melnick-Needles syndrome: - Gene Review: Gene Review: Otopalatodigital Spectrum Disorders - Genetic Testing Registry: Melnick-Needles syndrome These resources fro...
4. `medquad_ans_0000645-3` (score=15.7303) - Mutations in the FLNA gene cause Melnick-Needles syndrome. The FLNA gene provides instructions for producing the protein filamin A, which helps build the network of protein filaments (cytoskeleton) that gives structur...
5. `medquad_ans_0001853-3` (score=14.3676) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

The provided context does not explicitly state whether Melnick-Needles syndrome is inherited. While it mentions that mutations in the FLNA gene cause the syndrome, it does not specify the inheritance pattern. Therefore, based on the given information, the inheritance status cannot be determined.

Answer: Insufficient context to determine if Melnick-Needles syndrome is inherited. [medquad_ans_0000645-3]

**Supported citations**: `medquad_ans_0000645-3`
**Unsupported citations**: _none_

### Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Scores**: context_relevance=1.000, groundedness=0.956, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006138-4` (score=17.6638) - Is tracheobronchomalacia inherited? Primary tracheobronchomalacia (TBM) is often associated with certain genetic conditions. In some cases, an affected person inherits the condition from an affected parent. Other case...
2. `medquad_ans_0006138-5` (score=17.5807) - How is tracheobronchomalacia diagnosed? A diagnosis of tracheobronchomalacia (TBM) may be suspected based on the presence of characteristic signs and symptoms or abnormal pulmonary function tests. Additional testing s...
3. `medquad_ans_0006138-6` (score=15.3433) - How might tracheobronchomalacia be treated? Treatment is only medically necessary in people who have signs and symptoms of tracheobronchomalacia (TBM). Management of symptomatic TBM first involves identifying underlyi...
4. `medquad_ans_0006138-2` (score=13.8157) - What are the signs and symptoms of tracheobronchomalacia? Tracheobronchomalacia (TBM) is a condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway ...
5. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Tracheobronchomalacia (TBM) is a condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse [medquad_ans_0006138-2]. There are two forms: primary TBM (also called congenital TBM), which typically develops during infancy or early childhood, and secondary TBM (also called acquired TBM), which is usually seen in adults and results from trauma, chronic inflammation, or prolonged compression of the airways [medquad_ans_0006138-2]. Primary TBM may be inherited from an affected parent or result from new (de novo) gene mutations, while secondary TBM is not inherited [medquad_ans_0006138-4]. The condition is typically progressive and may present with signs and symptoms such as shortness

**Supported citations**: `medquad_ans_0006138-2`, `medquad_ans_0006138-4`
**Unsupported citations**: _none_

### Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Scores**: context_relevance=1.000, groundedness=0.993, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006030-1` (score=30.8094) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people m...
2. `medquad_ans_0000531-1__dup2` (score=17.4806) - Your jaw is a set of bones that holds your teeth. It consists of two main parts. The upper part is the maxilla. It doesn't move. The moveable lower part is called the mandible. You move it when you talk or chew. The t...
3. `medquad_ans_0004881-1` (score=16.8206) - What are the signs and symptoms of Pillay syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pillay syndrome. If the information is available, the table below includes how oft...
4. `medquad_ans_0000371-1__dup1` (score=15.2276) - What are the signs and symptoms of Ankylosis of teeth? The Human Phenotype Ontology provides the following list of signs and symptoms for Ankylosis of teeth. If the information is available, the table below includes h...
5. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene, and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

### Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.276, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000554-2` (score=20.5439) - Kallmann syndrome is estimated to affect 1 in 10,000 to 86,000 people and occurs more often in males than in females. Kallmann syndrome 1 is the most common form of the disorder.
2. `medquad_ans_0003339-3` (score=19.8213) - How is Kallmann syndrome inherited? Kallmann syndrome (KS) may be inherited in an X-linked recessive, autosomal dominant, or autosomal recessive manner depending on the gene(s) responsible. For example: KS due to muta...
3. `medquad_ans_0000554-4` (score=19.5163) - Kallmann syndrome 1 (caused by ANOS1 gene mutations) has an X-linked recessive pattern of inheritance. The ANOS1 gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only on...
4. `medquad_ans_0000554-1` (score=19.0699) - Kallmann syndrome is a condition characterized by delayed or absent puberty and an impaired sense of smell. This disorder is a form of hypogonadotropic hypogonadism (HH), which is a condition affecting the production ...
5. `medquad_ans_0001478-4` (score=19.0428) - Is congenital anosmia inherited? Most cases of isolated congenital anosmia (not associated with additional symptoms) occur sporadically in people with no family history of the condition. Rarely, more than one family m...

**Answer**:

Kallmann syndrome may be inherited in an X-linked recessive, autosomal dominant, or autosomal recessive manner depending on the gene(s) responsible. For example, Kallmann syndrome 1, caused by mutations in the KAL1 (ANOS1) gene, is inherited in an X-linked recessive manner, while mutations in genes such as FGFR1, PROKR2, PROK2, CHD7, or FGF8 are associated with autosomal dominant inheritance. Some forms can also be inherited in an autosomal recessive manner. However, in the majority of cases, the family history is negative, indicating sporadic occurrence. Affected individuals are at risk of passing the mutation to their children, with the risk varying by genetic cause and potentially reaching up to 50%. In many cases, the genetic cause remains unknown [doc_id: medquad_ans_0003339-3].

**Supported citations**: _none_
**Unsupported citations**: `doc_id: medquad_ans_0003339-3`

### Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0005967-3` (score=25.1493) - How is syndactyly type 3 inherited? Syndactyly type 3 has been shown to be inherited in an autosomal dominant manner. This means that having only one mutated copy of the causative gene is sufficient to cause the condi...
2. `medquad_ans_0005967-2` (score=23.4224) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how...
3. `medquad_ans_0005967-1` (score=23.4072) - Syndactyly type 3 (SD3) is a limb abnormality present at birth that is characterized by complete fusion of the 4th and 5th fingers on both hands. In most cases only the soft tissue is fused, but in some cases the bone...
4. `medquad_ans_0005970-1` (score=23.2410) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how...
5. `medquad_ans_0005020-1` (score=22.8541) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the tabl...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and fusion of the 4th and 5th fingers on both hands. In most cases, only the soft tissue is fused, but in some cases, the bones of the fingers (distal phalanges) are fused [medquad_ans_0005967-2].

**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

### Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Scores**: context_relevance=0.800, groundedness=0.975, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000021-1__dup3` (score=22.4908) - Espaol Bronchitis (bron-KI-tis) is a condition in which the bronchial tubes become inflamed. These tubes carry air to your lungs. (For more information about the bronchial tubes and airways, go to the Diseases and Con...
2. `medquad_ans_0000021-5__dup1` (score=21.5670) - Acute Bronchitis Acute bronchitis caused by an infection usually develops after you already have a cold or the flu. Symptoms of a cold or the flu include sore throat, fatigue (tiredness), fever, body aches, stuffy or ...
3. `medquad_ans_0000197-1__dup3` (score=19.6850) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...
4. `medquad_ans_0000021-4__dup2` (score=19.6247) - Bronchitis is a very common condition. Millions of cases occur every year. Elderly people, infants, and young children are at higher risk for acute bronchitis than people in other age groups. People of all ages can de...
5. `medquad_ans_0000021-3__dup2` (score=18.6122) - Acute Bronchitis Infections or lung irritants cause acute bronchitis. The same viruses that cause colds and the flu are the most common cause of acute bronchitis. Sometimes bacteria can cause the condition. Certain su...

**Answer**:

Chronic bronchitis is an ongoing, serious condition that occurs when the lining of the bronchial tubes is constantly irritated and inflamed, causing a long-term cough with mucus. Smoking is the main cause of chronic bronchitis. Viruses or bacteria can easily infect the irritated bronchial tubes, worsening the condition and causing symptoms to last longer. People with chronic bronchitis experience periods when symptoms get much worse than usual. It is a serious, long-term medical condition, and early diagnosis, treatment, quitting smoking, and avoiding secondhand smoke can improve quality of life. The chance of complete recovery is low for people who have severe chronic bronchitis [medquad_ans_0000021-1__dup3].

**Supported citations**: `medquad_ans_0000021-1__dup3`
**Unsupported citations**: _none_

### Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Scores**: context_relevance=0.875, groundedness=0.714, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000935-1` (score=29.8662) - Spinal muscular atrophy is a genetic disorder that affects the control of muscle movement. It is caused by a loss of specialized nerve cells, called motor neurons, in the spinal cord and the part of the brain that is ...
2. `medquad_ans_0005750-1` (score=28.3325) - What are the signs and symptoms of Spinal muscular atrophy Ryukyuan type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinal muscular atrophy Ryukyuan type. If the information is...
3. `medquad_ans_0005751-1` (score=28.3027) - What are the signs and symptoms of Spinal muscular atrophy type 1 with congenital bone fractures? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinal muscular atrophy type 1 with ...
4. `medquad_ans_0000935-3` (score=27.9939) - Mutations in the SMN1, UBA1, DYNC1H1, and VAPB genes cause spinal muscular atrophy. Extra copies of the SMN2 gene modify the severity of spinal muscular atrophy. The SMN1 and SMN2 genes provide instructions for making...
5. `medquad_ans_0000197-3__dup1` (score=27.8511) - Prognosis varies depending on the type of MND and the age of onset. Some MNDs, such as primary lateral sclerosis and Kennedy disease, are not fatal and progress slowly. Patients with spinal muscular atrophy may appear...

**Answer**:

The provided context does not mention any treatments for spinal muscular atrophy. Therefore, based on the given information, it is not possible to answer the question about treatments. [medquad_ans_0000935-1]

**Supported citations**: `medquad_ans_0000935-1`
**Unsupported citations**: _none_

### Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Scores**: context_relevance=0.889, groundedness=0.830, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000001_1-1` (score=42.9934) - Key Points - Adult acute lymphoblastic leukemia (ALL) is a type of cancer in which the bone marrow makes too many lymphocytes (a type of white blood cell). - Leukemia may affect red blood cells, white blood cells, and...
2. `medquad_ans_0000001_1-7` (score=42.9364) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - C...
3. `medquad_ans_0000001_6-1` (score=40.2190) - Key Points - Childhood acute lymphoblastic leukemia (ALL) is a type of cancer in which the bone marrow makes too many immature lymphocytes (a type of white blood cell). - Leukemia may affect red blood cells, white blo...
4. `medquad_ans_0000001_2-1` (score=39.4908) - Key Points - Adult acute myeloid leukemia (AML) is a type of cancer in which the bone marrow makes abnormal myeloblasts (a type of white blood cell), red blood cells, or platelets. - Leukemia may affect red blood cell...
5. `medquad_ans_0000001_6-7` (score=39.2317) - Key Points - There are different types of treatment for childhood acute lymphoblastic leukemia (ALL). - Children with ALL should have their treatment planned by a team of doctors who are experts in treating childhood ...

**Answer**:

The provided context does not mention specific treatments for Adult Acute Lymphoblastic Leukemia. While it describes the disease, its signs and symptoms, diagnostic methods, and how it affects blood cells, no information about treatment options is included [medquad_ans_0000001_1-1]. Therefore, based on the given context, the treatments for Adult Acute Lymphoblastic Leukemia cannot be determined.

**Supported citations**: `medquad_ans_0000001_1-1`
**Unsupported citations**: _none_

### Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000388-1` (score=25.6117) - Friedreich ataxia is a genetic condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features of...
2. `medquad_ans_0002439-3` (score=24.3153) - What causes Friedreich ataxia? Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known...
3. `medquad_ans_0000388-5` (score=23.0066) - These resources address the diagnosis or management of Friedreich ataxia: - Friedreich's Ataxia Research Alliance: Clinical Care Guidelines - Gene Review: Gene Review: Friedreich Ataxia - Genetic Testing Registry: Fri...
4. `medquad_ans_0002439-1` (score=21.4401) - Friedreich ataxia is an inherited condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features...
5. `medquad_ans_0002439-4` (score=18.7576) - How is Friedreich ataxia inherited? Friedreich ataxia is inherited in an autosomal recessive manner. This means that to be affected, a person must have a mutation in both copies of the responsible gene in each cell. T...

**Answer**:

Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known as a GAA trinucleotide repeat. Normally, this segment is repeated 5 to 33 times within the FXN gene. In people with Friedreich ataxia, the GAA segment is repeated 66 to more than 1,000 times. The length of the GAA trinucleotide repeat appears to be related to the age at which the symptoms of Friedreich ataxia appear. The abnormally long GAA trinucleotide repeat disrupts the production of frataxin, which severely reduces the amount of this protein in cells. Certain nerve and muscle cells cannot function properly with a shortage of frataxin, leading to the characteristic signs and symptoms of Friedreich ataxia [med

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Scores**: context_relevance=0.833, groundedness=0.980, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000341-2` (score=32.2098) - Atrial fibrillation is the most common type of sustained abnormal heart rhythm (arrhythmia), affecting more than 3 million people in the United States. The risk of developing this irregular heart rhythm increases with...
2. `medquad_ans_0000341-1` (score=31.9159) - Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm. This condition is characterized by uncoordinated electrical activity in the heart's upper chambers (the atria), which cau...
3. `medquad_ans_0002230-3` (score=29.9478) - How might familial atrial fibrillation be treated? We are unaware of treatment recommendations specific to familial atrial fibrillation, but there is information available about treatment for atrial fibrillation in ge...
4. `medquad_ans_0002230-1` (score=28.7406) - Familial atrial fibrillation is an inherited heart condition that disrupts the heart's rhythm. It is characterized by erratic electrical activity in the heart's upper chambers (the atria), causing an irregular respons...
5. `medquad_ans_0000341-4` (score=28.2846) - Familial atrial fibrillation appears to be inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm, characterized by uncoordinated or erratic electrical activity in the heart's upper chambers (the atria), which causes a fast and irregular heartbeat (arrhythmia) [medquad_ans_0000341-1]. This abnormal rhythm can lead to complications such as dizziness, chest pain, palpitations, shortness of breath, or fainting (syncope), and increases the risk of stroke and sudden death [medquad_ans_0000341-1]. While the incidence of the familial form is unknown, recent studies suggest that up to 30 percent of all people with atrial fibrillation may have a family history of the condition [medquad_ans_0000341-2]. Treatment recommendations specific to familial atrial fibrillation are not available, but general treatment for atrial fibr

**Supported citations**: `medquad_ans_0000341-1`, `medquad_ans_0000341-2`
**Unsupported citations**: _none_

### Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0004627-1` (score=46.4462) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the assoc...
2. `medquad_ans_0004627-2` (score=31.6392) - What are the signs and symptoms of Osteopathia striata cranial sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopathia striata cranial sclerosis. If the information is...
3. `medquad_ans_0003378-1` (score=16.2177) - What are the signs and symptoms of Keratosis palmoplantaris striata 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Keratosis palmoplantaris striata 1. If the information is avail...
4. `medquad_ans_0003615-1` (score=13.9223) - Limited cutaneous systemic sclerosis is a subtype of systemic sclerosis characterized by the association of Raynaud's phenomenon and skin fibrosis on the hands, face, feet and forearms. The exact cause of limited cuta...
5. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary significantly from person to person, even within a single family. Features of the condition are generally present at birth and may include skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected people may also have developmental delay, hearing loss, heart defects, and/or ophthalmoplegia (paralysis of the muscles surrounding the eyes). Osteopathia striata cranial sclerosis is caused by changes (mutations) in the WTX gene and is inherited in an X-linked dominant manner. Treatment is based on the signs and symptoms present in each person [medquad_ans_0004

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Scores**: context_relevance=1.000, groundedness=0.313, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0004097-3` (score=38.1071) - What causes mixed connective tissue disease? The exact underlying cause of mixed connective tissue disease (MCTD) is currently unknown. It is an autoimmune disorder, which means the immune system mistakes normal, heal...
2. `medquad_ans_0004097-4` (score=36.3231) - Is mixed connective tissue disease inherited? The role of genetics in the onset of mixed connective tissue disease (MCTD) is still unclear. Some people with MCTD have family members who are also affected by the condit...
3. `medquad_ans_0004097-5` (score=33.7111) - How is mixed connective tissue disease diagnosed? Mixed connective tissue disease (MCTD) is often suspected after a physical examination reveals signs and symptoms associated with the condition. The diagnosis is suppo...
4. `medquad_ans_0004097-6` (score=30.6313) - How might mixed connective tissue disease be treated? There is currently no cure for mixed connective tissue disease (MCTD). However, treatments can help manage symptoms of the condition. For example, medications such...
5. `medquad_ans_0004097-2` (score=30.5579) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus eryt...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. A condition called Raynaud's phenomenon sometimes occurs months or years before other symptoms of MCTD develop. Most people with MCTD have pain in multiple joints, and/or inflammation of joints (arthrit[doc_id]medquad_ans_0004097-2[/doc_id]).

**Supported citations**: _none_
**Unsupported citations**: `doc_id`, `/doc_id`

### Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Scores**: context_relevance=1.000, groundedness=0.844, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000153-1__dup2` (score=28.4805) - Summary : Caregivers provide help to another person in need. The person receiving care may be an adult - often a parent or a spouse - or a child with special medical needs. Some caregivers are family members. Others a...
2. `medquad_ans_0000024-1__dup2` (score=22.3109) - Caring for someone who has Alzheimer's disease (AD) can be stressful and overwhelming. It's important to take care of yourself. Ask for and accept help. Talk to the doctor. Find out what treatments might help control ...
3. `medquad_ans_0000004-24` (score=21.3722) - Currently, no medicines or treatments are known to prevent Alzheimer's disease, but scientists are studying many possibilities. These possibilities include lifestyle factors such as exercise and physical activity, a h...
4. `medquad_ans_0000004-5__dup2` (score=21.2568) - Medications Can Treat Symptoms There is no known cure for Alzheimer's disease, but there are medicines that can treat symptoms of the disease. Most Alzheimers medicines work best for people in the mild or moderate sta...
5. `medquad_ans_0000003-6__dup1` (score=21.1095) - Most people with Alzheimers disease are cared for at home by family members. Within families, caregiving is provided most often by wives and husbands, followed by daughters. As Alzheimers disease gets worse, the perso...

**Answer**:

The provided context does not specify treatments for Alzheimer's caregiving. It discusses treatments for Alzheimer's disease symptoms, such as medications that can help with memory loss or behavioral symptoms, and mentions support groups, respite services, and planning for the future as helpful for caregivers [medquad_ans_0000024-1__dup2]. However, there is no direct information on treatments specifically for caregiving itself. Therefore, based on the given context, the answer cannot be fully determined.

**Supported citations**: `medquad_ans_0000024-1__dup2`
**Unsupported citations**: _none_

### Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Scores**: context_relevance=0.900, groundedness=0.921, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000511-4` (score=39.0396) - Most cases of idiopathic inflammatory myopathy are sporadic, which means they occur in people with no history of the disorder in their family. However, several people with idiopathic inflammatory myopathy have had clo...
2. `medquad_ans_0000511-3` (score=38.5004) - Idiopathic inflammatory myopathy is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Researchers have identi...
3. `medquad_ans_0003141-1` (score=37.0147) - Idiopathic inflammatory myopathy refers to a group of conditions that affect the skeletal muscles (muscles used for movement). Although the condition can be diagnosed at any age, idiopathic inflammatory myopathy most ...
4. `medquad_ans_0000511-1` (score=35.7267) - Idiopathic inflammatory myopathy is a group of disorders characterized by inflammation of the muscles used for movement (skeletal muscles). Idiopathic inflammatory myopathy usually appears in adults between ages 40 an...
5. `medquad_ans_0000547-3` (score=33.9417) - Juvenile idiopathic arthritis is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Its signs and symptoms res...

**Answer**:

Genetic changes related to idiopathic inflammatory myopathy include variations in several genes, most commonly within the human leukocyte antigen (HLA) complex. These HLA gene variations may influence the risk of developing the disorder by affecting how the immune system distinguishes between the body's own proteins and foreign invaders. Specific variations of certain HLA genes have been associated with an increased risk of idiopathic inflammatory myopathy. Researchers are also studying variations in other genes related to immune function to better understand their contribution to the condition's development [medquad_ans_0000511-3].

**Supported citations**: `medquad_ans_0000511-3`
**Unsupported citations**: _none_

### Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.135, answer_relevance=0.967

**Top docs**:

1. `medquad_ans_0005223-6` (score=30.7246) - How might pyruvate dehydrogenase deficiency be treated? Treatment of pyruvate dehydrogenase (PDH) deficiency rarely influences the course of the disease, but goals include stimulating the pyruvate dehydrogenase comple...
2. `medquad_ans_0005223-3` (score=30.1591) - What causes pyruvate dehydrogenase deficiency? Pyruvate dehydrogenase (PDH) deficiency is usually caused by a deficiency of one or more enzymes or cofactors (such as thiamine) that are needed for an important chemical...
3. `medquad_ans_0000855-5` (score=29.9131) - These resources address the diagnosis or management of pyruvate dehydrogenase deficiency: - Genetic Testing Registry: Pyruvate dehydrogenase E1-beta deficiency - Genetic Testing Registry: Pyruvate dehydrogenase E2 def...
4. `medquad_ans_0005223-5` (score=29.5572) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for...
5. `medquad_ans_0000855-3` (score=29.1043) - The genes involved in pyruvate dehydrogenase deficiency each provide instructions for making a protein that is a component of a group of proteins called the pyruvate dehydrogenase complex. This complex plays an import...

**Answer**:

The provided context does not contain information on how to diagnose pyruvate dehydrogenase deficiency. Therefore, the question cannot be answered based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.501, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006562-1` (score=38.7687) - 17q23.1q23.2 microdeletion syndrome is a condition caused by a small deletion of genetic material from chromosome 17. The deletion occurs at a location encompassing bands 23.1 to 23.2 on the long (q) arm of the chromo...
2. `medquad_ans_0006562-2` (score=31.3862) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb a...
3. `medquad_ans_0006562-5` (score=29.3275) - How is 17q23.1q23.2 microdeletion syndrome diagnosed? The deletion can be identified by comparative genomic hybridization (CGH) microarray and fluorescence in situ hybridization (FISH).
4. `medquad_ans_0006607-2` (score=27.2124) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
5. `medquad_ans_0001079-3` (score=26.3017) - Most people with a 1q21.1 microdeletion are missing a sequence of about 1.35 million DNA building blocks (base pairs), also written as 1.35 megabases (Mb), in the q21.1 region of chromosome 1. However, the exact size ...

**Answer**:

The symptoms of 17q23.1q23.2 microdeletion syndrome include developmental delay, microcephaly, short stature, heart defects, and limb abnormalities. Specific manifestations include mild to moderate developmental delay, particularly speech delays, heart defects such as patent ductus arteriosus or atrial septal defects, long and thin fingers and toes, hypoplasia or aplasia of the patellae (knee caps), scoliosis, and mild, unspecific unusual facial features. Common symptoms reported in approximately 90% of patients include arachnodactyly (long, thin fingers and toes), cognitive impairment, and long toe. Other symptoms occurring in about 50% of patients include microcephaly, intrauterine growth retardation, neurological speech impairment, patent ductus arteriosus, pulmonary hypertension, and short stature. Additional findings include abnormalities of the epiphysis, eyelashes,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 186 (`0003860-3`)

**Question**: What causes MECP2 duplication syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003860-3` (score=30.0691) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
2. `medquad_ans_0003860-4` (score=27.0481) - Is MECP2 duplication syndrome inherited? MECP2 duplication syndrome is inherited in an X-linked manner. A condition is considered X-linked if the genetic change that causes the condition is located on the X chromosome...
3. `medquad_ans_0000635-5` (score=26.7431) - These resources address the diagnosis or management of MECP2 duplication syndrome: - Cincinnati Children's Hospital: MECP2-Related Disorders - Cleveland Clinic: Spasticity - Gene Review: Gene Review: MECP2 Duplication...
4. `medquad_ans_0003860-5` (score=26.3027) - How is MECP2 duplication syndrome diagnosed? A diagnosis of MECP2 duplication syndrome is often suspected based on the presence of characteristic signs and symptoms. Genetic testing can then be ordered to confirm the ...
5. `medquad_ans_0000635-4` (score=25.4926) - MECP2 duplication syndrome is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome...

**Answer**:

MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located on the long (q) arm of the X chromosome. The MECP2 gene encodes a protein that is important for normal brain functioning, and one of its most important functions is to regulate other genes in the brain by switching them on and off. A duplication of the MECP2 gene leads to the production of excess protein, which is unable to properly regulate the expression of other genes, resulting in irregular brain activity and the signs and symptoms of MECP2 duplication syndrome [medquad_ans_0003860-3].

**Supported citations**: `medquad_ans_0003860-3`
**Unsupported citations**: _none_

### Example 187 (`0004633-2`)

**Question**: What are the symptoms of Osteopetrosis autosomal dominant type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.780, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000759-5` (score=32.9437) - These resources address the diagnosis or management of osteopetrosis: - Gene Review: Gene Review: CLCN7-Related Osteopetrosis - Genetic Testing Registry: Ectodermal dysplasia, anhidrotic, with immunodeficiency, osteop...
2. `medquad_ans_0004632-2` (score=31.8331) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 1. If the informatio...
3. `medquad_ans_0000759-2` (score=31.5973) - Autosomal dominant osteopetrosis is the most common form of the disorder, affecting about 1 in 20,000 people. Autosomal recessive osteopetrosis is rarer, occurring in an estimated 1 in 250,000 people. Other forms of o...
4. `medquad_ans_0004633-2` (score=29.5762) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 2. If the informatio...
5. `medquad_ans_0000759-3` (score=29.1684) - Mutations in at least nine genes cause the various types of osteopetrosis. Mutations in the CLCN7 gene are responsible for about 75 percent of cases of autosomal dominant osteopetrosis, 10 to 15 percent of cases of au...

**Answer**:

The provided context does not contain information about the symptoms of Osteopetrosis autosomal dominant type 2. [medquad_ans_0004632-2] lists symptoms for Osteopetrosis autosomal dominant type 1, but not for type 2. Therefore, based on the available context, the symptoms of Osteopetrosis autosomal dominant type 2 cannot be determined.

**Supported citations**: `medquad_ans_0004632-2`
**Unsupported citations**: _none_

### Example 188 (`0003459-4`)

**Question**: What are the treatments for Lambert Eaton myasthenic syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.753, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0003459-3` (score=58.4309) - What causes Lambert Eaton myasthenic syndrome? Lambert Eaton myasthenic syndrome is the result of an autoimmune process which causes a disruption of electrical impulses between nerve cells and muscle fibers. In cases ...
2. `medquad_ans_0005914-2` (score=42.4301) - What are the signs and symptoms of subacute cerebellar degeneration? Signs and symptoms of subacute cerebellar degeneration, include ataxia, speech and swallowing problems, dementia (in about half of people with this ...
3. `medquad_ans_0003459-1` (score=37.8492) - Lambert Eaton myasthenic syndrome (LEMS) is a disorder of the neuromuscular junction. The neuromuscular junction is the site where nerve cells meet muscle cells and help activate the muscles. This syndrome occurs when...
4. `medquad_ans_0000235-3` (score=26.2428) - Mutations in many genes can cause congenital myasthenic syndrome. Mutations in the CHRNE gene are responsible for more than half of all cases. A large number of cases are also caused by mutations in the RAPSN, CHAT, C...
5. `medquad_ans_0001540-1` (score=24.3369) - Congenital myasthenic syndrome (CMS) is a group of genetic disorders that result in muscle weakness and fatigue. Symptoms can range from mild weakness to progressive disabling weakness. There are three main subtypes o...

**Answer**:

The provided context does not specify the treatments for Lambert Eaton myasthenic syndrome. It mentions that treatment of an underlying disorder or cancer is the first priority [medquad_ans_0003459-1], but no specific therapies are detailed. Therefore, based on the given information, the treatments for Lambert Eaton myasthenic syndrome cannot be determined.

**Supported citations**: `medquad_ans_0003459-1`
**Unsupported citations**: _none_

### Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Scores**: context_relevance=0.857, groundedness=0.967, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000445-3__dup1` (score=33.3078) - How might arginase deficiency be treated? The treatment and management of arginase deficiency is generally focused on lowering arginine levels and preventing hyperammonemia (an accumulation of ammonia in the blood). T...
2. `medquad_ans_0000065-3` (score=30.6746) - Mutations in the ARG1 gene cause arginase deficiency. Arginase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occurs in liver cells. This ...
3. `medquad_ans_0000065-1` (score=29.9062) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down ...
4. `medquad_ans_0000445-1__dup1` (score=28.2722) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...
5. `medquad_ans_0000065-5` (score=27.9829) - These resources address the diagnosis or management of arginase deficiency: - Baby's First Test - Gene Review: Gene Review: Arginase Deficiency - Gene Review: Gene Review: Urea Cycle Disorders Overview - Genetic Testi...

**Answer**:

The symptoms of arginase deficiency usually become evident by about the age of 3 and most often appear as stiffness, [medquad_ans_0000065-1]. Additionally, the accumulation of ammonia and arginine in the blood is believed to cause neurological problems and other signs and symptoms of the disorder [medquad_ans_0000065-1].

**Supported citations**: `medquad_ans_0000065-1`
**Unsupported citations**: _none_

### Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Scores**: context_relevance=0.889, groundedness=0.981, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006376-6` (score=47.8880) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but ar...
2. `medquad_ans_0006376-3` (score=44.9398) - What causes Von Hippel-Lindau disease? Von Hippel-Lindau (VHL) disease is caused by a mutation in the VHL gene. This gene is a tumor suppressor gene, which helps to control cell growth. Mutations in the VHL gene lead ...
3. `medquad_ans_0006376-5` (score=42.9892) - How is von Hippel-Lindau (VHL) disease diagnosed? The diagnosis of von Hippel-Lindau (VHL) disease can be made based on specific clinical criteria (signs and symptoms), or when molecular genetic testing reveals a muta...
4. `medquad_ans_0000022-1__dup1` (score=35.1697) - von Hippel-Lindau disease (VHL) is a rare, genetic multi-system disorder in which non-cancerous tumors grow in certain parts of the body. Slow-growing hemgioblastomas -- benign tumors with many blood vessels -- may de...
5. `medquad_ans_0000961-1__dup1` (score=33.4698) - Von Hippel-Lindau disease (VHL) is a rare, genetic disease that causes tumors and cysts to grow in your body. The tumors can be either cancerous or benign. They can grow in your brain and spinal cord, kidneys, pancrea...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms but are still small to prevent permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All people with VHL disease should be carefully followed by a physician or medical team familiar with the disorder [medquad_ans_0006376-6].

**Supported citations**: `medquad_ans_0006376-6`
**Unsupported citations**: _none_

### Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Scores**: context_relevance=1.000, groundedness=0.733, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000738-5` (score=21.3069) - These resources address the diagnosis or management of Noonan syndrome: - Gene Review: Gene Review: Noonan Syndrome - Genetic Testing Registry: Noonan syndrome - Genetic Testing Registry: Noonan syndrome 1 - Genetic T...
2. `medquad_ans_0004456-2` (score=19.4995) - What are the signs and symptoms of Noonan syndrome 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome 6. If the information is available, the table below includes how...
3. `medquad_ans_0004454-1` (score=18.9495) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004453-1` (score=18.9495) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004452-1` (score=18.9495) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

The provided context does not contain information about Noonan syndrome 3. Therefore, it is not possible to answer the question based on the given resources. [medquad_ans_0000738-5] and [medquad_ans_0004456-2] discuss Noonan syndrome in general and Noonan syndrome 6, respectively, but there is no mention of Noonan syndrome 3.

**Supported citations**: `medquad_ans_0000738-5`, `medquad_ans_0004456-2`
**Unsupported citations**: _none_

### Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Scores**: context_relevance=1.000, groundedness=0.538, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0005865-3` (score=33.8643) - How might Sprengel deformity be treated? Treatment of Sprengel deformity depends on the degree of functional impairment and/or cosmetic disfigurement. Many people with Sprengel deformity do not need surgery and may ha...
2. `medquad_ans_0005865-2` (score=29.8118) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may...
3. `medquad_ans_0005865-1` (score=28.2994) - Sprengel deformity is a congenital condition characterized by abnormal development and elevation of the shoulder blade (scapula). Severity can range considerably from being almost invisible when covered with clothes, ...
4. `medquad_ans_0001079-1__dup1` (score=23.5237) - What are the signs and symptoms of Cervical ribs, Sprengel anomaly, anal atresia, and urethral obstruction? The Human Phenotype Ontology provides the following list of signs and symptoms for Cervical ribs, Sprengel an...
5. `medquad_ans_0003402-2` (score=22.9760) - What are the signs and symptoms of Klippel Feil syndrome? Klippel Feil syndrome is characterized by the fusion of 2 or more spinal bones in the neck (cervical vertebrae). The condition is present from birth (congenita...

**Answer**:

Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may not have noticeable signs or symptoms. It more commonly occurs on the left side, but can occur on both sides (bilaterally). In addition to shoulder asymmetry, the elevated shoulder blade may cause a lump in the back of the base of the neck; underdeveloped or incomplete muscles in the surrounding area; and limited movement of the shoulder and arm on the affected side. Some people have bone, cartilage or fiber-like tissue between the shoulder blade and the spinal bones (vertebrae) next to it. Other features that have been found in association with Sprengel deformity include: scoliosis, Klippel-Feil syndrome, limb length discrepancy, an underdeveloped backbone (hemivertebrae), missing, fused, or extra ribs (cervical ribs),

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.929, answer_relevance=0.844

**Top docs**:

1. `medquad_ans_0005415-3` (score=29.3333) - How is Robinow syndrome inherited? Robinow syndrome may be inherited in an autosomal recessive or autosomal dominant manner. Autosomal recessive (AR) inheritance means both copies of the responsible gene in each cell ...
2. `medquad_ans_0005415-1` (score=28.7223) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
3. `medquad_ans_0005415-4` (score=28.7025) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...
4. `medquad_ans_0000882-1` (score=28.5933) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
5. `medquad_ans_0000882-3` (score=26.0670) - Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene. This gene provides instructions for making a protein whose function is not well understood, although it is involved in chemical signaling p...

**Answer**:

Genetic changes related to Robinow syndrome include mutations in the ROR2 gene, which are associated with autosomal recessive Robinow syndrome, and mutations in the WNT5A or DVL1 gene, which are associated with autosomal dominant Robinow syndrome. In some cases, the underlying cause of Robinow syndrome is unknown [medquad_ans_0005415-1].

**Supported citations**: `medquad_ans_0005415-1`
**Unsupported citations**: _none_

### Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Scores**: context_relevance=1.000, groundedness=0.711, answer_relevance=0.782

**Top docs**:

1. `medquad_ans_0000138-11` (score=54.6068) - You and your doctor will work together to choose a treatment that's best for you. The publications of the NIDDK Kidney Failure Series can help you learn about the specific issues you will face. Booklets - Treatment Me...
2. `medquad_ans_0000136-12` (score=53.3634) - You and your doctor will work together to choose a treatment that's best for you. The publications of the NIDDK Kidney Failure Series can help you learn about the specific issues you will face. Booklets - What I need ...
3. `medquad_ans_0000179-14` (score=52.4044) - You and your doctor will work together to choose a treatment that's best for you. The publications of the NIDDK Kidney Failure Series can help you learn about the specific issues you will face. Booklets - What I need ...
4. `medquad_ans_0000135-9` (score=46.4629) - No specific diet will prevent or delay acquired cystic kidney disease. In general, a diet designed for people on hemodialysis or peritoneal dialysis reduces the amount of wastes that accumulate in the body between dia...
5. `medquad_ans_0000158-13` (score=45.6189) - For people who are on dialysis or approaching total kidney failure, adequate nutrition is important for maintaining energy, strength, healthy sleep patterns, bone health, heart health, and good mental health. A person...

**Answer**:

The context does not provide a definition or description of "Kidney Failure: Eat Right to Feel Right on Hemodialysis." It only lists the title within a collection of publications from the NIDDK Kidney Failure Series, indicating that such a resource exists but without further details. [medquad_ans_0000138-11] [medquad_ans_0000136-12]

**Supported citations**: `medquad_ans_0000138-11`, `medquad_ans_0000136-12`
**Unsupported citations**: _none_

### Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Scores**: context_relevance=1.000, groundedness=0.862, answer_relevance=0.556

**Top docs**:

1. `medquad_ans_0000677-1` (score=29.9785) - Mucolipidosis type IV is an inherited disorder characterized by delayed development and vision impairment that worsens over time. The severe form of the disorder is called typical mucolipidosis type IV, and the mild f...
2. `medquad_ans_0000677-2` (score=23.8018) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
3. `medquad_ans_0000677-3` (score=23.0108) - Mutations in the MCOLN1 gene cause mucolipidosis type IV. This gene provides instructions for making a protein called mucolipin-1. This protein is located in the membranes of lysosomes and endosomes, compartments with...
4. `medquad_ans_0000677-5` (score=21.2622) - These resources address the diagnosis or management of mucolipidosis type IV: - Gene Review: Gene Review: Mucolipidosis IV - Genetic Testing Registry: Ganglioside sialidase deficiency - MedlinePlus Encyclopedia: Gastr...
5. `medquad_ans_0000674-1` (score=20.2086) - Mucolipidosis II alpha/beta (also known as I-cell disease) is a progressively debilitating disorder that affects many parts of the body. Most affected individuals do not survive past early childhood. At birth, childre...

**Answer**:

Mucolipidosis type IV occurs in approximately 1 in 40,000 people [medquad_ans_0000677-2].

**Supported citations**: `medquad_ans_0000677-2`
**Unsupported citations**: _none_

### Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Scores**: context_relevance=1.000, groundedness=0.994, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=15.3944) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chi...
2. `medquad_ans_0002062-1` (score=14.5663) - What are the signs and symptoms of Ehlers-Danlos syndrome, spondylocheirodysplastic type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, spondylocheirodyspl...
3. `medquad_ans_0001913-1` (score=13.9413) - Dowling-Degos disease is a skin condition characterized by a lacy or net-like (reticulate) pattern of abnormally dark skin coloring (hyperpigmentation), particularly in the body's folds and creases. Other features may...
4. `medquad_ans_0000159-6` (score=13.9068) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
5. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams [medquad_ans_0000795-1__dup1].

**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

### Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Scores**: context_relevance=0.889, groundedness=0.817, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001045-1` (score=31.9858) - X-linked adrenoleukodystrophy is a genetic disorder that occurs primarily in males. It mainly affects the nervous system and the adrenal glands, which are small glands located on top of each kidney. In this disorder, ...
2. `medquad_ans_0001045-5` (score=28.3262) - These resources address the diagnosis or management of X-linked adrenoleukodystrophy: - Gene Review: Gene Review: X-Linked Adrenoleukodystrophy - Genetic Testing Registry: Adrenoleukodystrophy - Genomics Education Pro...
3. `medquad_ans_0001045-4` (score=26.6929) - X-linked adrenoleukodystrophy is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each...
4. `medquad_ans_0001051-3` (score=25.4115) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
5. `medquad_ans_0001045-3` (score=25.0155) - Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. The ABCD1 gene provides instructions for producing the adrenoleukodystrophy protein (ALDP), which is involved in transporting certain fat molecules call...

**Answer**:

The provided context does not specify the exact genetic changes related to X-linked adrenoleukodystrophy. It describes the disorder's effects on the nervous system and adrenal glands, as well as its types and symptoms, but does not detail the underlying genetic mutations or molecular mechanisms. [medquad_ans_0001045-1]

**Supported citations**: `medquad_ans_0001045-1`
**Unsupported citations**: _none_

### Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Scores**: context_relevance=0.750, groundedness=0.786, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000988-1` (score=32.2047) - Transthyretin amyloidosis is a slowly progressive condition characterized by the buildup of abnormal deposits of a protein called amyloid (amyloidosis) in the body's organs and tissues. These protein deposits most fre...
2. `medquad_ans_0000988-2` (score=27.8239) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of Europe...
3. `medquad_ans_0000988-5` (score=26.9666) - These resources address the diagnosis or management of transthyretin amyloidosis: - Boston University: Amyloid Treatment & Research Program - Gene Review: Gene Review: Familial Transthyretin Amyloidosis - Genetic Test...
4. `medquad_ans_0000988-3` (score=19.1747) - Mutations in the TTR gene cause transthyretin amyloidosis. The TTR gene provides instructions for producing a protein called transthyretin. Transthyretin transports vitamin A (retinol) and a hormone called thyroxine t...
5. `medquad_ans_0000136-3__dup2` (score=17.4951) - Primary amyloidosis and dialysis-related amyloidosis are the types of amyloidosis that can affect the kidneys. Primary Amyloidosis of the Kidneys The kidneys are the organs most commonly affected by primary amyloidosi...

**Answer**:

The provided context does not specify the number of people affected by transthyretin amyloidosis. [medquad_ans_0000988-1] describes the condition's symptoms, progression, and forms but does not include epidemiological data on prevalence or incidence. Therefore, the number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000988-1`
**Unsupported citations**: _none_

### Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Scores**: context_relevance=1.000, groundedness=0.826, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0001780-1` (score=62.6762) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the informatio...
2. `medquad_ans_0006016-1` (score=34.2189) - What are the signs and symptoms of Taurodontism, microdontia, and dens invaginatus? The Human Phenotype Ontology provides the following list of signs and symptoms for Taurodontism, microdontia, and dens invaginatus. I...
3. `medquad_ans_0005769-1` (score=28.7801) - What are the signs and symptoms of Spinocerebellar ataxia 20? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia 20. If the information is available, the table be...
4. `medquad_ans_0000108-2__dup1` (score=28.3357) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
5. `medquad_ans_0000549-1__dup1` (score=27.9662) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include those listed by the Human Phenotype Ontology (HPO), which compiles information on how often a sign or symptom occurs in individuals with this condition. The frequency of these symptoms is typically estimated as a percentage or presented as a fraction (e.g., 25/25), indicating the number of patients with the symptom out of the total number examined in a specific study. However, the specific symptoms themselves are not detailed in the provided context [medquad_ans_0001780-1].

**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

### Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Scores**: context_relevance=0.875, groundedness=0.746, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000782-1` (score=30.3790) - Periventricular heterotopia is a condition in which nerve cells (neurons) do not migrate properly during the early development of the fetal brain, from about the 6th week to the 24th week of pregnancy. Heterotopia mea...
2. `medquad_ans_0004818-2` (score=28.9794) - What are the signs and symptoms of periventricular nodular heterotopia? The condition is first noticed when seizures appear, often during the teenage years. The nodules around the ventricles are then typically discove...
3. `medquad_ans_0004818-1` (score=28.4307) - Periventricular heterotopia is a condition in which the nerve cells (neurons) do not migrate properly during the early development of the fetal brain from about the 6th week to the 24th week of pregnancy. Affected peo...
4. `medquad_ans_0000782-3` (score=27.1234) - Periventricular heterotopia is related to chromosome 5. Mutations in the ARFGEF2 and FLNA genes cause periventricular heterotopia. In most cases, periventricular heterotopia is caused by mutations in the FLNA gene. Th...
5. `medquad_ans_0000782-4` (score=24.7110) - Periventricular heterotopia can have different inheritance patterns. When this condition is caused by mutations in the FLNA gene, it is inherited in an X-linked dominant pattern. A condition is considered X-linked if ...

**Answer**:

The provided context does not specify the number of people affected by periventricular heterotopia. [medquad_ans_0000782-1] and [medquad_ans_0004818-2] describe the condition, its symptoms, and associated features but do not include epidemiological data or prevalence statistics. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000782-1`, `medquad_ans_0004818-2`
**Unsupported citations**: _none_

### Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Scores**: context_relevance=0.727, groundedness=0.145, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000007_2-6` (score=26.0332) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000009_2-8` (score=25.1102) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000032_1-8` (score=24.9566) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000006_4-7` (score=24.7930) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000019_1-10` (score=24.5801) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for Landau-Kleffner Syndrome. [medquad_ans_0000007_2-6, medquad_ans_0000009_2-8] discuss clinical trials for cancers such as perihilar bile duct cancer and osteosarcoma, including targeted therapies like kinase inhibitor therapy and monoclonal antibody therapy, but there is no information related to Landau-Kleffner Syndrome. Therefore, based on the given context, no research or clinical trials for Landau-Kleffner Syndrome can be identified.

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000007_2-6, medquad_ans_0000009_2-8`

### Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Scores**: context_relevance=0.800, groundedness=0.135, answer_relevance=0.440

**Top docs**:

1. `medquad_ans_0000602-3` (score=19.5988) - The various forms of limb-girdle muscular dystrophy are caused by mutations in many different genes. These genes provide instructions for making proteins that are involved in muscle maintenance and repair. Some of the...
2. `medquad_ans_0000395-2` (score=19.4209) - Fukuyama congenital muscular dystrophy is seen almost exclusively in Japan, where it is the second most common form of childhood muscular dystrophy (after Duchenne muscular dystrophy). Fukuyama congenital muscular dys...
3. `medquad_ans_0001935-1` (score=18.4407) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystroph...
4. `medquad_ans_0000576-2` (score=18.4275) - The prevalence of early-onset LAMA2-related muscular dystrophy is estimated at 1 in 30,000 individuals. This condition accounts for between 30 and 40 percent of total cases of congenital muscular dystrophy, although i...
5. `medquad_ans_0000747-2` (score=18.4136) - In Europe, the prevalence of oculopharyngeal muscular dystrophy is estimated to be 1 in 100,000 people. The autosomal dominant form of this condition is much more common in the French-Canadian population of the Canadi...

**Answer**:

The provided context does not define "Muscular Dystrophy" in general terms. It describes specific types of muscular dystrophy, such as limb-girdle muscular dystrophy, Fukuyama congenital muscular dystrophy, and Duchenne muscular dystrophy, including their genetic causes and inheritance patterns. However, a general definition of "Muscular Dystrophy" is not available in the given context.

Therefore, based on the provided information, it is not possible to define muscular dystrophy. [insufficient context]

**Supported citations**: _none_
**Unsupported citations**: `insufficient context`

### Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.770, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005020-1` (score=14.1185) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the tabl...
2. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
3. `medquad_ans_0006103-1` (score=13.4478) - What are the signs and symptoms of Tibia absent polydactyly arachnoid cyst? The Human Phenotype Ontology provides the following list of signs and symptoms for Tibia absent polydactyly arachnoid cyst. If the informatio...
4. `medquad_ans_0003952-1` (score=13.1180) - What are the signs and symptoms of Merlob Grunebaum Reisner syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Merlob Grunebaum Reisner syndrome. If the information is availab...
5. `medquad_ans_0005019-1` (score=13.0061) - What are the signs and symptoms of Preaxial polydactyly type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 1. If the information is available, the tabl...

**Answer**:

Polydactyly is a congenital condition characterized by the presence of extra fingers or toes. In the context of Preaxial polydactyly type 2, it specifically refers to the duplication of digits on the thumb side of the hand or foot, including features such as duplication of the thumb phalanx, triphalangeal thumb, and preaxial hand or foot polydactyly, which are observed in up to 90% of affected individuals [medquad_ans_0005020-1]. Additionally, postaxial hand or foot polydactyly and syndactyly (fusion of fingers or toes) may also be present [medquad_ans_0005020-1]. The condition is inherited in an autosomal dominant pattern [medquad_ans_0005020-1].

**Supported citations**: `medquad_ans_0005020-1`
**Unsupported citations**: _none_

### Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002519-1` (score=24.8727) - Hereditary geniospasm is a movement disorder that causes episodes of involuntary tremors of the chin and lower lip. The episodes may last anywhere from a few seconds to hours and may occur spontaneously or be brought ...
2. `medquad_ans_0002519-3` (score=23.2358) - How is hereditary geniospasm inherited? Hereditary geniospasm is inherited in an autosomal dominant manner. This means that having only one mutated copy of the causative gene in each body cell is sufficient to cause s...
3. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
4. `medquad_ans_0000549-1__dup1` (score=20.4489) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...
5. `medquad_ans_0002259-1` (score=20.2941) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

The symptoms of geniospasm include episodes of involuntary tremors of the chin and lower lip. These episodes may last from a few seconds to hours and can occur spontaneously or be triggered by stress. The episodes typically first appear in infancy or childhood and tend to decrease in frequency with age [medquad_ans_0002519-1].

**Supported citations**: `medquad_ans_0002519-1`
**Unsupported citations**: _none_

### Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=0.800, groundedness=0.975, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000512-3` (score=40.6853) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of devel...
2. `medquad_ans_0000512-2` (score=38.4706) - Idiopathic pulmonary fibrosis has an estimated prevalence of 13 to 20 per 100,000 people worldwide. About 100,000 people are affected in the United States, and 30,000 to 40,000 new cases are diagnosed each year. Famil...
3. `medquad_ans_0000512-1` (score=38.0606) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effect...
4. `medquad_ans_0000512-4` (score=32.5296) - Most cases of idiopathic pulmonary fibrosis are sporadic; they occur in people with no history of the disorder in their family. Familial pulmonary fibrosis appears to have an autosomal dominant pattern of inheritance....
5. `medquad_ans_0000568-1__dup1` (score=31.3137) - Summary : A lung transplant removes a person's diseased lung and replaces it with a healthy one. The healthy lung comes from a donor who has died. Some people get one lung during a transplant. Other people get two. Lu...

**Answer**:

Genetic changes related to idiopathic pulmonary fibrosis include mutations in the TERC and TERT genes, which have been found in about 15 percent of all cases of familial pulmonary fibrosis and a smaller percentage of cases of sporadic idiopathic pulmonary fibrosis. These genes provide instructions for making components of an enzyme called telomerase, which maintains telomeres at the ends of chromosomes. [medquad_ans_0000512-3]

**Supported citations**: `medquad_ans_0000512-3`
**Unsupported citations**: _none_

### Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.588, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0005008-1` (score=20.1337) - Potocki-Shaffer syndrome is a contiguous gene deletion syndrome associated with deletions in a specific region of chromosome 11 (11p11.2). The characteristic features of Potocki-Shaffer syndrome include openings in th...
2. `medquad_ans_0000813-3` (score=19.3263) - Potocki-Shaffer syndrome (also known as proximal 11p deletion syndrome) is caused by a deletion of genetic material from the short (p) arm of chromosome 11 at a position designated 11p11.2. The size of the deletion va...
3. `medquad_ans_0000813-2` (score=19.1028) - Potocki-Shaffer syndrome is a rare condition, although its prevalence is unknown. Fewer than 100 cases have been reported in the scientific literature.
4. `medquad_ans_0000813-1` (score=18.3826) - Potocki-Shaffer syndrome is a disorder that affects development of the bones, nerve cells in the brain, and other tissues. Most people with this condition have multiple noncancerous (benign) bone tumors called osteoch...
5. `medquad_ans_0000813-5` (score=16.4224) - These resources address the diagnosis or management of Potocki-Shaffer syndrome: - Genetic Testing Registry: Potocki-Shaffer syndrome These resources from MedlinePlus offer information about the diagnosis and manageme...

**Answer**:

The provided context does not mention whether Potocki-Shaffer syndrome is inherited. Therefore, based on the available information, it cannot be determined if the syndrome is inherited. [medquad_ans_0005008-1] [medquad_ans_0000813-3] [medquad_ans_0000813-2]

**Supported citations**: `medquad_ans_0005008-1`, `medquad_ans_0000813-3`, `medquad_ans_0000813-2`
**Unsupported citations**: _none_

### Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Scores**: context_relevance=1.000, groundedness=0.791, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000223-10` (score=14.9913) - You can take the following steps to help prevent your child from getting a UTI: - Teach your child not to hold in urine and to go to the bathroom whenever your child feels the urge. - Teach your child how to properly ...
2. `medquad_ans_0000010-5__dup1` (score=14.6868) - Your primary care doctor will diagnose asthma based on your medical and family histories, a physical exam, and test results. Your doctor also will figure out the severity of your asthmathat is, whether it's intermitte...
3. `medquad_ans_0000146-1__dup4` (score=14.3518) - Growth failure is a complication of CKD in which children do not grow as expected. When a child is below the third percentilemeaning 97 percent of children the same age and gender are tallerhe or she has growth failur...
4. `medquad_ans_0000925-1__dup1` (score=14.1910) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
5. `medquad_ans_0000286-1__dup2` (score=14.1507) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...

**Answer**:

No, there is no information about child care in the provided context. [medquad_ans_0000223-10] and [medquad_ans_0000010-5__dup1] discuss preventive measures for urinary tract infections in children and the diagnosis of asthma, respectively, but neither mentions child care.

**Supported citations**: `medquad_ans_0000223-10`, `medquad_ans_0000010-5__dup1`
**Unsupported citations**: _none_

### Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Scores**: context_relevance=0.900, groundedness=0.732, answer_relevance=0.540

**Top docs**:

1. `medquad_ans_0000046-1` (score=38.8746) - Alternating hemiplegia of childhood is a neurological condition characterized by recurrent episodes of temporary paralysis, often affecting one side of the body (hemiplegia). During some episodes, the paralysis altern...
2. `medquad_ans_0000046-4` (score=36.9354) - Alternating hemiplegia of childhood is considered an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of alternating hemiplegia of chi...
3. `medquad_ans_0000046-3` (score=33.6867) - Alternating hemiplegia of childhood is primarily caused by mutations in the ATP1A3 gene. Very rarely, a mutation in the ATP1A2 gene is involved in the condition. These genes provide instructions for making very simila...
4. `medquad_ans_0000046-2` (score=32.2805) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
5. `medquad_ans_0000016-2__dup1` (score=26.8816) - Drug therapy including verapamil may help to reduce the severity and duration of attacks of paralysis associated with the more serious form of alternating hemiplegia

**Answer**:

The provided context does not specify the number of people affected by alternating hemiplegia of childhood. [medquad_ans_0000046-1] describes the symptoms and characteristics of the condition, [medquad_ans_0000046-4] discusses its genetic inheritance, and [medquad_ans_0000046-3] identifies the genes involved, but none of these sources provide population prevalence or affected individual numbers.

Answer: Insufficient context.

**Supported citations**: `medquad_ans_0000046-1`, `medquad_ans_0000046-4`, `medquad_ans_0000046-3`
**Unsupported citations**: _none_

### Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Scores**: context_relevance=1.000, groundedness=0.944, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0003687-1` (score=26.0393) - Lupus is an autoimmune disease that can affect almost every organ in the body. Symptoms of lupus can range from very mild to life-threatening. There are three types of lupus; systemic lupus erythematosus, discoid lupu...
2. `medquad_ans_0000159-6` (score=22.4418) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
3. `medquad_ans_0000569-1__dup2` (score=22.3472) - If you have lupus, your immune system attacks healthy cells and tissues by mistake. This can damage your joints, skin, blood vessels and organs. There are many kinds of lupus. The most common type, systemic lupus eryt...
4. `medquad_ans_0000159-3__dup2` (score=20.9696) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time...
5. `medquad_ans_0000159-1__dup3` (score=20.8450) - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). SLE is an autoimmune diseasea disorder in which the bodys immune system attacks the bodys own cells and organs. Up to 60 pe...

**Answer**:

Symptoms of lupus can range from very mild to life-threatening and include joint pain or swelling, muscle pain, fever with no known cause, and a rash that doesn't go away [medquad_ans_0000569-1__dup2]. Additionally, other symptoms may involve damage to joints, skin, blood vessels, and organs [medquad_ans_0003687-1].

**Supported citations**: `medquad_ans_0000569-1__dup2`, `medquad_ans_0003687-1`
**Unsupported citations**: _none_

### Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.991, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0001398-1` (score=22.6932) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
2. `medquad_ans_0001397-1` (score=22.6932) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
3. `medquad_ans_0001395-1` (score=22.6932) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
4. `medquad_ans_0000194-1` (score=21.2255) - Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features of this disorder include a failure to gain weight and grow at the expected rate (failure to thrive), a...
5. `medquad_ans_0000194-5` (score=18.7313) - These resources address the diagnosis or management of Cockayne syndrome: - Gene Review: Gene Review: Cockayne Syndrome - Genetic Testing Registry: Cockayne syndrome - Genetic Testing Registry: Cockayne syndrome type ...

**Answer**:

Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newborn, microcephaly, and impaired nervous system development. Other symptoms may include hearing loss, tooth decay, and eye and bone abnormalities. Cockayne syndrome is caused by mutations in either the ERCC8 (CSA) or ERCC6 (CSB) genes and is inherited in an autosomal recessive pattern [medquad_ans_0001398-1].

**Supported citations**: `medquad_ans_0001398-1`
**Unsupported citations**: _none_

### Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Scores**: context_relevance=0.833, groundedness=0.969, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000029-1__dup5` (score=23.6665) - An Infection of the Gums and Surrounding Tissues Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. The two forms of gum disease are gingivitis, a mild form that is...
2. `medquad_ans_0000029-13` (score=17.0230) - There are a number of risk factors that can increase your chances of developing periodontal disease. - Smoking is one of the most significant risk factors associated with the development of gum disease and can even lo...
3. `medquad_ans_0000029-11` (score=16.9507) - If left untreated, gum disease can lead to tooth loss. Gum disease is the leading cause of tooth loss in older adults.
4. `medquad_ans_0000029-6__dup1` (score=16.5046) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead...
5. `medquad_ans_0000029-12` (score=16.4033) - In some studies, researchers have observed that people with periodontal disease (when compared to people without periodontal disease) were more likely to develop heart disease or have difficulty controlling their bloo...

**Answer**:

Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. The two forms of gum disease are gingivitis, a mild form that is reversible with good oral hygiene, and periodontitis, a more severe form that can damage the soft tissues and bone that support teeth. If left untreated, periodontitis can lead to tooth loss [medquad_ans_0000029-1__dup5]. In gingivitis, the gums become red, swollen, and can bleed easily, but this form does not include any loss of bone and tissue that hold teeth in place. When gingivitis is not treated, it can advance to periodontitis, where gums pull away from the teeth and form spaces (called "pockets") that become infected, leading to the breakdown of bone and connective tissue that support the teeth [medquad_ans_000002

**Supported citations**: `medquad_ans_0000029-1__dup5`
**Unsupported citations**: _none_

### Example 212 (`0000494-2`)

**Question**: How many people are affected by hyperkalemic periodic paralysis ?

**Scores**: context_relevance=1.000, groundedness=0.732, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000494-5` (score=37.8363) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis ...
2. `medquad_ans_0000494-1` (score=30.9681) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to...
3. `medquad_ans_0000126-1__dup1` (score=30.9488) - Familial periodic paralyses are a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. They are characterized by episodes in which the affect...
4. `medquad_ans_0000494-2` (score=29.2575) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
5. `medquad_ans_0003000-2` (score=27.4261) - What are the signs and symptoms of Hyperkalemic periodic paralysis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hyperkalemic periodic paralysis. If the information is available, ...

**Answer**:

The provided context does not specify the number of people affected by hyperkalemic periodic paralysis. [medquad_ans_0000494-1] describes the condition's symptoms, triggers, and mechanisms but does not include epidemiological data or prevalence statistics. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000494-1`
**Unsupported citations**: _none_

### Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Scores**: context_relevance=1.000, groundedness=0.953, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0005212-3` (score=21.5670) - Is Pyle disease inherited? Pyle disease in inherited in an autosomal recessive manner, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive conditio...
2. `medquad_ans_0005212-1` (score=19.9380) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end o...
3. `medquad_ans_0005212-2` (score=16.0646) - What are the signs and symptoms of Pyle disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyle disease. If the information is available, the table below includes how often the...
4. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0000108-2__dup1` (score=12.3473) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (relative constriction of the diaphysis and flaring of the metaphysis), widening of the ribs and clavicles (collarbones), platyspondyly (flattening of the bones of the spine), and cortical thinning. It is inherited in an autosomal recessive manner, meaning both copies of the gene in each cell have mutations. The parents of an individual with Pyle disease each carry one copy of the mutated gene and are typically asymptomatic, though some carriers may show minor skeletal changes [medquad_ans_0005212-1][medquad_ans_0005212-3]. Only about 30 cases have been reported in the literature, and cranial involvement is minimal, with some patients showing mild hyperostosis of the skull base and

**Supported citations**: `medquad_ans_0005212-1`, `medquad_ans_0005212-3`
**Unsupported citations**: _none_

### Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Scores**: context_relevance=0.600, groundedness=0.908, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000051-5__dup1` (score=24.7656) - The signs and symptoms of deep vein thrombosis (DVT) might be related to DVT itself or pulmonary embolism (PE). See your doctor right away if you have signs or symptoms of either condition. Both DVT and PE can cause s...
2. `medquad_ans_0000256-1__dup3` (score=22.4330) - Deep vein thrombosis, or DVT, is a blood clot that forms in a vein deep in the body. Most deep vein clots occur in the lower leg or thigh. If the vein swells, the condition is called thrombophlebitis. A deep vein thro...
3. `medquad_ans_0000051-1__dup2` (score=21.1711) - Espaol Deep vein thrombosis (throm-BO-sis), or DVT, is a blood clot that forms in a vein deep in the body. Blood clots occur when blood thickens and clumps together. Most deep vein blood clots occur in the lower leg o...
4. `medquad_ans_0000051-4__dup2` (score=20.7438) - The risk factors for deep vein thrombosis (DVT) include: A history of DVT. Conditions or factors that make your blood thicker or more likely to clot than normal. Some inherited blood disorders (such as factor V Leiden...
5. `medquad_ans_0000104-1__dup4` (score=20.0560) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent b...

**Answer**:

Deep vein thrombosis (DVT) is caused by blood thickening and clumping together, which leads to the formation of a blood clot in a deep vein. Most DVT clots occur in the lower leg or thigh. Risk factors include sitting still for a long time, certain medicines, and disorders that increase the risk of blood clots [medquad_ans_0000051-1\_\_dup2][medquad_ans_0000256-1__dup3].

**Supported citations**: `medquad_ans_0000051-1__dup2`, `medquad_ans_0000256-1__dup3`
**Unsupported citations**: _none_

### Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.736, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000178-3__dup2` (score=26.2682) - The prognosis for individuals with Leigh's disease is poor. Individuals who lack mitochondrial complex IV activity and those with pyruvate dehydrogenase deficiency tend to have the worst prognosis and die within a few...
2. `medquad_ans_0004079-1` (score=25.8866) - Complex II deficiency is a mitochondrial disease. Mitochondria are specialized compartments in cells that create more than 90% of the energy needed by the body. In mitochondrial diseases, the mitochondria don't work c...
3. `medquad_ans_0000661-1` (score=23.5320) - Mitochondrial complex III deficiency is a genetic condition that can affect several parts of the body, including the brain, kidneys, liver, heart, and the muscles used for movement (skeletal muscles). Signs and sympto...
4. `medquad_ans_0000661-2` (score=23.5016) - The prevalence of mitochondrial complex III deficiency is unknown, although the condition is thought to be rare.
5. `medquad_ans_0004079-3` (score=23.3003) - What causes mitochondrial complex II deficiency? Many genes must work together to ensure that the enzyme, complex II (succinate dehydrogenase), can perform its job normally in the body. Changes in the SDHA, SDHB, SDHC...

**Answer**:

The provided context does not contain specific information about Mitochondrial complex IV deficiency. While it mentions that individuals with Leigh's disease and those lacking mitochondrial complex IV activity have a poor prognosis and may die within a few years, no detailed description of Mitochondrial complex IV deficiency is given [medquad_ans_0000178-3__dup2]. Therefore, based on the available context, a complete answer cannot be provided.

**Supported citations**: `medquad_ans_0000178-3__dup2`
**Unsupported citations**: _none_

### Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Scores**: context_relevance=1.000, groundedness=0.516, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000078-1` (score=40.3701) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...
2. `medquad_ans_0000078-5` (score=39.5867) - These resources address the diagnosis or management of ataxia with oculomotor apraxia: - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type 1 - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type ...
3. `medquad_ans_0000078-2` (score=36.3794) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.
4. `medquad_ans_0000496-1__dup1` (score=32.8555) - What are the signs and symptoms of Ataxia with Oculomotor Apraxia Type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Ataxia with Oculomotor Apraxia Type 2. If the information is...
5. `medquad_ans_0000078-3` (score=32.3034) - Mutations in the APTX and SETX genes cause ataxia with oculomotor apraxia types 1 and 2, respectively. These genes provide instructions for making proteins that are involved in DNA repair. Mutations in the APTX or SET...

**Answer**:

Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement, with the hallmark being difficulty coordinating movements (ataxia), which is often the first symptom. Most affected individuals also have oculomotor apraxia, making it difficult to move their eyes side-to-side, requiring them to turn their head to see peripheral vision. There are multiple types of this condition, which are very similar but caused by mutations in different genes. The two most common types (types 1 and 2) share features including involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, ataxia begins around age 4, while in type 2, it begins around age 15. Chorea and myoclonus tend to disappear gradually in type 1 but persist throughout life in type 2. Individuals with

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Scores**: context_relevance=1.000, groundedness=0.543, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0000221-5` (score=70.2633) - These resources address the diagnosis or management of LAMM syndrome: - Gene Review: Gene Review: Congenital Deafness with Labyrinthine Aplasia, Microtia, and Microdontia - Genetic Testing Registry: Deafness with laby...
2. `medquad_ans_0000221-1` (score=59.1963) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that fo...
3. `medquad_ans_0001744-1` (score=42.1121) - What are the signs and symptoms of Deafness with labyrinthine aplasia microtia and microdontia (LAMM)? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness with labyrinthine apla...
4. `medquad_ans_0001496-1` (score=26.1551) - Congenital deafness with vitiligo and achalasia is a syndrome characterized by deafness present from birth (congenital), associated with short stature, vitiligo, muscle wasting and achalasia (swallowing difficulties)....
5. `medquad_ans_0001527-2` (score=23.5570) - What are the signs and symptoms associated with congenital laryngeal paralysis? The following online resources provide information on the signs and symptoms of congenital laryngeal paralysis: National Institute on Dea...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects the development of the ears and teeth. In people with this condition, the structures that form the inner ear are usually completely absent (labyrinthine aplasia). Rarely, affected individuals have some underdeveloped inner ear structures in one or both ears. The abnormalities of the inner ear cause a form of hearing loss called sensorineural deafness that is present from birth (congenital). Because the inner ear is important for balance as well as hearing, development of motor skills, such as sitting and crawling, may be delayed in affected infants. In addition, people with LAMM syndrome often have abnormally small outer ears (microtia) with narrow ear canals. They can also have unusually small, widely spaced teeth (microdontia) [medquad_ans_000

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000064-4` (score=23.7912) - Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases of Apert syndrome result from new mutations in ...
2. `medquad_ans_0000420-3__dup1` (score=19.9647) - How is Apert syndrome diagnosed? Is genetic testing needed to confirm the diagnosis? Apert syndrome and the other conditions associated with FGFR-related craniosynostosis were clinically defined long before the molecu...
3. `medquad_ans_0000064-1` (score=18.9361) - Apert syndrome is a genetic disorder characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the head and fa...
4. `medquad_ans_0000420-2__dup1` (score=16.6851) - What are the signs and symptoms of Apert syndrome? Apert syndrome is characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affe...
5. `medquad_ans_0000420-1__dup1` (score=16.6797) - Apert syndrome is a disorder mainly characterized by craniosynostosis (premature fusion of skull bones, causing abnormalities in the shape of the head and face) and syndactyly (fusion or webbing or fingers and/or toes...

**Answer**:

Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. However, almost all cases result from new mutations and occur in people with no family history of the disorder. Individuals with Apert syndrome can pass the condition to their offspring [medquad_ans_0000064-4].

**Supported citations**: `medquad_ans_0000064-4`
**Unsupported citations**: _none_

### Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Scores**: context_relevance=1.000, groundedness=0.882, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000341-3__dup2` (score=26.3738) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact ...
2. `medquad_ans_0000341-1__dup3` (score=25.0934) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella bu...
3. `medquad_ans_0005229-1` (score=21.6867) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organi...
4. `medquad_ans_0000341-4__dup2` (score=19.6736) - In the United States, Q fever outbreaks have resulted mainly from occupational exposure involving veterinarians, meat processing plant workers, sheep and dairy workers, livestock farmers, and researchers at facilities...
5. `medquad_ans_0001304-1` (score=18.6712) - Chromosomes are the structures found in every cell of the body that contain our DNA, the instructions that tell our body what to do. Humans have 23 pairs of chromosomes, which means that each human cell contains 46 ch...

**Answer**:

The provided context does not define what Q fever is. It discusses diagnosis, management, reporting, testing procedures, and contact information related to Q fever but does not include a definition of the disease itself. [medquad_ans_0000341-3__dup2]

**Supported citations**: `medquad_ans_0000341-3__dup2`
**Unsupported citations**: _none_

### Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0000038-5__dup1` (score=31.0969) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
2. `medquad_ans_0000038-9` (score=26.0126) - At this time, most types of childhood interstitial lung disease (chILD) can't be prevented. People who have a family history of inherited (genetic) interstitial lung disease may want to consider genetic counseling. A ...
3. `medquad_ans_0000038-1__dup2` (score=25.6214) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chroni...
4. `medquad_ans_0001152-2` (score=24.7886) - How might chILD be treated? There is no single treatment for interstitial lung diseases in children. Different forms of chILD require different treatments and support depending on the condition. The goals of treatment...
5. `medquad_ans_0000522-1__dup1` (score=24.7070) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...

**Answer**:

Children who have a family history of interstitial lung disease or childhood interstitial lung disease (chILD), those with an inherited surfactant disorder or a family history of such disorders, children with problems involving aspiration (inhaling food, liquid, or vomit into the lungs), children with immune system disorders, children exposed to environmental lung irritants such as molds and chemicals, children with systemic or autoimmune diseases (e.g., collagen vascular disease or inflammatory bowel disease), children undergoing cancer treatments like radiation or chemotherapy, and children who have had a bone marrow or lung transplant are at increased risk for childhood interstitial lung disease [medquad_ans_0000038-5__dup1]. Additionally, certain types of chILD are more common in infants and young children, while others can occur in children of any age [medquad_ans_0000038-5__dup1].

**Supported citations**: `medquad_ans_0000038-5__dup1`
**Unsupported citations**: _none_

### Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Scores**: context_relevance=0.833, groundedness=0.927, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003334-1` (score=38.0585) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in lat...
2. `medquad_ans_0000087-1__dup1` (score=20.3976) - Vasculitis is an inflammation of blood vessels, which includes the veins, arteries, and capillaries. Inflammation occurs with infection or is thought to be due to a faulty immune system response. It also can be caused...
3. `medquad_ans_0000404-1__dup1` (score=18.4039) - Giant cell arteritis is a disorder that causes inflammation of your arteries, usually in the scalp, neck, and arms. It narrows the arteries, which keeps blood from flowing well. Giant cell arteritis often occurs with ...
4. `medquad_ans_0006001-3` (score=16.4642) - How might Takayasu arteritis be treated? The treatment of Takayasu arteritis is focused on controlling both the inflammatory process and hypertension . Treatment options might include: corticosteroids, medications tha...
5. `medquad_ans_0006001-1` (score=15.8859) - Takayasu arteritis is a condition that causes inflammation of the main blood vessel that carries blood from the heart to the rest of the body (aorta) and its associated branched blood vessels. As a result of the infla...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels, specifically affecting the temporal arteries located at the lower sides of the skull, directly underneath the temple. It is generally diagnosed in late childhood or early adulthood and often presents without signs or symptoms, with affected individuals sometimes having only a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically with no family history. It is usually treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

### Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.519, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001082-2` (score=33.9931) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyry...
2. `medquad_ans_0001082-1` (score=31.6057) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as...
3. `medquad_ans_0001082-5` (score=27.9270) - These resources address the diagnosis or management of 2-methylbutyryl-CoA dehydrogenase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of 2-methylbutyryl-CoA dehydrogenase These resources from...
4. `medquad_ans_0001082-3` (score=26.6728) - Mutations in the ACADSB gene cause 2-methylbutyryl-CoA dehydrogenase deficiency. The ACADSB gene provides instructions for making an enzyme called 2-methylbutyryl-CoA dehydrogenase that helps process the amino acid is...
5. `medquad_ans_0006577-1` (score=26.0888) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particul...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder, more common among Hmong populations in southeast Asia and Hmong Americans, with an incidence of 1 in 250 to 1 in 500 people of Hmong ancestry. It is a type of organic acid disorder in which the body is unable to properly process proteins due to inadequate levels of the enzyme 2-methylbutyryl-CoA dehydrogenase, which is necessary for processing the amino acid isoleucine. This leads to an abnormal buildup of organic acids in the blood, urine, and tissues, which can be toxic and cause serious health problems. Symptoms vary widely, ranging from severe and life-threatening (such as difficulty breathing, seizures, coma, poor growth, vision problems, learning disabilities, muscle weakness, and motor skill delays) to mild or absent. Initial symptoms often include poor feeding, lethargy,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Scores**: context_relevance=1.000, groundedness=0.532, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000947-1` (score=37.7775) - Spondyloepiphyseal dysplasia congenita is an inherited bone growth disorder that results in short stature (dwarfism), skeletal abnormalities, and problems with vision and hearing. This condition affects the bones of t...
2. `medquad_ans_0000947-5` (score=33.5150) - These resources address the diagnosis or management of spondyloepiphyseal dysplasia congenita: - Genetic Testing Registry: Spondyloepiphyseal dysplasia congenita - MedlinePlus Encyclopedia: Clubfoot - MedlinePlus Ency...
3. `medquad_ans_0005841-2` (score=30.0981) - What are the signs and symptoms of Spondyloepiphyseal dysplasia Maroteaux type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia Maroteaux type. If the in...
4. `medquad_ans_0001064-1` (score=29.2519) - X-linked spondyloepiphyseal dysplasia tarda is a condition that impairs bone growth and occurs almost exclusively in males. The name of the condition indicates that it affects the bones of the spine (spondylo-) and th...
5. `medquad_ans_0000947-3` (score=29.1708) - Spondyloepiphyseal dysplasia congenita is one of a spectrum of skeletal disorders caused by mutations in the COL2A1 gene. This gene provides instructions for making a protein that forms type II collagen. This type of ...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include short stature from birth, a very short trunk and neck, shortened limbs, and average-sized hands and feet. Adult height ranges from 3 feet to just over 4 feet. Abnormal curvature of the spine (kyphoscoliosis and lordosis) becomes more severe during childhood. Instability of the spinal bones (vertebrae) in the neck may increase the risk of spinal cord damage. Other skeletal features include flattened vertebrae (platyspondyly), an abnormality of the hip joint causing the upper leg bones to turn inward (coxa vara), a foot deformity called clubfoot, and a broad, barrel-shaped chest. Abnormal development of the chest can cause problems with breathing. Arthritis and decreased joint mobility often develop early in life. Mild changes in facial features are present, such as flattened cheekbones near the nose. Some infants

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Scores**: context_relevance=1.000, groundedness=0.543, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000853-5` (score=24.3907) - These resources address the diagnosis or management of pyridoxine-dependent epilepsy: - Gene Review: Gene Review: Pyridoxine-Dependent Epilepsy - Genetic Testing Registry: Pyridoxine-dependent epilepsy - MedlinePlus E...
2. `medquad_ans_0005219-3` (score=21.9185) - What causes pyridoxine-dependent epilepsy? Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each ...
3. `medquad_ans_0005219-4` (score=21.5699) - How might pyridoxine-dependent epilepsy be treated? Anticonvulsant drugs, which are usually given to control seizures, are ineffective in people with pyridoxine-dependent epilepsy. Instead, people with this type of se...
4. `medquad_ans_0000853-2` (score=21.1729) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
5. `medquad_ans_0000853-1` (score=20.9868) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepti...

**Answer**:

Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The ALDH7A1 gene provides instructions for making an enzyme called α-aminoadipic semialdehyde (α-AASA) dehydrogenase, also known as antiquitin. This enzyme is involved in the breakdown of the protein building block (amino acid) lysine in the brain. When antiquitin is deficient, a molecule that interferes with vitamin B6 function builds up in various tissues. It is unclear how a lack of pyridoxine causes the seizures that are characteristic of this condition. Some individuals with pyridoxine-dependent epilepsy do not have identified mutations in the ALDH7A1 gene. In these cases, the cause of the condition is unknown [medquad_ans_00052

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000004_6-1` (score=28.7479) - Key Points - Childhood Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two types of childhood Hodgkin lymphoma. - Epstein-Barr virus infection increases the risk o...
2. `medquad_ans_0000004_6-2` (score=28.4150) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
3. `medquad_ans_0000004_6-8` (score=27.7763) - Key Points - There are different types of treatment for children with Hodgkin lymphoma. - Children with Hodgkin lymphoma should have their treatment planned by a team of health care providers who are experts in treati...
4. `medquad_ans_0000004_1-2` (score=25.4764) - Age, gender, and Epstein-Barr infection can affect the risk of adult Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will g...
5. `medquad_ans_0000004_1-1` (score=24.8902) - Key Points - Adult Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two main types of Hodgkin lymphoma: classical and nodular lymphocyte-predominant. - Age, gender,...

**Answer**:

Individuals at risk for childhood Hodgkin lymphoma include those with Epstein-Barr virus infection. [medquad_ans_0000004_6-1]

**Supported citations**: `medquad_ans_0000004_6-1`
**Unsupported citations**: _none_

### Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Scores**: context_relevance=1.000, groundedness=0.904, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0001795-2` (score=29.0957) - What are the signs and symptoms of Dermatitis herpetiformis ? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatitis herpetiformis . If the information is available, the table be...
2. `medquad_ans_0000088-4__dup2` (score=24.4189) - A person may experience digestive signs and symptoms, or symptoms in other parts of the body. Digestive signs and symptoms are more common in children and can include - abdominal bloating - chronic diarrhea - constipa...
3. `medquad_ans_0001795-3` (score=23.6044) - How might dermatitis herpetiformis be treated? The antibiotic dapsone is extremely effective in treating this condition. Symptomatic improvement may occur in as little as several hours after the first dose. However, d...
4. `medquad_ans_0001795-1` (score=22.1556) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow...
5. `medquad_ans_0000095-2__dup3` (score=21.7392) - Dermatitis herpetiformis is caused by the deposit of immunoglobulin A (IgA) in the skin, which triggers further immunologic reactions resulting in lesion formation. DH is an external manifestation of an abnormal immun...

**Answer**:

Dermatitis herpetiformis is an itchy, blistering skin rash that is associated with autoimmune conditions and malabsorption. It is commonly seen in individuals with celiac disease and is characterized by signs such as abnormal blistering of the skin, pruritus (itching), and autoimmunity. The condition affects approximately 90% of patients, with other associated features including malabsorption, microcytic anemia, and eczema (in 50% of cases). It may also present with symptoms such as bone pain, edema, and lichenification in smaller percentages of patients [medquad_ans_0001795-2]. Additionally, dermatitis herpetiformis can be linked to digestive signs and symptoms, including chronic diarrhea, abdominal bloating, and stomach pain, particularly in children [medquad_ans_0000088-4__dup2].

**Supported citations**: `medquad_ans_0001795-2`, `medquad_ans_0000088-4__dup2`
**Unsupported citations**: _none_

### Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Scores**: context_relevance=1.000, groundedness=0.812, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0004287-2` (score=30.7017) - What are the signs and symptoms of Myotonia congenita autosomal dominant? The Human Phenotype Ontology provides the following list of signs and symptoms for Myotonia congenita autosomal dominant. If the information is...
2. `medquad_ans_0000040-4__dup1` (score=29.7464) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to myotonia congenita and also supports additional research through grants to major research institutions across the countr...
3. `medquad_ans_0000709-1` (score=28.2562) - Myotonia congenita is a disorder that affects muscles used for movement (skeletal muscles). Beginning in childhood, people with this condition experience bouts of sustained muscle tensing (myotonia) that prevent muscl...
4. `medquad_ans_0000709-5` (score=27.2674) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Regi...
5. `medquad_ans_0000771-1` (score=25.5156) - Paramyotonia congenita is a disorder that affects muscles used for movement (skeletal muscles). Beginning in infancy or early childhood, people with this condition experience bouts of sustained muscle tensing (myotoni...

**Answer**:

The provided context does not mention any specific treatments for myotonia congenita. While it discusses research into the molecular mechanisms of the disorder and the development of animal models for testing potential therapies, no treatments are explicitly described [medquad_ans_0000040-4__dup1]. Therefore, based on the available information, treatments for myotonia congenita cannot be determined.

**Supported citations**: `medquad_ans_0000040-4__dup1`
**Unsupported citations**: _none_

### Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Scores**: context_relevance=1.000, groundedness=0.978, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000133-13` (score=21.3119) - - Wilson disease is a genetic disease that prevents the body from removing extra copper. - Normally, the liver filters extra copper and releases it into bile. In Wilson disease, the liver does not filter copper correc...
2. `medquad_ans_0000133-9` (score=21.1525) - A health care provider may use several tests and exams to diagnose Wilson disease, including the following: - medical and family history - physical exam - blood tests - urine tests - liver biopsy - imaging tests Healt...
3. `medquad_ans_0000133-10` (score=20.9838) - A health care provider will treat Wilson disease with a lifelong effort to reduce and control the amount of copper in the body. Treatment may include - medications - changes in eating, diet, and nutrition - a liver tr...
4. `medquad_ans_0006449-5` (score=20.3231) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lif...
5. `medquad_ans_0006449-1` (score=19.9830) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver,...

**Answer**:

To manage Wilson disease, a health care provider will implement a lifelong effort to reduce and control the amount of copper in the body. This may include medications, changes in eating, diet, and nutrition, and in some cases, a liver transplant. People with Wilson disease should reduce their dietary copper intake by avoiding foods high in copper, such as shellfish, liver, mushrooms, nuts, and chocolate [medquad_ans_0000133-13].

**Supported citations**: `medquad_ans_0000133-13`
**Unsupported citations**: _none_

### Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.942, answer_relevance=0.644

**Top docs**:

1. `medquad_ans_0000592-3` (score=34.9904) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are re...
2. `medquad_ans_0000592-1` (score=27.6530) - Lennox-Gastaut syndrome is a form of severe epilepsy that begins in childhood. It is characterized by multiple types of seizures and intellectual disability. People with Lennox-Gastaut syndrome begin having frequent s...
3. `medquad_ans_0000592-4` (score=27.3348) - Most cases of Lennox-Gastaut syndrome are sporadic, which means they occur in people with no history of the disorder in their family. However, 3 to 30 percent of people with this condition have a family history of som...
4. `medquad_ans_0000667-3` (score=24.1068) - The causes of Moebius syndrome are unknown, although the condition probably results from a combination of environmental and genetic factors. Researchers are working to identify and describe specific genes related to t...
5. `medquad_ans_0003520-3` (score=23.9825) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. Th...

**Answer**:

Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component [medquad_ans_0000592-3]. In about one-third of cases, the cause is unknown and described as cryptogenic, with no history of epilepsy, neurological problems, or delayed development [medquad_ans_0000592-1]. The syndrome can be associated with genetic disorders such as tuberous sclerosis complex or brain malformations like cortical dysplasia [medquad_ans_0000592-3]. However, no specific genes have been identified as directly causing Lennox-Gastaut syndrome.

**Supported citations**: `medquad_ans_0000592-3`, `medquad_ans_0000592-1`
**Unsupported citations**: _none_

### Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.969, answer_relevance=0.480

**Top docs**:

1. `medquad_ans_0000066-3` (score=37.9205) - Mutations in the GATM gene cause arginine:glycine amidinotransferase deficiency. The GATM gene provides instructions for making the enzyme arginine:glycine amidinotransferase. This enzyme participates in the two-step ...
2. `medquad_ans_0000066-1` (score=35.7390) - Arginine:glycine amidinotransferase deficiency is an inherited disorder that primarily affects the brain. People with this disorder have mild to moderate intellectual disability and delayed speech development. Some af...
3. `medquad_ans_0000066-5` (score=35.1818) - These resources address the diagnosis or management of arginine:glycine amidinotransferase deficiency: - Gene Review: Gene Review: Creatine Deficiency Syndromes - Genetic Testing Registry: Arginine:glycine amidinotran...
4. `medquad_ans_0000066-2` (score=32.1100) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
5. `medquad_ans_0003472-1` (score=17.7100) - L-arginine:glycine amidinotransferase (AGAT) deficiency is a rare condition that primarily affects the brain. People with AGAT deficiency generally have mild to moderate intellectual disability. Other signs and sympto...

**Answer**:

The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families. [medquad_ans_0000066-2]

**Supported citations**: `medquad_ans_0000066-2`
**Unsupported citations**: _none_

### Example 231 (`0002399-2`)

**Question**: What are the symptoms of Focal dermal hypoplasia ?

**Scores**: context_relevance=0.875, groundedness=0.544, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000379-1` (score=40.4470) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. About 90 percent of affected individuals are female. Males usually have milder signs and symptoms than females. ...
2. `medquad_ans_0002399-3` (score=39.9512) - How is this condition inherited? Focal dermal hypoplasia is caused by mutations in the PORCN gene and is inherited in an X-linked dominant manner. Many cases of focal dermal hypoplasia result from a new mutation and o...
3. `medquad_ans_0002399-1` (score=39.2654) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. The skin abnormalities are present from birth and can include streaks of very thin skin (dermal hypoplasia), cut...
4. `medquad_ans_0000379-4` (score=38.8095) - Focal dermal hypoplasia is inherited in an X-linked dominant pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In females (who have two X chromos...
5. `medquad_ans_0000379-5` (score=37.8642) - These resources address the diagnosis or management of focal dermal hypoplasia: - Gene Review: Gene Review: Focal Dermal Hypoplasia - Genetic Testing Registry: Focal dermal hypoplasia - MedlinePlus Encyclopedia: Ectod...

**Answer**:

Symptoms of focal dermal hypoplasia include skin abnormalities present from birth such as streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), small clusters of veins on the surface of the skin (telangiectases), and streaks of slightly darker or lighter skin. These skin changes may cause pain, itching, irritation, or lead to skin infections. Wart-like growths called papillomas develop with age and may form around the nostrils, lips, anus, and female genitalia, and can also be present in the throat (esophagus or larynx), causing problems with swallowing, breathing, or sleeping. Affected individuals may have small, ridged fingernails and toenails, sparse or brittle scalp hair or absent hair. Hand and foot abnormalities include missing fingers or toes

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Scores**: context_relevance=0.909, groundedness=0.945, answer_relevance=0.618

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=27.5215) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
2. `medquad_ans_0004252-2` (score=23.3774) - What causes myelodysplastic/myeloproliferative disease? In most cases, the cause of myelodysplastic/myeloproliferative disease is unknown, and there is limited information regarding potential causes. No specific genet...
3. `medquad_ans_0000014_3-3` (score=22.8287) - Key Points - Avoiding risk factors and increasing protective factors may help prevent cancer. - The following risk factors increase the risk of endometrial cancer: - Endometrial hyperplasia - Estrogen - Tamoxifen - Ob...
4. `medquad_ans_0001853-3` (score=22.7308) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0000096-2__dup2` (score=22.7296) - Acute diarrhea is usually caused by a bacterial, viral, or parasitic infection. Chronic diarrhea is usually related to a functional disorder such as irritable bowel syndrome or an intestinal disease such as Crohns dis...

**Answer**:

People at risk for trichinellosis include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating even small amounts of undercooked meat during preparation or cooking can put individuals at risk. Individuals who consume undercooked wild game, especially bear meat, are at increased risk. The risk from commercially raised and properly prepared pork is very low, but the risk remains for those who eat raw or undercooked wild game [medquad_ans_0000418-2__dup1].

**Supported citations**: `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

### Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.927, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000108-3__dup1` (score=28.9234) - As children with Dravet syndrome get older, their decline in cognitive function stabilizes, and in many, it improves slightly. However, most teenagers with Dravet syndrome are dependent on caregivers. The degree of co...
2. `medquad_ans_0000108-1__dup2` (score=28.0151) - Dravet syndrome, also called severe myoclonic epilepsy of infancy (SMEI), is a severe form of epilepsy. It appears during the first year of life with frequent febrile seizures fever-related seizures that, by definitio...
3. `medquad_ans_0000108-4__dup1` (score=25.9372) - The NINDS conducts and supports a broad program of basic and clinical research on all types of epilepsy, including Dravet syndrome. Study of the genetic defects responsible for Dravet syndrome and related disorders is...
4. `medquad_ans_0001924-1` (score=24.5247) - Dravet syndrome is a severe form of epilepsy. The condition appears during the first year of life as frequent fever-related (febrile) seizures. As the condition progresses, other types of seizures typically occur, inc...
5. `medquad_ans_0001924-2` (score=24.1725) - What are the signs and symptoms of Dravet syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Dravet syndrome. If the information is available, the table below includes how oft...

**Answer**:

The symptoms of Dravet syndrome include frequent fever-related (febrile) seizures that begin during the first year of life, later progressing to other types of seizures such as myoclonus and status epilepticus. Children often experience poor development of language and motor skills, hyperactivity, and difficulty relating to others. Intellectual development begins to deteriorate around age 2, and affected individuals may have a lack of coordination. In some cases, a family history of epilepsy or febrile seizures is present in 15 to 25 percent of cases. In 30 to 80 percent of cases, the condition is caused by defects in the SCN1A gene, which is essential for proper brain cell function [medquad_ans_0000108-1\_\_dup2][medquad_ans_0001924-1].

**Supported citations**: `medquad_ans_0000108-1__dup2`, `medquad_ans_0001924-1`
**Unsupported citations**: _none_

### Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Scores**: context_relevance=0.889, groundedness=0.978, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0000089-4__dup1` (score=28.8312) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system, and to use that knowledge to reduce the burden of neurological di...
2. `medquad_ans_0001663-3` (score=28.4901) - What causes Creutzfeldt-Jakob disease? Some researchers believe an unusual 'slow virus' or another organism causes Creutzfeldt-Jakob disease (CJD). However, they have never been able to isolate a virus or other organi...
3. `medquad_ans_0002316-5` (score=27.7776) - Is genetic testing available for fatal familial insomnia? Yes, genetic testing is available for PRNP, the gene known to cause fatal familial insomnia (FFI). Carrier testing for at-risk relatives and prenatal testing a...
4. `medquad_ans_0000011-13` (score=27.7456) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...
5. `medquad_ans_0000826-4` (score=27.5978) - Familial forms of prion disease are inherited in an autosomal dominant pattern, which means one copy of the altered PRNP gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherit...

**Answer**:

About 5 to 10 percent of all CJD cases are inherited. These cases arise from a mutation, or change, in the gene that controls formation of the normal prion protein [medquad_ans_0001663-3].

**Supported citations**: `medquad_ans_0001663-3`
**Unsupported citations**: _none_

### Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.984, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0006430-1` (score=42.8370) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information ...
2. `medquad_ans_0001034-1` (score=27.9424) - Weyers acrofacial dysostosis is a disorder that affects the development of the teeth, nails, and bones. Dental abnormalities can include small, peg-shaped teeth; fewer teeth than normal (hypodontia); and one front too...
3. `medquad_ans_0000314-1` (score=26.4085) - Ellis-van Creveld syndrome is an inherited disorder of bone growth that results in very short stature (dwarfism). People with this condition have particularly short forearms and lower legs and a narrow chest with shor...
4. `medquad_ans_0006429-1` (score=24.5579) - What are the signs and symptoms of Weyers acrofacial dysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers acrofacial dysostosis. If the information is available, the ta...
5. `medquad_ans_0001034-3` (score=23.6955) - Most cases of Weyers acrofacial dysostosis result from mutations in the EVC2 gene. A mutation in a similar gene, EVC, has been found in at least one person with the characteristic features of the disorder. Little is k...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, hypoplasia of the radius, oligodactyly (hands), proximal placement of thumb, proximal radial head dislocation, cleft palate, cleft upper lip, high palate, hydronephrosis, hypotelorism, long face, mesomelia, narrow face, and high palate [medquad_ans_0006430-1].

**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

### Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Scores**: context_relevance=1.000, groundedness=0.920, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0005029-1` (score=37.7848) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the ta...
2. `medquad_ans_0002060-1` (score=26.9836) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the in...
3. `medquad_ans_0006422-1` (score=25.6468) - Werner's syndrome is a disease chiefly characterized by premature aging and cancer predisposition. Development is typically normal until the end of the first decade; the first sign is the lack of a growth spurt during...
4. `medquad_ans_0004524-1` (score=24.1087) - What are the signs and symptoms of Okamoto syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Okamoto syndrome. If the information is available, the table below includes how o...
5. `medquad_ans_0000622-1` (score=23.3276) - Mandibuloacral dysplasia is a condition that causes a variety of abnormalities involving bone development, skin coloring (pigmentation), and fat distribution. People with this condition may grow slowly after birth. Mo...

**Answer**:

The symptoms of Premature aging Okamoto type include abnormality of blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which collects information from sources such as Orphanet, with frequencies often listed as percentages or fractions representing the proportion of patients exhibiting each symptom [medquad_ans_0005029-1].

**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

### Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.756, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0004919-1` (score=22.9055) - Platelet storage pool deficiency refers to a group of conditions that are caused by problems with the platelet granules. Platelet granules are tiny storage sacs found within the platelets which release various substan...
2. `medquad_ans_0000979-1` (score=22.6474) - Thrombocytopenia-absent radius (TAR) syndrome is characterized by the absence of a bone called the radius in each forearm. Affected individuals also have a shortage (deficiency) of blood cells involved in clotting (pl...
3. `medquad_ans_0001079-3` (score=20.5523) - Most people with a 1q21.1 microdeletion are missing a sequence of about 1.35 million DNA building blocks (base pairs), also written as 1.35 megabases (Mb), in the q21.1 region of chromosome 1. However, the exact size ...
4. `medquad_ans_0004919-3` (score=20.2409) - What causes platelet storage pool deficiency? Platelet storage pool deficiency refers to a group of conditions that are caused by problems with the platelet granules. Platelet granules are tiny storage sacs found with...
5. `medquad_ans_0005249-1` (score=15.0765) - What are the signs and symptoms of Radius absent anogenital anomalies? The Human Phenotype Ontology provides the following list of signs and symptoms for Radius absent anogenital anomalies. If the information is avail...

**Answer**:

The provided context does not specify the number of people affected by thrombocytopenia-absent radius (TAR) syndrome. [medquad_ans_0000979-1] describes the features and characteristics of the syndrome but does not include population prevalence or affected individual counts. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000979-1`
**Unsupported citations**: _none_

### Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Scores**: context_relevance=0.727, groundedness=0.764, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000007_2-6` (score=26.0332) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000009_2-8` (score=25.1102) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000032_1-8` (score=24.9566) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000006_4-7` (score=24.7930) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000019_1-10` (score=24.5801) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for Tay-Sachs Disease. [medquad_ans_0000007_2-6], [medquad_ans_0000009_2-8] discuss clinical trials for cancers such as perihilar bile duct cancer and osteosarcoma, not Tay-Sachs Disease. Therefore, no information is available in the context regarding research or clinical trials for Tay-Sachs Disease.

**Supported citations**: `medquad_ans_0000007_2-6`, `medquad_ans_0000009_2-8`
**Unsupported citations**: _none_

### Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000008-1` (score=20.1885) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achro...
2. `medquad_ans_0000008-2` (score=19.5200) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on o...
3. `medquad_ans_0000008-3` (score=17.8993) - Achromatopsia results from changes in one of several genes: CNGA3, CNGB3, GNAT2, PDE6C, or PDE6H. A particular CNGB3 gene mutation underlies the condition in Pingelapese islanders. Achromatopsia is a disorder of the r...
4. `medquad_ans_0000050-1__dup1` (score=17.8745) - What are the signs and symptoms of Achromatopsia 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Achromatopsia 3. If the information is available, the table below includes how oft...
5. `medquad_ans_0000008-5` (score=14.8516) - These resources address the diagnosis or management of achromatopsia: - Gene Review: Gene Review: Achromatopsia - Genetic Testing Registry: Achromatopsia - MedlinePlus Encyclopedia: Color Vision Test These resources f...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. The condition also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals may have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia results from mutations in genes such as CNGA3, CNGB3, GNAT2, PDE6C, or PDE6H, which interfere with phototransduction in the retina,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Scores**: context_relevance=1.000, groundedness=0.896, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000544-1__dup1` (score=32.8404) - Knee replacement is surgery for people with severe knee damage. Knee replacement can relieve pain and allow you to be more active. Your doctor may recommend it if you have knee pain and medicine and other treatments a...
2. `medquad_ans_0000039-2__dup2` (score=28.7800) - There are many different types and designs of artificial knees. Most consist of three components: - the femoral component, which is the part that attaches to the thigh bone - the tibial component, the part that attach...
3. `medquad_ans_0000039-11` (score=25.9731) - Treatments your doctor will likely recommend before knee replacement include - exercises to strengthen the muscles around the knee and improve flexibility - weight loss, if needed, to reduce the load the knee must bea...
4. `medquad_ans_0000039-13` (score=25.7954) - Knee replacement may be either total or partial/unicompartmental. In total knee replacement, as the name suggests, the entire knee joint is replaced. You will likely need a total knee replacement if you have damage to...
5. `medquad_ans_0004155-5` (score=25.2375) - What treatment is available for mucopolysaccharidosis I (MPS I)? The two main treatments for MPS I are enzyme replacement therapy (ERT) and bone marrow transplant. Both of these treatments work by replacing the missin...

**Answer**:

The treatments for knee replacement include total knee replacement and partial (unicompartmental) knee replacement. In a total knee replacement, the surgeon removes damaged cartilage and bone from the knee joint and replaces them with a man-made surface of metal and plastic. In a partial knee replacement, only the damaged part of the knee joint is replaced [medquad_ans_0000544-1__dup1]. Additionally, joint components may be attached to the bone using cement or through biologic fixation, where bone grows into a porous surface to secure the implant [medquad_ans_0000039-2__dup2]. Minimally invasive surgery, which uses smaller incisions (3 to 5 inches or even smaller), may also be performed to reduce pain and shorten recovery time [medquad_ans_0000039-2__dup2]. These procedures are typically recommended when knee pain persists despite

**Supported citations**: `medquad_ans_0000544-1__dup1`, `medquad_ans_0000039-2__dup2`
**Unsupported citations**: _none_

### Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.857, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002259-1` (score=24.1024) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0000090-4__dup2` (score=21.8445) - People at greatest risk for metabolic syndrome have these underlying causes: Abdominal obesity (a large waistline) An inactive lifestyle Insulin resistance Some people are at risk for metabolic syndrome because they t...
3. `medquad_ans_0000090-1__dup2` (score=21.6830) - Metabolicsyndrome is the name for a group of risk factors that raises your risk for heart disease and other health problems, such as diabetes and stroke. The term "metabolic" refers to the biochemical processes involv...
4. `medquad_ans_0000277-2__dup2` (score=21.2503) - There is no cure for Zellweger syndrome, nor is there a standard course of treatment. Since the metabolic and neurological abnormalities that cause the symptoms of Zellweger syndrome are caused during fetal developmen...
5. `medquad_ans_0001621-3` (score=21.1945) - What causes cramp-fasciculation syndrome? In many cases, the exact underlying cause of cramp-fasciculation syndrome (CFS) is unknown (idiopathic). In general, it is thought to be related to abnormal excitability (over...

**Answer**:

The provided context does not list the specific symptoms of metabolic syndrome. It mentions risk factors such as abdominal obesity, inactive lifestyle, insulin resistance, certain medications, racial and ethnic groups (e.g., Mexican Americans), personal or family history of diabetes, and polycystic ovarian syndrome, but does not detail the symptoms of metabolic syndrome itself [medquad_ans_0000090-4__dup2]. Therefore, based on the given information, the symptoms of metabolic syndrome cannot be identified.

**Supported citations**: `medquad_ans_0000090-4__dup2`
**Unsupported citations**: _none_

### Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Scores**: context_relevance=0.833, groundedness=0.701, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
2. `medquad_ans_0000549-1__dup1` (score=20.4489) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...
3. `medquad_ans_0002259-1` (score=20.2941) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
4. `medquad_ans_0001795-2` (score=20.2537) - What are the signs and symptoms of Dermatitis herpetiformis ? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatitis herpetiformis . If the information is available, the table be...
5. `medquad_ans_0003571-1` (score=20.1052) - What are the signs and symptoms of Leukoencephalopathy - dystonia - motor neuropathy ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoencephalopathy - dystonia - motor neuropat...

**Answer**:

The provided context does not contain any information about the symptoms of Pyknoachondrogenesis. [medquad_ans_0000108-2__dup1] and [medquad_ans_0000549-1__dup1] discuss symptoms of Acute alcohol sensitivity and Autosomal dominant caf au lait spots, respectively, but neither mentions Pyknoachondrogenesis. Therefore, the symptoms of Pyknoachondrogenesis cannot be determined from the given context.

**Supported citations**: `medquad_ans_0000108-2__dup1`, `medquad_ans_0000549-1__dup1`
**Unsupported citations**: _none_

### Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.970, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000647-1` (score=35.1861) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the sub...
2. `medquad_ans_0000647-3` (score=34.3996) - Most individuals with metachromatic leukodystrophy have mutations in the ARSA gene, which provides instructions for making the enzyme arylsulfatase A. This enzyme is located in cellular structures called lysosomes, wh...
3. `medquad_ans_0003965-3` (score=32.1629) - How is metachromatic leukodystrophy inherited? Metachromatic leukodystrophy is inherited in an autosomal recessive manner. This means that both copies of the disease-causing gene in each cell must have a mutation for ...
4. `medquad_ans_0003966-1` (score=30.4112) - What are the signs and symptoms of Metachromatic leukodystrophy due to saposin B deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy due to sapos...
5. `medquad_ans_0003566-1` (score=29.6539) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to def...

**Answer**:

The symptoms of metachromatic leukodystrophy include progressive deterioration of intellectual functions and motor skills, such as the ability to walk, loss of sensation in the extremities (peripheral neuropathy), incontinence, seizures, paralysis, an inability to speak, blindness, and hearing loss. Eventually, individuals lose awareness of their surroundings and become unresponsive. In the late infantile form, children lose speech, become weak, and develop gait disturbance; muscle tone first decreases and then increases to rigidity. In the juvenile form, initial signs may include behavioral problems and difficulty with schoolwork, with slower progression and survival of about 20 years after diagnosis. In the adult form, symptoms often begin during teenage years or later and may include behavioral problems such as alcoholism, drug abuse, or difficulties at school or work, along with psychiatric symptoms like delusions or hallucinations [medquad_ans_0000647-1].

**Supported citations**: `medquad_ans_0000647-1`
**Unsupported citations**: _none_

### Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.748, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000186-1` (score=24.4604) - Chronic granulomatous disease is a disorder that causes the immune system to malfunction, resulting in a form of immunodeficiency. Immunodeficiencies are conditions in which the immune system is not able to protect th...
2. `medquad_ans_0000186-4` (score=24.0308) - When chronic granulomatous disease is caused by mutations in the CYBB gene, the condition is inherited in an X-linked recessive pattern. The CYBB gene is located on the X chromosome, which is one of the two sex chromo...
3. `medquad_ans_0000186-5` (score=23.0542) - These resources address the diagnosis or management of chronic granulomatous disease: - American Academy of Allergy, Asthma, and Immunology - Gene Review: Gene Review: Chronic Granulomatous Disease - Genetic Testing R...
4. `medquad_ans_0000186-2` (score=22.0053) - Chronic granulomatous disease is estimated to occur in 1 in 200,000 to 250,000 people worldwide.
5. `medquad_ans_0004341-3` (score=20.5079) - Is nephrocalcinosis inherited? Nephrocalcinosis may be caused by a large variety of things, including underlying disorders, certain medications and supplements, and infections. Nephrocalcinosis itself is not inherited...

**Answer**:

The provided context does not state whether chronic granulomatous disease is inherited. Therefore, based on the given information, it cannot be determined if the condition is inherited. [medquad_ans_0000186-1]

**Supported citations**: `medquad_ans_0000186-1`
**Unsupported citations**: _none_

### Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.879, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000061-2__dup1` (score=25.6429) - There is no cure for central cord syndrome although some people recover near-normal function. There is no standard course of treatment, although drug therapy, surgery, and rest are often part of the program. Magnetic ...
2. `medquad_ans_0000061-1__dup1` (score=25.3686) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to a...
3. `medquad_ans_0000006_4-1` (score=24.3121) - Key Points - Central nervous system atypical teratoid/rhabdoid tumor is a disease in which malignant (cancer) cells form in the tissues of the brain. - Certain genetic changes may increase the risk of atypical teratoi...
4. `medquad_ans_0003573-2` (score=23.4745) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...
5. `medquad_ans_0000061-3__dup1` (score=22.6044) - The prognosis for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict he likeliho...

**Answer**:

There is no cure for central cord syndrome, although some people recover near-normal function. The overall amount and type of functional loss depend on the severity of nerve damage. While there is no standard treatment, drug therapy, surgery, and rest are often part of the management plan. Recent reports suggest that earlier surgery may improve recovery chances, and surgery can be beneficial in individuals with persistent spinal cord compression and ongoing neurological deterioration [medquad_ans_0000061-2__dup1]. However, the provided context does not specify a definitive prognosis or long-term outlook beyond these points. Therefore, the outlook is variable and depends on the severity of the injury and the timeliness and effectiveness of treatment. [medquad_ans_0000061-1__dup1]

**Supported citations**: `medquad_ans_0000061-2__dup1`, `medquad_ans_0000061-1__dup1`
**Unsupported citations**: _none_

### Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.939, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000094-13` (score=23.6836) - - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no ...
2. `medquad_ans_0000261-1` (score=23.2383) - Cyclic vomiting syndrome is a disorder that causes recurrent episodes of nausea, vomiting, and tiredness (lethargy). This condition is diagnosed most often in young children, but it can affect people of any age. The e...
3. `medquad_ans_0000094-4__dup3` (score=22.3873) - Cyclic vomiting syndrome is more common in children than adults, although reports of the syndrome in adults have increased in recent years.1 Usually, children are about 5 years old when diagnosed with cyclic vomiting ...
4. `medquad_ans_0000261-3` (score=21.9758) - Although the causes of cyclic vomiting syndrome have yet to be determined, researchers have proposed several factors that may contribute to the disorder. These factors include changes in brain function, hormonal abnor...
5. `medquad_ans_0000094-1__dup6` (score=20.1163) - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no ap...

**Answer**:

The cause of cyclic vomiting syndrome is unknown [medquad_ans_0000094-13].

**Supported citations**: `medquad_ans_0000094-13`
**Unsupported citations**: _none_

### Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Scores**: context_relevance=1.000, groundedness=0.753, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000408-5` (score=18.9387) - These resources address the diagnosis or management of Gillespie syndrome: - Eunice Kennedy Shriver National Institute of Child Health and Human Development: How Do Health Care Providers Diagnose Intellectual and Deve...
2. `medquad_ans_0000875-1__dup1` (score=18.1599) - Summary : Being a teenager is hard. You're under stress to be liked, do well in school, get along with your family, and make big decisions. You can't avoid most of these pressures, and worrying about them is normal. B...
3. `medquad_ans_0000005-19` (score=17.8113) - Most insurance plans, including health maintenance organizations (HMOs), will cover treatment for anxiety disorders. Check with your insurance company and find out. If you dont have insurance, the Health and Human Ser...
4. `medquad_ans_0000219-7` (score=17.3351) - To find the cause of your ED, your doctor may - take your medical and sexual history - ask you questions about your mental health - give you a physical exam - test your blood - give you a nighttime erection test - per...
5. `medquad_ans_0000965-5` (score=15.9164) - These resources address the diagnosis or management of SYNGAP1-related intellectual disability: - Eunice Kennedy Shriver National Institute of Child Health and Human Development: What Are Treatments for Intellectual a...

**Answer**:

The provided context does not contain specific information about child mental health. While there is mention of mental health problems in adolescents and general mental health resources, no direct information about child mental health is available in the given context. [medquad_ans_0000875-1__dup1] discusses teenage mental health pressures and warning signs, which may overlap with adolescent mental health, but does not specifically address child mental health. Therefore, based on the provided sources, there is insufficient information to answer the question about child mental health.

**Supported citations**: `medquad_ans_0000875-1__dup1`
**Unsupported citations**: _none_

### Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000246-5` (score=29.7485) - These resources address the diagnosis or management of Cowden syndrome: - Gene Review: Gene Review: PTEN Hamartoma Tumor Syndrome (PHTS) - Genetic Testing Registry: Cowden syndrome - Genetic Testing Registry: Cowden s...
2. `medquad_ans_0000246-1` (score=29.6276) - Cowden syndrome is a disorder characterized by multiple noncancerous, tumor-like growths called hamartomas and an increased risk of developing certain cancers. Almost everyone with Cowden syndrome develops hamartomas....
3. `medquad_ans_0001618-3` (score=28.8967) - What causes Cowden syndrome? Most cases of Cowden syndrome are caused by changes (mutations) in the PTEN gene. PTEN is a tumor suppressor gene which means that it encodes a protein that helps keep cells from growing a...
4. `medquad_ans_0000246-3` (score=28.0733) - Changes involving at least four genes, PTEN, SDHB, SDHD, and KLLN, have been identified in people with Cowden syndrome or Cowden-like syndrome. Most cases of Cowden syndrome and a small percentage of cases of Cowden-l...
5. `medquad_ans_0001618-5` (score=27.9251) - How is Cowden syndrome diagnosed? A diagnosis of Cowden syndrome is based on the presence of characteristic signs and symptoms. Genetic testing for a change (mutation) in the PTEN gene can then be ordered to confirm t...

**Answer**:

The symptoms of Cowden syndrome include multiple noncancerous, tumor-like growths called hamartomas, which are most commonly found on the skin and mucous membranes (such as the lining of the mouth and nose), and can also occur in the intestine and other parts of the body. These hamartomas typically become apparent by a person's late twenties. Other symptoms include an increased risk of developing certain cancers, particularly breast, thyroid, and endometrial cancers, which often occur at younger ages (in the thirties or forties). Additional signs and symptoms may include an enlarged head (macrocephaly) and a rare, noncancerous brain tumor called Lhermitte-Duclos disease. A small percentage of affected individuals have delayed development or intellectual disability [medquad_ans_0000246-1].

**Supported citations**: `medquad_ans_0000246-1`
**Unsupported citations**: _none_

### Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Scores**: context_relevance=0.889, groundedness=0.784, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000395-1` (score=44.1048) - Fukuyama congenital muscular dystrophy is an inherited condition that predominantly affects the muscles, brain, and eyes. Congenital muscular dystrophies are a group of genetic conditions that cause muscle weakness an...
2. `medquad_ans_0000395-5` (score=40.8899) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
3. `medquad_ans_0000395-2` (score=39.6315) - Fukuyama congenital muscular dystrophy is seen almost exclusively in Japan, where it is the second most common form of childhood muscular dystrophy (after Duchenne muscular dystrophy). Fukuyama congenital muscular dys...
4. `medquad_ans_0000395-3` (score=34.4714) - Fukuyama congenital muscular dystrophy is caused by mutations in the FKTN gene. This gene provides instructions for making a protein called fukutin. Although the exact function of fukutin is unclear, researchers predi...
5. `medquad_ans_0002466-1` (score=31.3477) - What are the signs and symptoms of Fukuyama type muscular dystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Fukuyama type muscular dystrophy. If the information is available...

**Answer**:

The provided context does not specify any treatments for Fukuyama congenital muscular dystrophy. It describes the signs, symptoms, and complications of the disorder but does not detail therapeutic interventions or management strategies. While resources for diagnosis and management are listed, such as genetic counseling, palliative care, drug therapy, and rehabilitation, the specific treatments for Fukuyama congenital muscular dystrophy are not outlined in the context [medquad_ans_0000395-5]. Therefore, based on the given information, a definitive answer about treatments cannot be provided.

**Supported citations**: `medquad_ans_0000395-5`
**Unsupported citations**: _none_

### Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000004_6-1` (score=34.6057) - Key Points - Childhood Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two types of childhood Hodgkin lymphoma. - Epstein-Barr virus infection increases the risk o...
2. `medquad_ans_0000004_6-8` (score=29.6196) - Key Points - There are different types of treatment for children with Hodgkin lymphoma. - Children with Hodgkin lymphoma should have their treatment planned by a team of health care providers who are experts in treati...
3. `medquad_ans_0000004_1-1` (score=29.4370) - Key Points - Adult Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two main types of Hodgkin lymphoma: classical and nodular lymphocyte-predominant. - Age, gender,...
4. `medquad_ans_0000004_6-2` (score=28.3487) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
5. `medquad_ans_0000004_7-1` (score=27.6588) - Key Points - Childhood non-Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - The main types of lymphoma are Hodgkin lymphoma and non-Hodgkin lymphoma. - There are three major ...

**Answer**:

Tests that examine the lymph system are used to detect (find) and diagnose childhood Hodgkin lymphoma [medquad_ans_0000004_6-1].

**Supported citations**: `medquad_ans_0000004_6-1`
**Unsupported citations**: _none_

### Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.847, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0003361-1` (score=21.9826) - KBG syndrome is a rare condition characterized mainly by skeletal abnormalities, distinctive facial features, and intellectual disability. Specific signs and symptoms may include delayed bone age; abnormalities of the...
2. `medquad_ans_0000556-1` (score=20.6194) - KBG syndrome is a rare disorder that affects several body systems. "KBG" represents the surname initials of the first families diagnosed with the disorder. Common signs and symptoms in individuals with this condition ...
3. `medquad_ans_0000556-3` (score=19.5571) - KBG syndrome is caused by mutations in the ANKRD11 gene. The protein produced from this gene enables other proteins to interact with each other and helps control gene activity. The ANKRD11 protein is found in nerve ce...
4. `medquad_ans_0000556-2` (score=18.1324) - KBG syndrome is a rare disorder that has been reported in around 60 individuals. For unknown reasons, males are affected more often than females. Doctors think the disorder is underdiagnosed because the signs and symp...
5. `medquad_ans_0000556-5` (score=16.9311) - These resources address the diagnosis or management of KBG syndrome: - Genetic Testing Registry: KBG syndrome These resources from MedlinePlus offer information about the diagnosis and management of various health con...

**Answer**:

Yes, KBG syndrome can be inherited in an autosomal dominant manner in some cases, and it can also occur sporadically. Inheritance is possible when a mutation in the ANKRD11 gene is passed from a parent to a child, but in other cases, the genetic cause is unclear and the condition occurs without a family history [medquad_ans_0003361-1]. The ANKRD11 gene mutation is responsible for the syndrome in some individuals, and the protein it produces plays a role in brain development and neuronal function [medquad_ans_0000556-3].

**Supported citations**: `medquad_ans_0003361-1`, `medquad_ans_0000556-3`
**Unsupported citations**: _none_

### Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Scores**: context_relevance=0.833, groundedness=0.488, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000029-1` (score=21.0337) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. Howeve...
2. `medquad_ans_0000029-2` (score=18.5886) - Aicardi-Goutieres syndrome is a rare disorder. Its exact prevalence is unknown.
3. `medquad_ans_0000197-1__dup1` (score=17.1983) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000195-1__dup1` (score=17.1983) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
5. `medquad_ans_0000194-1__dup1` (score=17.1983) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...

**Answer**:

Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns do not show signs or symptoms at birth, but about 20 percent are born with features such as enlarged liver and spleen (hepatosplenomegaly), elevated liver enzymes, decreased blood platelets (thrombocytopenia), and abnormal neurological responses. This combination is typically associated with an immune response to congenital viral infection, although no actual infection is found, leading to the syndrome being referred to as a "mimic of congenital infection." Within the first year of life, most individuals experience severe brain dysfunction (encephalopathy), characterized by irritability, poor feeding, intermittent fevers (sterile pyrexias), seizures, developmental regression, loss of acquired skills, slowed brain and skull growth (microcephaly), and inflammation detectable in the cerebrospinal fluid

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Scores**: context_relevance=0.917, groundedness=0.862, answer_relevance=0.783

**Top docs**:

1. `medquad_ans_0000026_2-7` (score=37.5285) - Key Points - There are different types of treatment for patients with transitional cell cancer of the renal pelvis and ureter. - One type of standard treatment is used: - Surgery - New types of treatment are being tes...
2. `medquad_ans_0000026_1-7` (score=34.3804) - Key Points - There are different types of treatment for patients with renal cell cancer. - Five types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Biologic therapy - Targeted therapy ...
3. `medquad_ans_0000026_2-8` (score=32.5885) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000032_1-8` (score=31.3001) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000007_2-6` (score=30.9513) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context discusses clinical trials for transitional cell cancer of the renal pelvis and ureter, not for renal cell cancer. Therefore, there is no information available in the context about research or clinical trials specifically for renal cell cancer. [medquad_ans_0000026_2-7]

**Supported citations**: `medquad_ans_0000026_2-7`
**Unsupported citations**: _none_

### Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Scores**: context_relevance=0.857, groundedness=0.940, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000190-9` (score=34.5481) - - Primary biliary cirrhosis is a chronic disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. - When chronic inflammation damages the bile ducts, bile and toxi...
2. `medquad_ans_0000190-3__dup2` (score=29.3280) - The first and most common symptoms of primary biliary cirrhosis are - fatigue, or feeling tired - itching skin, and darkened skin in itching areas due to scratching - dry eyes and mouth Some people may have jaundice, ...
3. `medquad_ans_0000190-5__dup1` (score=28.6281) - A health care provider may use the following tests to diagnose primary biliary cirrhosis: - a medical and family history - a physical exam - blood tests - imaging tests - a liver biopsy A health care provider usually ...
4. `medquad_ans_0000190-2__dup2` (score=27.6367) - The causes of primary biliary cirrhosis are unknown. Most research suggests it is an autoimmune disease. The immune system protects people from infection by identifying and destroying bacteria, viruses, and other pote...
5. `medquad_ans_0000190-6` (score=24.9440) - Treatment for primary biliary cirrhosis depends on how early a health care provider diagnoses the disease and whether complications are present. In the early stages of primary biliary cirrhosis, treatment can slow the...

**Answer**:

For primary biliary cirrhosis, health care providers prescribe ursodiol (Actigall, Urso) to treat the condition. Early treatment with this medication reduces the likelihood of needing a liver transplant and improves survival. A health care provider may consider a liver transplant when cirrhosis leads to liver failure or treatment for complications is ineffective [medquad_ans_0000190-9]. Diagnosis is typically based on a medical and family history, physical exam, blood tests, imaging tests, and sometimes a liver biopsy, with a diagnosis often confirmed by two out of three criteria: elevated liver enzymes, presence of anti-mitochondrial antibodies (AMA), or signs of the disease on a liver biopsy [medquad_ans_0000190-5__dup1]. Routine blood tests showing abnormal liver enzyme levels may also lead to suspicion of the disease in asymptomatic individuals [medquad_ans_000

**Supported citations**: `medquad_ans_0000190-9`, `medquad_ans_0000190-5__dup1`
**Unsupported citations**: _none_

### Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Scores**: context_relevance=1.000, groundedness=0.900, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=14.4059) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People ...
2. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
3. `medquad_ans_0000916-1__dup2` (score=13.7677) - Summary : A tsunami is a series of huge ocean waves created by an underwater disturbance. Causes include earthquakes, landslides, volcanic eruptions, or meteorites--chunks of rock from space that strike the surface of...
4. `medquad_ans_0003663-2` (score=12.1111) - What are the signs and symptoms of Long QT syndrome? Signs and symptoms of the arrhythmias experienced by people with long QT syndrome includes unexplained fainting, seizures, drowning or near drowning, and sudden car...
5. `medquad_ans_0000108-2__dup1` (score=11.5977) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

Drowning is a condition that occurs when a person inhales water, leading to excessive water in the lungs, which can result in respiratory failure and death. It can happen quickly and silently, even in as little as an inch or two of water. Babies can drown in a sink or bathtub, and preschoolers are most likely to drown in a swimming pool. People with seizure disorders are also at increased risk in the water. Drowning is also noted as the most common cause of death related to a tsunami [medquad_ans_0000299-1\_\_dup2][medquad_ans_0000916-1__dup2]. Additionally, unexplained fainting, seizures, and sudden cardiac arrest or death can be signs of arrhythmias associated with long QT syndrome, which may lead to drowning or near drowning [medquad_ans_0003663-2].

**Supported citations**: `medquad_ans_0000299-1__dup2`, `medquad_ans_0000916-1__dup2`, `medquad_ans_0003663-2`
**Unsupported citations**: _none_

### Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Scores**: context_relevance=1.000, groundedness=0.239, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002058-3` (score=32.2111) - What causes Ehlers-Danlos syndrome, kyphoscoliosis type? Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is caused by changes (mutations) in the PLOD1 gene. This gene encodes an enzyme that helps process molecules w...
2. `medquad_ans_0002058-5` (score=31.7765) - How is Ehlers-Danlos syndrome, kyphoscoliosis type diagnosed? A diagnosis of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is typically based on the presence of characteristic signs and symptoms. The following tes...
3. `medquad_ans_0000313-1` (score=29.5180) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and sympt...
4. `medquad_ans_0002058-6` (score=28.3232) - How might Ehlers-Danlos syndrome, kyphoscoliosis type be treated? The treatment of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is focused on preventing serious complications and relieving associated signs and sy...
5. `medquad_ans_0002058-4` (score=27.1859) - Is Ehlers-Danlos Syndrome, kyphoscoliotic type inherited? Ehlers-Danlos syndrome, kyphoscoliosis type is inherited in an autosomal recessive manner. This means that to be affected, a person must have a mutation in bot...

**Answer**:

Ehlers-Danlos syndrome, kyphoscoliosis type is a disorder of the connective tissues that affects the skin, bones, blood vessels, and other organs and tissues. It is caused by mutations in the PLOD1 gene, which encodes an enzyme essential for collagen formation and stability. These mutations result in reduced levels of functional enzyme, disrupting collagen networks and leading to weakened connective tissues. This condition is characterized by kyphoscoliosis (abnormal curvature of the spine) and other signs and symptoms related to connective tissue fragility. Diagnosis is typically based on characteristic signs and symptoms, and may be confirmed through urine tests, skin biopsy, or genetic testing for mutations in the PLOD1 gene [medquad_ans_0002058-3, medquad_ans_00058-5, medquad_ans_0000313-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0002058-3, medquad_ans_00058-5, medquad_ans_0000313-1`

### Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001860-2` (score=32.6737) - What are the signs and symptoms of Dihydropteridine reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropteridine reductase deficiency. If the information is...
2. `medquad_ans_0000974-5` (score=32.0759) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Ge...
3. `medquad_ans_0001860-1` (score=31.9777) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...
4. `medquad_ans_0001098-1` (score=28.5262) - 5-alpha reductase deficiency is a condition that affects male sexual development before birth and during puberty. People with this condition are genetically male, with one X and one Y chromosome in each cell, and they...
5. `medquad_ans_0000898-1` (score=27.9728) - Sepiapterin reductase deficiency is a condition characterized by movement problems, most often a pattern of involuntary, sustained muscle contractions known as dystonia. Other movement problems can include muscle stif...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include cognitive impairment, microcephaly, cerebral calcification, choreoathetosis, dysphagia, dystonia, episodic fever, excessive salivation, hyperphenylalaninemia, hypertonia, infantile onset, intellectual disability, irritability, muscular hypotonia, myoclonus, progressive neurologic deterioration, seizures, tremor, and variable expressivity [medquad_ans_0001860-2].

**Supported citations**: `medquad_ans_0001860-2`
**Unsupported citations**: _none_

### Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Scores**: context_relevance=1.000, groundedness=0.732, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000669-1` (score=18.7277) - Monilethrix is a condition that affects hair growth. Its most characteristic feature is that individual strands of hair have a beaded appearance like the beads of a necklace. The name monilethrix comes from the Latin ...
2. `medquad_ans_0000669-3` (score=14.8330) - Monilethrix is caused by mutations in one of several genes. Mutations in the KRT81 gene, the KRT83 gene, the KRT86 gene, or the DSG4 gene account for most cases of monilethrix. These genes provide instructions for mak...
3. `medquad_ans_0000669-2` (score=14.5563) - The prevalence of monilethrix is unknown.
4. `medquad_ans_0000108-2__dup1` (score=12.2811) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
5. `medquad_ans_0000549-1__dup1` (score=12.0581) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...

**Answer**:

The provided context does not specify the number of people affected by monilethrix. [medquad_ans_0000669-1] and [medquad_ans_0000669-3] describe the characteristics, genetic causes, and clinical features of monilethrix but do not include epidemiological data on its prevalence or affected population size. Therefore, the number of people affected by monilethrix cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000669-1`, `medquad_ans_0000669-3`
**Unsupported citations**: _none_

### Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Scores**: context_relevance=1.000, groundedness=0.515, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0002733-1` (score=22.6855) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...
2. `medquad_ans_0002733-2` (score=17.8519) - What are the signs and symptoms of Hartnup disease? The signs and symptoms of Hartnup disease may vary and include the following: Skin findings: sensitivity to sunlight Neurologic symptoms: ataxia, spasticity, headach...
3. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
4. `medquad_ans_0000108-2__dup1` (score=12.3473) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
5. `medquad_ans_0000549-1__dup1` (score=12.1314) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system, making it a type of aminoaciduria. It is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. Most people with Hartnup disease are asymptomatic, but those who do show symptoms typically experience onset between the ages of 3 and 9, although it may occasionally present in adulthood. The signs and symptoms include skin photosensitivity, neurologic findings (such as ataxia, spasticity, headaches, and hypotonia), psychiatric symptoms (like anxiety, emotional instability, and mood changes), and ocular findings (such as double vision, nystagmus, strabismus, and photophobia). Symptoms may be triggered by sunlight exposure, fever, drugs, or emotional or physical stress and often last 1–4 weeks before spontaneous remission

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Scores**: context_relevance=0.833, groundedness=0.760, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000428-5` (score=40.5308) - These resources address the diagnosis or management of glycogen storage disease type VII: - Genetic Testing Registry: Glycogen storage disease, type VII - The Swedish Information Centre for Rare Diseases These resourc...
2. `medquad_ans_0002615-4` (score=37.3439) - How is glycogen storage disease type 13 diagnosed? Glycogen storage disease type 13 is diagnosed by taking a sample of muscle tissue (muscle biopsy) to determine if there is enough of the chemical enolase working in t...
3. `medquad_ans_0002615-3` (score=35.2135) - What causes glycogen storage disease type 13? Glycogen storage disease type 13 (GSD13) is caused by changes (mutations) in the ENO3 gene. Glycogen is a substance that is stored in muscle tissue and is used as an impor...
4. `medquad_ans_0000424-5` (score=34.0902) - These resources address the diagnosis or management of glycogen storage disease type IV: - Gene Review: Gene Review: Glycogen Storage Disease Type IV - Genetic Testing Registry: Glycogen storage disease, type IV - Med...
5. `medquad_ans_0002620-3` (score=34.0207) - What causes glycogen storage disease type 4? Glycogen storage disease type 4 (GSD 4) is caused by mutations in the GBE1 gene. The GBE1 gene normally provides instructions for making the glycogen branching enzyme. This...

**Answer**:

The provided context does not contain information about the genetic changes related to glycogen storage disease type VII. [medquad_ans_0000428-5] mentions resources for diagnosis or management of glycogen storage disease type VII, including genetic testing, but does not specify the genetic changes associated with it. Therefore, based on the given context, the genetic changes related to glycogen storage disease type VII cannot be determined.

**Supported citations**: `medquad_ans_0000428-5`
**Unsupported citations**: _none_

### Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.880, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000977-1` (score=44.0314) - Thiamine-responsive megaloblastic anemia syndrome is a rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Megaloblastic anemia occurs when a person has a low numb...
2. `medquad_ans_0000977-5` (score=39.0318) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megal...
3. `medquad_ans_0006061-1` (score=35.6775) - Thiamine-responsive megaloblastic anemia syndrome is a very rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Affected individuals begin to show symptoms of this...
4. `medquad_ans_0000977-3` (score=34.1050) - Mutations in the SLC19A2 gene cause thiamine-responsive megaloblastic anemia syndrome. This gene provides instructions for making a protein called thiamine transporter 1, which transports thiamine into cells. Thiamine...
5. `medquad_ans_0006061-2` (score=33.3976) - What are the signs and symptoms of Thiamine responsive megaloblastic anemia syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thiamine responsive megaloblastic anemia syndrom...

**Answer**:

The treatments for thiamine-responsive megaloblastic anemia syndrome include high doses of vitamin B1 (thiamine), which can treat the anemia and, in some cases, reduce the amount of insulin needed for diabetes management. Thiamine treatment may also help reduce the need for insulin in individuals with diabetes associated with this syndrome [medquad_ans_0000977-1]. Additionally, individuals with diabetes typically require insulin to manage their condition [medquad_ans_0000977-1]. For other complications such as optic atrophy or cardiovascular problems, specific treatments are not detailed in the provided context. Management resources include genetic counseling, drug therapy, and diagnostic testing, as outlined in MedlinePlus resources [medquad_ans_0000977-5].

**Supported citations**: `medquad_ans_0000977-1`, `medquad_ans_0000977-5`
**Unsupported citations**: _none_

### Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.718, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000804-3` (score=23.2017) - Mutations in the PKD1, PKD2, and PKHD1 genes cause polycystic kidney disease. Mutations in either the PKD1 or PKD2 gene can cause autosomal dominant polycystic kidney disease; PKD1 gene mutations cause ADPKD type 1, a...
2. `medquad_ans_0000587-3__dup1` (score=21.8763) - How is autosomal recessive polycystic kidney disease inherited? Autosomal recessive polycystic kidney disease (ARPKD) is inherited in an autosomal recessive manner. This means that an affected individual has two gene ...
3. `medquad_ans_0004939-1` (score=21.7711) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the ...
4. `medquad_ans_0004945-1` (score=20.8672) - Polycystic liver disease is an inherited condition characterized by many cysts of various sizes scattered throughout the liver. Abdominal discomfort from swelling of the liver may occur; however, most affected individ...
5. `medquad_ans_0000038-22` (score=20.4494) - During your next health care visit, talk to your provider about your test results and how to manage your kidney disease. Below is a list of questions you may want to ask. Add any questions you think are missing, and m...

**Answer**:

The provided context does not include information on what to do for polycystic kidney disease, such as treatment options, management strategies, or clinical recommendations. Therefore, based on the given context, it is not possible to provide a specific answer to the question. [medquad_ans_0000804-3][medquad_ans_0000587-3__dup1][medquad_ans_0004939-1]

**Supported citations**: `medquad_ans_0000804-3`, `medquad_ans_0000587-3__dup1`, `medquad_ans_0004939-1`
**Unsupported citations**: _none_

### Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Scores**: context_relevance=0.909, groundedness=0.785, answer_relevance=0.836

**Top docs**:

1. `medquad_ans_0000061-12` (score=28.8214) - There is no single, definitive test for rheumatoid arthritis. Common tests for rheumatoid arthritis include - The rheumatoid factor test. Rheumatoid factor is an antibody that is present eventually in the blood of mos...
2. `medquad_ans_0000061-3__dup3` (score=28.6855) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by i...
3. `medquad_ans_0002327-3` (score=27.8209) - What causes Felty's syndrome? The exact cause of Felty's syndrome is unknown, although several causes and risk factors have been proposed. Some experts believe it may be an autoimmune disorder, and that it may sometim...
4. `medquad_ans_0000061-5__dup2` (score=27.4583) - Scientists are making rapid progress in understanding the complexities of rheumatoid arthritis. They are learning more about how and why it develops and why some people have more severe symptoms than others. Research ...
5. `medquad_ans_0000061-9` (score=27.4132) - Scientists believe that rheumatoid arthritis may result from the interaction of many factors such as genetics, hormones, and the environment. Although rheumatoid arthritis sometimes runs in families, the actual cause ...

**Answer**:

The provided context does not mention any specific research or clinical trials being conducted for rheumatoid arthritis. [medquad_ans_0000061-12] and [medquad_ans_0000061-3__dup3] discuss diagnostic tests and symptoms but do not include information on ongoing research or clinical trials. Therefore, based on the given context, it is not possible to provide details about current research or clinical trials for rheumatoid arthritis.

**Supported citations**: `medquad_ans_0000061-12`, `medquad_ans_0000061-3__dup3`
**Unsupported citations**: _none_

### Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002561-2` (score=25.1254) - What are the signs and symptoms of Glass-Chapman-Hockley syndrome? Glass-Chapman-Hockley syndrome has only been described in one family with five affected family members in three generations. The signs and symptoms se...
2. `medquad_ans_0002561-4` (score=24.3816) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (an...
3. `medquad_ans_0003103-4` (score=21.4654) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...
4. `medquad_ans_0002561-3` (score=20.2947) - How is Glass-Chapman-Hockley syndrome inherited? Based on the only family that has been reported in the medical literature, to date, the syndrome is believed to be inherited in an autosomal dominant manner.
5. `medquad_ans_0002561-1` (score=19.6948) - The Glass-Chapman-Hockley syndrome is a very rare disease. To date, the syndrome has only been reported in one family with five members affected in three generations. The first patients were two brothers that had an a...

**Answer**:

The treatment for Glass-Chapman-Hockley syndrome is typically surgery for craniosynostosis, which is based on the individual's specific signs and symptoms. The goal of the surgery is to increase the space in the front (anterior) part of the skull. This operation is usually performed between 9 to 12 months of age [medquad_ans_0002561-4]. If other sutures besides the coronal suture are involved, additional surgeries may be required [medquad_ans_0002561-4].

**Supported citations**: `medquad_ans_0002561-4`
**Unsupported citations**: _none_

### Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.851, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000868-1` (score=28.9293) - Renpenning syndrome is a disorder that almost exclusively affects males, causing developmental delay, moderate to severe intellectual disability, and distinctive physical features. Individuals with Renpenning syndrome...
2. `medquad_ans_0000868-5` (score=23.6801) - These resources address the diagnosis or management of Renpenning syndrome: - Genetic Testing Registry: Renpenning syndrome 1 - Greenwood Genetics Center: X-Linked Intellectual Disability - Kennedy Krieger Institute: ...
3. `medquad_ans_0000868-3` (score=23.1962) - Renpenning syndrome is caused by mutations in the PQBP1 gene. This gene provides instructions for making a protein called polyglutamine-binding protein 1. This protein attaches (binds) to stretches of multiple copies ...
4. `medquad_ans_0003103-4` (score=21.4654) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...
5. `medquad_ans_0000108-2__dup1` (score=19.0184) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The provided context does not specify any treatments for Renpenning syndrome. It mentions resources that address the diagnosis or management of the syndrome, such as genetic testing, genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, and palliative care [medquad_ans_0000868-5], but does not detail specific treatments. Therefore, based on the given information, no definitive treatments can be identified.

**Supported citations**: `medquad_ans_0000868-5`
**Unsupported citations**: _none_

### Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Scores**: context_relevance=1.000, groundedness=0.767, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000173-1` (score=27.6040) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
2. `medquad_ans_0001088-1__dup1` (score=26.6913) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
3. `medquad_ans_0001104-5` (score=25.8793) - Is genetic testing available for Charcot-Marie-Tooth disease type 2F? Yes. GeneTests lists the names of laboratories that are performing clincial genetic testing for Charcot-Marie-Tooth disease type 2F. To view the co...
4. `medquad_ans_0001100-1__dup1` (score=25.0570) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is a...
5. `medquad_ans_0000559-1__dup1` (score=24.9742) - What are the signs and symptoms of Autosomal dominant intermediate Charcot-Marie-Tooth disease type C? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant intermediate...

**Answer**:

The provided context does not contain specific information about the symptoms of Charcot-Marie-Tooth disease type 2G. While it describes general symptoms of Charcot-Marie-Tooth disease and differentiates between types 1, 2, and 4, there is no mention of type 2G [medquad_ans_0000173-1]. Therefore, the symptoms of Charcot-Marie-Tooth disease type 2G cannot be determined from the given context.

**Supported citations**: `medquad_ans_0000173-1`
**Unsupported citations**: _none_

### Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Scores**: context_relevance=1.000, groundedness=0.954, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0001393-1` (score=21.8781) - Coccygodynia is a rare condition in that causes pain in and around the coccyx (tailbone). Although various causes have been described for the condition, the more common causes are direct falls and injury.
2. `medquad_ans_0001393-4` (score=21.7079) - What treatment is available for coccygodynia? Treatment for coccygodynia generally falls into conservative management or surgical intervention categories. The conservative approach typically includes hot sitz baths, N...
3. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
4. `medquad_ans_0000549-1__dup1` (score=20.4489) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...
5. `medquad_ans_0002259-1` (score=20.2941) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

The symptoms of coccygodynia are pain in and around the coccyx (tailbone) [medquad_ans_0001393-1].

**Supported citations**: `medquad_ans_0001393-1`
**Unsupported citations**: _none_

### Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Scores**: context_relevance=1.000, groundedness=0.943, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0004025-1` (score=55.7370) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgen...
2. `medquad_ans_0003270-2` (score=35.1269) - What are the signs and symptoms of Isolated corpus callosum agenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Isolated corpus callosum agenesis. If the information is availab...
3. `medquad_ans_0002968-1` (score=33.8395) - What are the signs and symptoms of Hydrolethalus syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Hydrolethalus syndrome. If the information is available, the table below in...
4. `medquad_ans_0005878-1` (score=32.5839) - What are the signs and symptoms of Stargardt macular degeneration absent or hypoplastic corpus callosum mental retardation and dysmorphic features? The Human Phenotype Ontology provides the following list of signs and...
5. `medquad_ans_0001811-1` (score=32.1456) - What are the signs and symptoms of Desmosterolosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Desmosterolosis. If the information is available, the table below includes how oft...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, and microcephaly. These symptoms are listed in the Human Phenotype Ontology as part of the signs and symptoms for this condition [medquad_ans_0004025-1].

**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

### Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.994, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003402-5` (score=30.0819) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to d...
2. `medquad_ans_0003402-3` (score=29.0897) - What causes Klippel Feil syndrome (KFS)? The specific underlying causes and mechanisms of Klippel Feil syndrome (KFS)are not well understood. In general medical researchers believe KFS happens when the tissue of the e...
3. `medquad_ans_0003402-4` (score=25.3339) - Is Klippel Feil syndrome inherited? In some cases, Klippel Feil syndrome (KFS) appears to occur randomly for unknown reasons (sporadically). In other cases, the condition appears to be genetic and may occur in more th...
4. `medquad_ans_0000562-5` (score=23.2119) - These resources address the diagnosis or management of Klippel-Feil syndrome: - Genetic Testing Registry: Klippel Feil syndrome - Genetic Testing Registry: Klippel-Feil syndrome 1, autosomal dominant - Genetic Testing...
5. `medquad_ans_0003402-6` (score=22.4101) - How might Klippel-Feil syndrome be treated? There is no cure for Klippel Feil syndrome (KFS); treatment is generally symptomatic and supportive. Management depends on the features and severity in each person, and can ...

**Answer**:

Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal abnormalities, and additional imaging studies may be needed to assess the extent of the abnormality. Other initial exams are needed to detect additional physical abnormalities or underlying conditions, including examination of the chest to rule out involvement of the heart and lungs, examination of the chest wall to detect possible rib anomalies, MRI for spinal stenosis or neurological deficits, ultrasound of the kidneys for renal abnormalities, hearing evaluation due to the high incidence of hearing loss, and various lab tests to assess organ function. Additional tests or consultations with specialists may be recommended depending on the features present in each person with KFS [medquad_ans_0003402-5].

**Supported citations**: `medquad_ans_0003402-5`
**Unsupported citations**: _none_

### Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Scores**: context_relevance=1.000, groundedness=0.503, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000222-1` (score=28.6260) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
2. `medquad_ans_0000222-4` (score=28.1833) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or c...
3. `medquad_ans_0000222-5` (score=27.7459) - These resources address the diagnosis or management of congenital diaphragmatic hernia: - Boston Children's Hospital - Children's Hospital of Philadelphia - Columbia University Medical Center: DHREAMS - Columbia Unive...
4. `medquad_ans_0000222-3` (score=27.4481) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...
5. `medquad_ans_0001497-2` (score=23.0168) - What are the signs and symptoms of Congenital diaphragmatic hernia? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital diaphragmatic hernia. If the information is available, ...

**Answer**:

Congenital diaphragmatic hernia is a defect in the diaphragm resulting from abnormal development before birth. The diaphragm, composed of muscle and fibrous tissue, separates the abdominal and chest cavities. A defect allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs and potentially leading to pulmonary hypoplasia and life-threatening breathing difficulties at birth. The condition can range from a thinned area in the diaphragm to its complete absence. It is often classified by location: Bochdalek hernias, which occur in the side or back of the diaphragm and account for 80 to 90 percent of cases, and Morgagni hernias, which involve the front part and are less likely to cause severe symptoms. In about 1 percent of cases, there are no symptoms and the condition is detected incidentally. Isolated congenital diaphragmatic hernia is

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Scores**: context_relevance=1.000, groundedness=0.545, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000173-1` (score=17.1370) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
2. `medquad_ans_0001094-1__dup1` (score=16.1195) - Charcot-Marie-Tooth disease type 1E (CMT1E) is a form of Charcot-Marie-Tooth disease, which is a group of rare conditions that affect the peripheral nerves. Signs and symptoms of CMT1E generally become apparent betwee...
3. `medquad_ans_0001088-1__dup1` (score=15.5531) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
4. `medquad_ans_0001104-3` (score=15.4219) - What causes Charcot-Marie-Tooth disease type 2F? Charcot-Marie-Tooth disease type 2F (CMT2F) is caused by mutations in the HSPB1 gene. This gene provides instructions for making a protein (heat shock protein beta-1) w...
5. `medquad_ans_0001104-4` (score=15.2365) - How is Charcot-Marie-Tooth disease type 2F inherited? Charcot-Marie-Tooth disease type 2F is inherited in an autosomal dominant manner. This means that only one mutated copy of the gene in each cell is sufficient to c...

**Answer**:

Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves, which connect the brain and spinal cord to muscles and sensory cells that detect sensations such as touch, pain, heat, and sound. Damage to these nerves can result in loss of sensation and wasting (atrophy) of muscles in the feet, legs, and hands. The disease typically becomes apparent in adolescence or early adulthood, though onset can occur from early childhood through late adulthood. Symptoms vary in severity even among family members and may include balance difficulties, clumsiness, muscle weakness in the feet, foot abnormalities such as high arches (pes cavus), flat feet (pes planus), or curled toes (hammer toes), difficulty flexing the foot or walking on the heel, a higher than normal step, and decreased sensitivity to touch, heat, and cold. In some cases, individuals may experience gradual hearing loss, deafness, or loss of vision. There are

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Scores**: context_relevance=1.000, groundedness=0.912, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000669-2` (score=18.2816) - The prevalence of monilethrix is unknown.
2. `medquad_ans_0000669-1` (score=17.7033) - Monilethrix is a condition that affects hair growth. Its most characteristic feature is that individual strands of hair have a beaded appearance like the beads of a necklace. The name monilethrix comes from the Latin ...
3. `medquad_ans_0000669-4` (score=17.2678) - Monilethrix can have multiple patterns of inheritance. When the condition is caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene...
4. `medquad_ans_0001853-3` (score=14.3676) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0000669-3` (score=12.4639) - Monilethrix is caused by mutations in one of several genes. Mutations in the KRT81 gene, the KRT83 gene, the KRT86 gene, or the DSG4 gene account for most cases of monilethrix. These genes provide instructions for mak...

**Answer**:

Yes, monilethrix can be inherited in different patterns depending on the cause. When caused by mutations in genes such as KRT81, KRT83, KRT86, or DSG4, it can be inherited in an autosomal dominant pattern, meaning one copy of the altered gene is sufficient to cause the disorder. In rare cases, it results from a new mutation and is not inherited. When caused by mutations in the DSG4 gene, it follows an autosomal recessive pattern, requiring mutations in both copies of the gene [medquad_ans_0000669-4]. Therefore, monilethrix can be inherited in either an autosomal dominant or autosomal recessive pattern [medquad_ans_0000669-3].

**Supported citations**: `medquad_ans_0000669-4`, `medquad_ans_0000669-3`
**Unsupported citations**: _none_

### Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Scores**: context_relevance=0.875, groundedness=0.486, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000223-9` (score=28.0663) - Talk with your childs health care provider after your childs UTI is gone. The health care provider may want to do more tests to check for VUR or a blockage in the urinary tract. Repeated infections in an abnormal urin...
2. `medquad_ans_0000071-7__dup1` (score=27.2311) - Urinary tract infections (UTI) are a common bladder problem, especially as people age. UTIs are the second most common type of infection in the body. Each year, UTIs cause more than 8 million visits to health care pro...
3. `medquad_ans_0000215-15` (score=24.5104) - - Urinary tract infections (UTIs) usually occur when the body fails to remove bacteria rapidly from the urinary tract. - UTIs affect about 3 percent of children in the United States every year. - Most UTIs are not ser...
4. `medquad_ans_0000215-4__dup2` (score=24.0967) - Urinary tract infections affect about 3 percent of children in the United States every year. UTIs account for more than 1 million visits to pediatricians offices every year.1
5. `medquad_ans_0000223-2__dup2` (score=23.6637) - Most UTIs are caused by bacteria that live in the bowel, the part of the digestive tract where stool is changed from liquid to solid. The bacterium Escherichia coli (E. coli) causes most UTIs. The urinary tract has se...

**Answer**:

Urinary tract infections (UTIs) in children are a common bladder problem, especially in young children, affecting about 3 percent of children in the United States every year. Most UTIs are caused by bacteria, such as Escherichia coli (E. coli), that live in the bowel. These infections occur when the body fails to remove bacteria rapidly from the urinary tract. Symptoms may range from slight burning with urination or unusual-smelling urine to severe pain and high fever; some children may have no symptoms at all. A UTI in a young child may be a sign of an abnormality in the urinary tract that could lead to repeated infections or chronic kidney infections, which can cause permanent damage. After a child's UTI resolves, it is important to talk with a healthcare provider, as further tests may be needed to check for vesicoureteral reflux (VUR) or a blockage in the urinary tract, which could lead

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000804-1` (score=32.4136) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the...
2. `medquad_ans_0000135-2__dup2` (score=30.7690) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems wi...
3. `medquad_ans_0000804-3` (score=28.6834) - Mutations in the PKD1, PKD2, and PKHD1 genes cause polycystic kidney disease. Mutations in either the PKD1 or PKD2 gene can cause autosomal dominant polycystic kidney disease; PKD1 gene mutations cause ADPKD type 1, a...
4. `medquad_ans_0004945-1` (score=28.6793) - Polycystic liver disease is an inherited condition characterized by many cysts of various sizes scattered throughout the liver. Abdominal discomfort from swelling of the liver may occur; however, most affected individ...
5. `medquad_ans_0004939-2` (score=28.3603) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their abilit...

**Answer**:

The complications of Polycystic Kidney Disease include dangerously high blood pressure (hypertension), pain in the back or sides, blood in the urine (hematuria), recurrent urinary tract infections, kidney stones, heart valve abnormalities, and an increased risk of an abnormal bulging (aneurysm) in the aorta or in blood vessels at the base of the brain [medquad_ans_0000804-1].

**Supported citations**: `medquad_ans_0000804-1`
**Unsupported citations**: _none_

### Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000123-4__dup2` (score=23.1431) - Four types of viruses cause most cases of viral gastroenteritis. Rotavirus Rotavirus is the leading cause of gastroenteritis among infants and young children. Rotavirus infections are most common in infants 3 to 15 mo...
2. `medquad_ans_0000649-1__dup1` (score=17.7417) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contami...
3. `medquad_ans_0000391-1__dup1` (score=15.8372) - Have you ever had the "stomach flu?" What you probably had was gastroenteritis - not a type of flu at all. Gastroenteritis is an inflammation of the lining of the intestines caused by a virus, bacteria or parasites. V...
4. `medquad_ans_0000123-6` (score=14.5254) - Viral gastroenteritis is usually diagnosed based on symptoms alone. People who have symptoms that are severe or last for more than a few days may want to see a health care provider for additional tests. A health care ...
5. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by infection with noroviruses, a group of related viruses. These infections lead to inflammation of the stomach and intestines and can spread from person to person, through contaminated food or water, or by touching a contaminated surface. Common symptoms include diarrhea, nausea and vomiting, and stomach pain; other possible symptoms are fever, headache, or body aches. The illness is usually mild but can be serious, especially in young children and older adults. Symptoms typically appear 1 to 2 days after exposure and last for 1 to 3 days [medquad_ans_0000123-4__dup2]. Treatment involves bed rest and plenty of fluids to prevent dehydration [medquad_ans_0000649-1__dup1].

**Supported citations**: `medquad_ans_0000123-4__dup2`, `medquad_ans_0000649-1__dup1`
**Unsupported citations**: _none_

### Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.900, groundedness=0.929, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000222-3` (score=39.6986) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...
2. `medquad_ans_0000222-1` (score=36.8332) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
3. `medquad_ans_0000222-5` (score=34.2801) - These resources address the diagnosis or management of congenital diaphragmatic hernia: - Boston Children's Hospital - Children's Hospital of Philadelphia - Columbia University Medical Center: DHREAMS - Columbia Unive...
4. `medquad_ans_0000222-4` (score=33.0224) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or c...
5. `medquad_ans_0001497-2` (score=29.9641) - What are the signs and symptoms of Congenital diaphragmatic hernia? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital diaphragmatic hernia. If the information is available, ...

**Answer**:

Genetic changes related to congenital diaphragmatic hernia include changes in single genes and chromosomal abnormalities that affect several genes. Some syndromes associated with congenital diaphragmatic hernia, such as Donnai-Barrow syndrome, Fryns syndrome, and Pallister-Killian mosaic syndrome, are caused by these genetic changes. Additionally, researchers are studying changes in several genes involved in diaphragm development, including transcription factors that regulate gene expression and genes that provide instructions for making proteins involved in cell structure or cell migration during embryonic development [medquad_ans_0000222-3]. However, in approximately 80 percent of cases, no known genetic syndrome or chromosomal abnormality is identified, and the cause remains unknown [medquad_ans_0000222-3].

**Supported citations**: `medquad_ans_0000222-3`
**Unsupported citations**: _none_

### Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.960, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004534-3` (score=24.2616) - What causes Omenn syndrome? Omenn syndrome is a genetically heterogeneous condition (meaning that it may be caused by a number of different genes). While most cases are attributed to mutations in the RAG genes (RAG-1 ...
2. `medquad_ans_0004534-4` (score=18.0406) - How might Omenn syndrome be treated? The standard treatment for Omenn syndrome is bone marrow transplantation or cord blood stem cell transplantation. General care for any patient with severe combined immunodeficiency...
3. `medquad_ans_0004534-1` (score=17.0270) - Omenn syndrome is an autosomal recessive form of severe combined immunodeficiency (SCID) characterized by erythroderma (skin redness), desquamation (peeling skin), alopecia (hair loss), chronic diarrhea, failure to th...
4. `medquad_ans_0004534-2` (score=14.8009) - What are the signs and symptoms of Omenn syndrome? Infants with Omenn syndrome typically present shortly after birth, usually by 3 months of age. This is similar to other types of severe combined immunodeficiency (SCI...
5. `medquad_ans_0001853-3` (score=13.3144) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Omenn syndrome is a genetically heterogeneous condition that may be caused by mutations in several genes. Most cases are attributed to mutations in the RAG1 or RAG2 genes, which are located on chromosome band 11p13. Recent reports have also described Omenn syndrome in the absence of RAG mutations, with causative genes including ARTEMIS, ADA, ILRA2, ILRA7, CHD7, and DNA ligase 4. Additionally, some cases have been found in association with 22q11 microdeletion syndrome [medquad_ans_0004534-3].

**Supported citations**: `medquad_ans_0004534-3`
**Unsupported citations**: _none_

### Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Scores**: context_relevance=1.000, groundedness=0.993, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000097-5` (score=22.1779) - These resources address the diagnosis or management of ARCA1: - Gene Review: Gene Review: SYNE1-Related Autosomal Recessive Cerebellar Ataxia - Genetic Testing Registry: Spinocerebellar ataxia, autosomal recessive 8 -...
2. `medquad_ans_0000097-1` (score=21.8959) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...
3. `medquad_ans_0005792-2` (score=20.9774) - What are the signs and symptoms of Spinocerebellar ataxia autosomal recessive 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia autosomal recessive 7. If the ...
4. `medquad_ans_0000495-1__dup1` (score=20.7781) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...
5. `medquad_ans_0005793-1` (score=20.4355) - What are the signs and symptoms of Spinocerebellar ataxia autosomal recessive 8? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia autosomal recessive 8. If the ...

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the cerebellum). Signs and symptoms of the disorder first appear in early to mid-adulthood. People with this condition initially experience impaired speech (dysarthria), problems with coordination and balance (ataxia), or both. They may also have difficulty with movements that involve judging distance or scale (dysmetria). Other features of ARCA1 include abnormal eye movements (nystagmus) and problems following the movements of objects with the eyes. The movement problems are slowly progressive, often resulting in the need for a cane, walker, or wheelchair [medquad_ans_0000097-1].

**Supported citations**: `medquad_ans_0000097-1`
**Unsupported citations**: _none_

### Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Scores**: context_relevance=0.750, groundedness=0.012, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000049-1__dup1` (score=43.8262) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Erb-Duchenne (Erb's) palsy refers to paral...
2. `medquad_ans_0000050-1__dup2` (score=43.4678) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Symptoms may include a limp or paralyzed a...
3. `medquad_ans_0000123-1__dup2` (score=42.3438) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Symptoms may include - A limp or paralyzed...
4. `medquad_ans_0000050-2__dup1` (score=41.5787) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
5. `medquad_ans_0000049-2__dup1` (score=41.5787) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...

**Answer**:

The provided context does not contain information about treatments for brachial plexus injuries. Therefore, a specific treatment answer cannot be given based on the available information. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Scores**: context_relevance=0.833, groundedness=0.894, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000071-11` (score=20.9949) - Although everyone has some risk for UTIs, some people are more likely to get UTIs than others. These include people who have - spinal cord injuries or other nerve damage around the bladder - a blockage in the urinary ...
2. `medquad_ans_0005123-3` (score=20.1638) - What causes prune belly syndrome? The underlying cause of prune belly syndrome is unknown. The condition may occur if there is a blockage preventing the flow of urine through the urinary tract. The blockage can cause ...
3. `medquad_ans_0000216-4__dup1` (score=19.9991) - When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis. Hydronephrosis is the most common problem found during prenatal ...
4. `medquad_ans_0000154-7` (score=19.9369) - The complications of kidney dysplasia can include - hydronephrosis of the working kidney. A baby with kidney dysplasia in only one kidney might have other urinary tract defects. When other defects in the urinary tract...
5. `medquad_ans_0000216-8` (score=19.8721) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PU...

**Answer**:

Urine blockage in newborns may occur due to a blockage preventing the flow of urine through the urinary tract, which can cause urine to back up and lead to swelling of the ureters and kidneys [medquad_ans_0005123-3]. This blockage may result from an underlying condition such as prune belly syndrome, where the cause is unknown but the blockage prevents urine flow [medquad_ans_0005123-3]. The blockage can cause hydronephrosis and hydroureter, and in severe cases, may impair kidney function and reduce the ability to filter urine [medquad_ans_0000216-4__dup1]. Additionally, urine blockage after birth increases a newborn's risk of developing a urinary tract infection (UTI) [medquad_ans_0000216-4__dup1].

**Supported citations**: `medquad_ans_0005123-3`, `medquad_ans_0000216-4__dup1`
**Unsupported citations**: _none_

### Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Scores**: context_relevance=0.778, groundedness=0.753, answer_relevance=0.689

**Top docs**:

1. `medquad_ans_0000526-1` (score=41.7375) - Intranuclear rod myopathy is a disorder that primarily affects skeletal muscles, which are muscles that the body uses for movement. People with intranuclear rod myopathy have severe muscle weakness (myopathy) and poor...
2. `medquad_ans_0000526-3` (score=39.6718) - Intranuclear rod myopathy is caused by a mutation in the ACTA1 gene. This gene provides instructions for making a protein called skeletal alpha ()-actin, which is part of the actin protein family. Actin proteins are i...
3. `medquad_ans_0000526-4` (score=32.8846) - Intranuclear rod myopathy is an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases are not inherited; they result from new mutations in ...
4. `medquad_ans_0000526-2` (score=32.4025) - Intranuclear rod myopathy is a rare disorder that has been identified in only a small number of individuals. Its exact prevalence is unknown.
5. `medquad_ans_0000526-5` (score=29.5477) - These resources address the diagnosis or management of intranuclear rod myopathy: - Genetic Testing Registry: Nemaline myopathy 3 These resources from MedlinePlus offer information about the diagnosis and management o...

**Answer**:

The provided context does not specify the number of people affected by intranuclear rod myopathy. [medquad_ans_0000526-1] and [medquad_ans_0000526-3] describe the condition's symptoms, causes, and mechanisms but do not include epidemiological data on prevalence or affected population size. Therefore, the number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000526-1`, `medquad_ans_0000526-3`
**Unsupported citations**: _none_

### Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Scores**: context_relevance=1.000, groundedness=0.812, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000036-19` (score=19.8013) - If you are diagnosed with high blood pressure, here are questions to ask your doctor. - Your Blood Pressure Numbers - What is my blood pressure reading in numbers? - What is my goal blood pressure? - Is my blood press...
2. `medquad_ans_0000150-8` (score=19.7627) - - Blood pressure is the force of blood pushing against blood vessel walls as the heart pumps out blood, and high blood pressure, also called hypertension, is an increase in the amount of force that blood places on blo...
3. `medquad_ans_0000036-1__dup3` (score=19.7139) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. What Is Blood Pressure? Blood pressure is the force of blood pushing against the walls of ...
4. `medquad_ans_0000071-1__dup2` (score=19.2953) - Espaol High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. Measuring Blood Pressure Blood pressure is the force of blood pushing against the w...
5. `medquad_ans_0000071-4__dup2` (score=19.2076) - Anyone can develop high blood pressure; however, age, race or ethnicity, being overweight, gender, lifestyle habits, and a family history of high blood pressure can increase your risk for developing high blood pressur...

**Answer**:

The provided context does not define what high blood pressure is. It discusses questions to ask a doctor upon being diagnosed with high blood pressure, such as blood pressure readings, goals, and lifestyle and medication considerations, but it does not provide a definition of high blood pressure [medquad_ans_0000036-19]. Therefore, based on the given information, a definition cannot be provided.

**Supported citations**: `medquad_ans_0000036-19`
**Unsupported citations**: _none_

### Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Scores**: context_relevance=1.000, groundedness=0.831, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0001051-1` (score=23.0470) - X-linked dystonia-parkinsonism is a movement disorder that has been found only in people of Filipino descent. This condition affects men much more often than women. Parkinsonism is usually the first sign of X-linked d...
2. `medquad_ans_0001051-2` (score=20.9449) - X-linked dystonia-parkinsonism has been reported in more than 500 people of Filipino descent, although it is likely that many more Filipinos are affected. Most people with this condition can trace their mother's ances...
3. `medquad_ans_0001051-5` (score=18.8159) - These resources address the diagnosis or management of X-linked dystonia-parkinsonism: - Gene Review: Gene Review: X-Linked Dystonia-Parkinsonism Syndrome - Genetic Testing Registry: Dystonia 3, torsion, X-linked Thes...
4. `medquad_ans_0001853-3` (score=14.3676) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0000369-4` (score=13.5761) - Fanconi anemia is most often inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry ...

**Answer**:

Yes, X-linked dystonia-parkinsonism is inherited. It is an X-linked disorder, meaning the gene responsible is located on the X chromosome. The condition primarily affects men more than women, which is consistent with X-linked inheritance patterns, where males (who have one X chromosome) are more likely to express the condition if they inherit the mutated gene [medquad_ans_0001051-1]. Most people with the condition can trace their mother's ancestry to the island of Panay in the Philippines, suggesting a maternal lineage transmission pattern typical of X-linked inheritance [medquad_ans_0001051-2].

Supporting sources: [medquad_ans_0001051-1], [medquad_ans_0001051-2]

**Supported citations**: `medquad_ans_0001051-1`, `medquad_ans_0001051-2`
**Unsupported citations**: _none_

### Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.526, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000120-2` (score=21.9710) - Beta-ketothiolase deficiency appears to be very rare. It is estimated to affect fewer than 1 in 1 million newborns.
2. `medquad_ans_0000120-1` (score=21.2783) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process keto...
3. `medquad_ans_0000702-1__dup1` (score=17.4473) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ket...
4. `medquad_ans_0000120-5` (score=16.3336) - These resources address the diagnosis or management of beta-ketothiolase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of acetyl-CoA acetyltransferase These resources from MedlinePlus offer in...
5. `medquad_ans_0001853-3` (score=13.7774) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. Signs and symptoms typically appear between the ages of 6 months and 24 months and include episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, known as ketoacidotic attacks, may lead to coma and are frequently triggered by infections, periods without food (fasting), or increased intake of protein-rich foods. The condition is inherited in an autosomal recessive fashion and is caused by mutations in the ACAT1 gene. It is estimated to affect fewer than 1 in 1 million newborns [medquad_ans_0000120-1, medquad_ans_0

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 285 (`0000006_5-7`)

**Question**: What are the stages of Childhood Brain Stem Glioma ?

**Scores**: context_relevance=0.889, groundedness=0.833, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000006_5-1` (score=43.3497) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - T...
2. `medquad_ans_0000006_5-9` (score=37.4918) - Key Points - There are different types of treatment for children with brain stem glioma. - Children with brain stem glioma should have their treatment planned by a team of health care providers who are experts in trea...
3. `medquad_ans_0000006_5-4` (score=33.6600) - The signs and symptoms of brain stem glioma are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain. - The size of the tumor and whether it has spread all thro...
4. `medquad_ans_0000006_3-1` (score=31.4214) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
5. `medquad_ans_0000006_5-7` (score=30.1971) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the...

**Answer**:

The provided context does not mention the stages of childhood brain stem glioma. Therefore, it is not possible to determine the stages based on the given information. [medquad_ans_0000006_5-1]

**Supported citations**: `medquad_ans_0000006_5-1`
**Unsupported citations**: _none_

### Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.528, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000266-1__dup2` (score=33.2644) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal c...
2. `medquad_ans_0000266-3__dup1` (score=25.4586) - With treatment, individuals with tethered spinal cord syndrome have a normal life expectancy. However, some neurological and motor impairments may not be fully correctable. Surgery soon after symptoms emerge appears t...
3. `medquad_ans_0004257-2` (score=21.4622) - What are the signs and symptoms of myelomeningocele? A baby born with a myelomeningocele may have a sac sticking out of the mid to lower back that the doctor cannot see through when shining a light behind it. Symptoms...
4. `medquad_ans_0003573-2` (score=19.6817) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...
5. `medquad_ans_0000061-2__dup1` (score=19.3955) - There is no cure for central cord syndrome although some people recover near-normal function. There is no standard course of treatment, although drug therapy, surgery, and rest are often part of the program. Magnetic ...

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course. Symptoms in children may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is often linked to improper growth of the neural tube during fetal development and is closely associated with spina bifida. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. Delayed symptom presentation is related to chronic strain on the spinal cord, which may be exacerbated

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.024, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000155-1__dup3` (score=33.8456) - You're working at your desk, trying to ignore the tingling or numbness you've had for some time in your hand and wrist. Suddenly, a sharp, piercing pain shoots through the wrist and up your arm. Just a passing cramp? ...
2. `medquad_ans_0000058-1__dup1` (score=28.5549) - Carpal tunnel syndrome (CTS) occurs when the median nerve, which runs from the forearm into the palm of the hand, becomes pressed or squeezed at the wrist. The carpal tunnel is a narrow, rigid passageway of ligament a...
3. `medquad_ans_0000419-1__dup1` (score=27.2994) - No matter how old you are or what you do for a living, you are always using your hands. When there is something wrong with them, you may not be able to do your regular activities. Hand problems include - Carpal tunnel...
4. `medquad_ans_0000058-4__dup1` (score=26.8574) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to conduct fundamental research on the brain and nervous system, and to use that knowledge to reduce the burden of neurological dis...
5. `medquad_ans_0000335-1__dup2` (score=25.7052) - Summary : Ergonomics looks at what kind of work you do, what tools you use and your whole job environment. The aim is to find the best fit between you and your job conditions. Examples of ergonomic changes to your wor...

**Answer**:

The provided context does not contain information about the outlook or prognosis for carpal tunnel syndrome. Therefore, it is not possible to determine the outlook based on the given sources. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.710, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000296-3` (score=24.4543) - Mutations in the LRP2 gene cause Donnai-Barrow syndrome. The LRP2 gene provides instructions for making a protein called megalin, which functions as a receptor. Receptor proteins have specific sites into which certain...
2. `medquad_ans_0000296-1` (score=23.2691) - Donnai-Barrow syndrome is an inherited disorder that affects many parts of the body. This disorder is characterized by unusual facial features, including prominent, wide-set eyes with outer corners that point downward...
3. `medquad_ans_0003103-4` (score=21.4654) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...
4. `medquad_ans_0000222-3` (score=20.1933) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...
5. `medquad_ans_0000296-5` (score=19.8389) - These resources address the diagnosis or management of Donnai-Barrow syndrome: - Gene Review: Gene Review: Donnai-Barrow Syndrome - Genetic Testing Registry: Donnai Barrow syndrome - MedlinePlus Encyclopedia: Diaphrag...

**Answer**:

The provided context does not mention any treatments for Donnai-Barrow syndrome. [medquad_ans_0000296-3] and [medquad_ans_0000296-1] describe the genetic basis, protein function, and clinical features of the disorder but do not address therapeutic interventions. Therefore, based on the given information, treatments for Donnai-Barrow syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000296-3`, `medquad_ans_0000296-1`
**Unsupported citations**: _none_

### Example 289 (`0000811-1`)

**Question**: What is (are) porphyria ?

**Scores**: context_relevance=0.750, groundedness=0.990, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0001134-1` (score=18.3941) - Chester porphyria is a unique type of porphyria with the signs and symptoms of acute intermittent porphyria (AIP) and the biochemical defects of both AIP and variegate porphyria (VP). Chester porphyria does not confor...
2. `medquad_ans_0000189-4__dup2` (score=17.8078) - The exact rates of porphyria are unknown and vary around the world. For example, porphyria cutanea tarda is most common in the United States, and variegate porphyria is most common in South America.1
3. `medquad_ans_0000811-2` (score=17.5474) - The exact prevalence of porphyria is unknown, but it probably ranges from 1 in 500 to 1 in 50,000 people worldwide. Overall, porphyria cutanea tarda is the most common type of porphyria. For some forms of porphyria, t...
4. `medquad_ans_0000189-10` (score=17.1485) - - Porphyrias are rare disorders that affect mainly the skin or nervous system and may cause abdominal pain. - Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosyntheti...
5. `medquad_ans_0000189-3__dup2` (score=15.9830) - Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosynthetic pathway. Experts often classify porphyrias as acute or cutaneous based on the symptoms a person experiences:...

**Answer**:

Porphyrias are rare disorders that affect mainly the skin or nervous system and may cause abdominal pain. Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosynthetic pathway. Most porphyrias result from inheriting an abnormal gene, also called a gene mutation, from one parent. Porphyria cutanea tarda is usually an acquired disorder, meaning factors other than genes cause the enzyme deficiency [medquad_ans_0000189-10].

**Supported citations**: `medquad_ans_0000189-10`
**Unsupported citations**: _none_

### Example 290 (`0000585-1__dup1`)

**Question**: What is (are) Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000585-1__dup1` (score=53.6011) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
2. `medquad_ans_0000585-2__dup1` (score=41.7622) - What are the signs and symptoms of Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal recessive opt...
3. `medquad_ans_0006496-1` (score=40.6601) - X-linked Charcot-Marie-Tooth disease type 5 (CMTX5) is a neurological condition characterized by peripheral neuropathy, early-onset bilateral profound sensorineural hearing loss, and optic neuropathy leading to visual...
4. `medquad_ans_0003414-1` (score=28.3675) - Konigsmark Knox Hussels syndrome is an inherited condition that causes both hearing and vision loss. This condition is characterized by late-onset progressive sensorineural deafness and progressive optic atrophy, whic...
5. `medquad_ans_0000674-1__dup1` (score=27.7558) - Behr syndrome is a disorder mainly characterized by early-onset optic atrophy, ataxia, and spasticity. Other signs and symptoms may be present and vary from person to person. Although the exact cause is unknown, the s...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. Autosomal recessive inheritance has been suggested [medquad_ans_0000585-1__dup1]. The condition also includes signs and symptoms such as areflexia, broad-based gait, distal muscle weakness, distal sensory impairment, distal upper limb amyotrophy, gait ataxia, joint contracture of the hand, pectus excavatum, peripheral demyelination, positive Romberg sign, progressive sensorineural hearing impairment, short thumb, thoracic scoliosis, ulnar deviation of the hand, variable expressivity, and visual impairment [medquad_ans

**Supported citations**: `medquad_ans_0000585-1__dup1`
**Unsupported citations**: _none_

### Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Scores**: context_relevance=0.900, groundedness=0.953, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000794-1` (score=40.0645) - Phosphoglycerate kinase deficiency is a genetic disorder that affects the body's ability to break down the simple sugar glucose, which is the primary energy source for most cells. Researchers have described two major ...
2. `medquad_ans_0000794-5` (score=38.3330) - These resources address the diagnosis or management of phosphoglycerate kinase deficiency: - Children Living with Inherited Metabolic Diseases (CLIMB) (UK): Phosphoglycerate Kinase Deficiency - Genetic Testing Registr...
3. `medquad_ans_0000794-3` (score=34.0287) - Phosphoglycerate kinase deficiency is caused by mutations in the PGK1 gene. This gene provides instructions for making an enzyme called phosphoglycerate kinase, which is involved in a critical energy-producing process...
4. `medquad_ans_0000794-2` (score=32.2051) - Phosphoglycerate kinase deficiency appears to be a rare disorder. About 30 families with affected members have been reported in the scientific literature.
5. `medquad_ans_0001041-3` (score=29.9176) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...

**Answer**:

The genetic changes related to phosphoglycerate kinase deficiency are mutations in the PGK1 gene. These mutations reduce the activity of the enzyme phosphoglycerate kinase, which is involved in glycolysis, a critical energy-producing process in cells. This disruption in energy production leads to cell damage or cell death [medquad_ans_0000794-1]. It is unclear why this abnormality preferentially affects red blood cells, brain cells, or muscle cells in different individuals, but researchers speculate that different PGK1 gene mutations may have varying effects on enzyme activity [medquad_ans_0000794-3].

**Supported citations**: `medquad_ans_0000794-1`, `medquad_ans_0000794-3`
**Unsupported citations**: _none_

### Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Scores**: context_relevance=1.000, groundedness=0.585, answer_relevance=0.920

**Top docs**:

1. `medquad_ans_0000850-5` (score=30.6896) - These resources address the diagnosis or management of pulmonary veno-occlusive disease: - Genetic Testing Registry: Pulmonary veno-occlusive disease These resources from MedlinePlus offer information about the diagno...
2. `medquad_ans_0000453-1` (score=27.8086) - Hepatic veno-occlusive disease with immunodeficiency (also called VODI) is a hereditary disorder of the liver and immune system. Its signs and symptoms appear after the first few months of life. Hepatic veno-occlusive...
3. `medquad_ans_0000850-1` (score=25.0604) - Pulmonary veno-occlusive disease (PVOD) is characterized by the blockage (occlusion) of the blood vessels that carry oxygen-rich (oxygenated) blood from the lungs to the heart (the pulmonary veins). The occlusion is c...
4. `medquad_ans_0000512-3` (score=25.0509) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of devel...
5. `medquad_ans_0003520-3` (score=23.9825) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. Th...

**Answer**:

The provided context does not contain information about the specific genetic changes related to pulmonary veno-occlusive disease (PVOD). While it mentions genetic counseling as a resource for PVOD [medquad_ans_0000850-5] and discusses hepatic veno-occlusive disease with immunodeficiency (VODI) as a hereditary disorder [medquad_ans_0000453-1], no details on genetic mutations or changes associated with PVOD are given. Therefore, the genetic changes related to PVOD cannot be determined from the provided text.

Answer: The context does not provide information on the genetic changes related to pulmonary veno-occlusive disease. [medquad_ans_0000850-5, medquad_ans_0000453-1]

**Supported citations**: `medquad_ans_0000850-5`, `medquad_ans_0000453-1`
**Unsupported citations**: `medquad_ans_0000850-5, medquad_ans_0000453-1`

### Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Scores**: context_relevance=1.000, groundedness=0.850, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0001006-1` (score=29.0907) - Tyrosinemia is a genetic disorder characterized by disruptions in the multistep process that breaks down the amino acid tyrosine, a building block of most proteins. If untreated, tyrosine and its byproducts build up i...
2. `medquad_ans_0001006-2` (score=23.9872) - Worldwide, tyrosinemia type I affects about 1 in 100,000 individuals. This type is more common in Norway where 1 in 60,000 to 74,000 individuals are affected. Tyrosinemia type I is even more common in Quebec, Canada w...
3. `medquad_ans_0003520-3` (score=23.9825) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. Th...
4. `medquad_ans_0000971-3` (score=23.8818) - The causes of task-specific focal dystonia are unknown, although the disorder likely results from a combination of genetic and environmental factors. Certain genetic changes probably increase the likelihood of develop...
5. `medquad_ans_0006260-3` (score=22.5196) - How might tyrosinemia type 1 be treated? There is currently no cure for tyrosinemia type 1. Individuals with this condition need to be on a special diet restricted in two amino acids, tyrosine and phenylalanine, throu...

**Answer**:

The provided context does not specify the exact genetic changes related to tyrosinemia. It states that there are three types of tyrosinemia, each distinguished by their symptoms and genetic cause, but it does not detail the specific genetic mutations or changes associated with each type [medquad_ans_0001006-1]. Therefore, based on the given information, the genetic changes related to tyrosinemia cannot be identified.

**Supported citations**: `medquad_ans_0001006-1`
**Unsupported citations**: _none_

### Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Scores**: context_relevance=1.000, groundedness=0.735, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002675-2` (score=23.5101) - What are the signs and symptoms of Graves' disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Graves' disease. If the information is available, the table below includes how oft...
2. `medquad_ans_0000108-2__dup1` (score=19.7680) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
3. `medquad_ans_0000549-1__dup1` (score=19.4839) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...
4. `medquad_ans_0003103-4` (score=19.3836) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...
5. `medquad_ans_0001354-2` (score=19.3795) - What treatments for metastatic clear cell renal cell carcinoma are available in North America? There are several treatments for metastatic clear cell renal cell carcinoma available in North America. IL-2 and sunitinib...

**Answer**:

The provided context does not include information on the treatments for Graves' disease. [medquad_ans_0002675-2] lists signs and symptoms of Graves' disease but does not mention any treatments. Therefore, based on the available context, the treatments for Graves' disease cannot be determined.

**Supported citations**: `medquad_ans_0002675-2`
**Unsupported citations**: _none_

### Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Scores**: context_relevance=0.800, groundedness=0.499, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000039-1__dup2` (score=19.1767) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, ...
2. `medquad_ans_0000039-3__dup1` (score=18.5976) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten dise...
3. `medquad_ans_0000545-1` (score=18.3811) - Juvenile Batten disease is an inherited disorder that primarily affects the nervous system. After a few years of normal development, children with this condition develop progressive vision loss, intellectual and motor...
4. `medquad_ans_0000545-3` (score=17.7563) - Most cases of juvenile Batten disease are caused by mutations in the CLN3 gene. This gene provides instructions for making a protein whose function is unknown. It is unclear how mutations in the CLN3 gene lead to the ...
5. `medquad_ans_0000545-2` (score=16.2600) - Juvenile Batten disease is the most common type of NCL, but its exact prevalence is unknown. Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, wher...

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. It is characterized by a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. The disease often presents with early signs such as personality and behavior changes, slow learning, clumsiness, or stumbling. Vision loss is frequently an early symptom and may lead to suspicion during an eye exam. Over time, affected children experience cognitive impairment, worsening seizures, and progressive loss of sight and motor skills, eventually becoming blind, bedridden, and demented. Juvenile Batten disease, a form of the disorder, typically manifests between ages 4 and 8 with progressive vision loss, intellectual and motor disability, speech difficulties, and seizures. It is part of a group of disorders known as neuronal ceroid lipofuscinoses (NCLs), which affect the nervous system and cause progressive issues with vision, movement

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Scores**: context_relevance=1.000, groundedness=0.953, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002061-1` (score=33.3946) - Ehlers-Danlos syndrome progeroid type is a genetic disorder of the connective tissue, which is the material between the cells of the body that gives tissues form and strength. The disorder primarily affects the skin, ...
2. `medquad_ans_0002061-3` (score=32.2883) - What causes Ehlers-Danlos syndrome progeroid type? Ehlers-Danlos syndrome progeroid type is caused by changes (mutations) in both of an individual's copies of the B4GALT7 gene, which is located on chromosome 5. This g...
3. `medquad_ans_0000313-5` (score=31.5665) - These resources address the diagnosis or management of Ehlers-Danlos syndrome: - Gene Review: Gene Review: Ehlers-Danlos Syndrome, Classic Type - Gene Review: Gene Review: Ehlers-Danlos Syndrome, Hypermobility Type - ...
4. `medquad_ans_0002061-5` (score=27.5598) - How might Ehlers-Danlos syndrome progeroid type be treated? Individuals with Ehlers-Danlos Syndrome progeroid type can benefit from a variety of treatments depending on their symptoms. Affected children with weak musc...
5. `medquad_ans_0002061-2` (score=26.4609) - What are the signs and symptoms of Ehlers-Danlos syndrome, progeroid type? Ehlers-Danlos syndrome refers to a group of connective tissue disorders characterized by stretchy or kneadable skin, double jointedness, and d...

**Answer**:

Ehlers-Danlos syndrome, progeroid type is a genetic disorder of the connective tissue that primarily affects the skin, hair, and skeletal system. Symptoms usually appear by childhood or adolescence and include unusually flexible joints, loose elastic skin, and easy scarring. Unique features of this type include sparse scalp hair and eyebrows, and loose elastic skin on the face, which cause affected individuals to look older than their age. Additional symptoms may include bone weakness, weak muscle tone, mild intellectual disability, and delayed growth in affected children. The disorder is caused by mutations in both copies of the B4GALT7 gene, located on chromosome 5, which results in reduced production of an enzyme necessary for proper collagen formation in connective tissue. This leads to weak connective tissue and the associated symptoms. The condition is inherited in an autosomal recessive pattern [medquad_ans_0002061-1]medquad*ans*

**Supported citations**: `medquad_ans_0002061-1`
**Unsupported citations**: _none_

### Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=0.900, groundedness=0.978, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000416-3` (score=49.9318) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
2. `medquad_ans_0000416-1` (score=41.1383) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with ...
3. `medquad_ans_0000416-5` (score=39.3759) - These resources address the diagnosis or management of glutamate formiminotransferase deficiency: - Baby's First Test - Genetic Testing Registry: Glutamate formiminotransferase deficiency These resources from MedlineP...
4. `medquad_ans_0002601-1` (score=38.3128) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...
5. `medquad_ans_0000416-2` (score=34.3732) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the metabolism of the amino acid histidine and in the production of folate [medquad_ans_0000416-3]. It is unclear how these changes are related to the specific health problems associated with the mild and severe forms of the disorder or why individuals are affected by one form or the other [medquad_ans_0000416-3].

**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

### Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Scores**: context_relevance=1.000, groundedness=0.928, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0001047-5` (score=52.7654) - These resources address the diagnosis or management of X-linked chondrodysplasia punctata 1: - Gene Review: Gene Review: Chondrodysplasia Punctata 1, X-Linked - Genetic Testing Registry: Chondrodysplasia punctata 1, X...
2. `medquad_ans_0001169-1` (score=46.8863) - Chondrodysplasia punctata 1, X-linked recessive (CDPX1) is a genetic disorder present from birth that affects bone and cartilage development. On x-ray, affected infants have characteristic spots at the ends of their b...
3. `medquad_ans_0001047-1` (score=46.5995) - X-linked chondrodysplasia punctata 1 is a disorder of cartilage and bone development that occurs almost exclusively in males. Chondrodysplasia punctata is an abnormality that appears on x-rays as spots (stippling) nea...
4. `medquad_ans_0001169-2` (score=43.9766) - What are the signs and symptoms of Chondrodysplasia punctata 1, X-linked recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 1, X-linked recessive. I...
5. `medquad_ans_0001048-1` (score=42.3106) - X-linked chondrodysplasia punctata 2 is a disorder characterized by bone, skin, and eye abnormalities. It occurs almost exclusively in females. Although the signs and symptoms of this condition vary widely, almost all...

**Answer**:

Symptoms of Chondrodysplasia punctata 1, X-linked recessive (CDPX1) include characteristic spots (stippled epiphyses) on x-rays at the ends of bones, which typically disappear between ages 2 and 3. Other common features are shortened fingers, a flat nose, and breathing abnormalities. Some individuals may have hearing loss, abnormalities of the spinal bones in the neck, and delayed intellectual development. Additionally, abnormal thickening (stenosis) of the airway cartilage can restrict breathing, and spinal cord compression due to neck bone abnormalities can cause pain, numbness, and weakness. Less commonly, there may be delays in development [medquad_ans_0001169-1].

**Supported citations**: `medquad_ans_0001169-1`
**Unsupported citations**: _none_

### Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Scores**: context_relevance=1.000, groundedness=0.937, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000132-11` (score=25.3556) - - Whipple disease is a rare bacterial infection that primarily affects the small intestine. Left untreated, Whipple disease gets worse and is usually life threatening. - Bacteria called Tropheryma whipplei (T. whipple...
2. `medquad_ans_0000132-5__dup1` (score=23.6847) - Signs and symptoms of Whipple disease can vary widely from person to person. The most common symptoms of Whipple disease are - diarrhea - weight loss caused by malabsorption A person may not have diarrhea. Instead, ot...
3. `medquad_ans_0000108-2__dup1` (score=19.7680) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
4. `medquad_ans_0000549-1__dup1` (score=19.4839) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...
5. `medquad_ans_0003103-4` (score=19.3836) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...

**Answer**:

The treatments for Whipple disease involve antibiotics. The healthcare provider prescribes antibiotics to destroy the Tropheryma whipplei bacteria and treat the disease. Initially, intravenous (IV) antibiotics are usually prescribed for the first 2 weeks of treatment. Most patients feel relief from symptoms within the first week or two. After completing the IV antibiotics, the healthcare provider will prescribe long-term oral antibiotics. Most patients have good outcomes with an early diagnosis and complete treatment [medquad_ans_0000132-11].

**Supported citations**: `medquad_ans_0000132-11`
**Unsupported citations**: _none_

### Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Scores**: context_relevance=0.714, groundedness=0.642, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000042-1__dup2` (score=27.4051) - Bell's palsy is a form of temporary facial paralysis resulting from damage or trauma to the 7th cranial nerve, one of the facial nerves. It is the most common cause of facial paralysis. Generally, Bell's palsy affects...
2. `medquad_ans_0000086-1__dup2` (score=26.6709) - Bell's palsy is the most common cause of facial paralysis. It usually affects just one side of the face. Symptoms appear suddenly and are at their worst about 48 hours after they start. They can range from mild to sev...
3. `medquad_ans_0000042-3__dup2` (score=25.6440) - The prognosis for individuals with Bell's palsy is generally very good. The extent of nerve damage determines the extent of recovery. With or without treatment, most individuals begin to get better within 2 weeks afte...
4. `medquad_ans_0000349-1__dup1` (score=25.2352) - Face injuries and disorders can cause pain and affect how you look. In severe cases, they can affect sight, speech, breathing and your ability to swallow. Broken bones, especially the bones of your nose, cheekbone and...
5. `medquad_ans_0000676-1__dup1` (score=23.7110) - Bell's palsy is a form of temporary facial paralysis which results from damage or trauma to one of the facial nerves. This disorder is characterized by the sudden onset of facial paralysis that often affects just one ...

**Answer**:

The provided context does not mention any specific treatments for Bell's palsy. While it discusses the symptoms, causes, and prognosis, there is no information regarding therapeutic interventions or treatment options. [medquad_ans_0000042-1__dup2], [medquad_ans_0000086-1__dup2], [medquad_ans_0000042-3__dup2], and [medquad_ans_0000349-1__dup1] do not include details on treatments. Therefore, based on the given context, the treatments for Bell's palsy cannot be determined.

**Supported citations**: `medquad_ans_0000042-1__dup2`, `medquad_ans_0000086-1__dup2`, `medquad_ans_0000042-3__dup2`, `medquad_ans_0000349-1__dup1`
**Unsupported citations**: _none_
