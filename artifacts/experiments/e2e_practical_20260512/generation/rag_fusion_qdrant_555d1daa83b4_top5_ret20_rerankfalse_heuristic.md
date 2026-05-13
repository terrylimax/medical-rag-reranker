# Generation Evaluation

## Summary

- `avg_answer_chars`: 538.6533
- `avg_answer_relevance`: 0.7546
- `avg_citation_presence_rate`: 0.8500
- `avg_context_relevance`: 0.8499
- `avg_empty_answer`: 0.0000
- `avg_end_to_end_latency_ms`: 6941.3220
- `avg_generation_latency_ms`: 5757.5117
- `avg_groundedness`: 0.7159
- `avg_insufficient_context`: 0.0433
- `avg_num_retrieved_docs`: 5.0000
- `avg_rerank_latency_ms`: 0.0000
- `avg_retrieval_latency_ms`: 1183.5875
- `avg_supported_citation_rate`: 0.6961
- `avg_unsupported_citation_rate`: 0.1539
- `generation_remote_concurrency`: 4.0000
- `num_examples`: 300
- `reranker_enabled_rate`: 0.0000

## Examples

### Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Scores**: context_relevance=0.750, groundedness=0.042, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000573-1__dup1` (score=0.0674) - Lymphoma is a cancer of a part of the immune system called the lymph system. There are many types of lymphoma. One type is Hodgkin disease. The rest are called non-Hodgkin lymphomas. Non-Hodgkin lymphomas begin when a...
2. `medquad_ans_0000483-1__dup1` (score=0.0530) - Hodgkin disease is a type of lymphoma. Lymphoma is a cancer of a part of the immune system called the lymph system. The first sign of Hodgkin disease is often an enlarged lymph node. The disease can spread to nearby l...
3. `medquad_ans_0003442-1` (score=0.0487) - L1 syndrome is a mild to severe congenital disorder with hydrocephalus of varying degrees of severity, intellectual disability, spasticity of the legs, and adducted thumbs. It includes several conditions, some more se...
4. `medquad_ans_0000572-1__dup1` (score=0.0476) - Lymphedema is the name of a type of swelling. It happens when lymph builds up in your body's soft tissues. Lymph is a fluid that contains white blood cells that defend against germs. It can build up when the lymph sys...
5. `medquad_ans_0006502-4` (score=0.0469) - How is X-linked hypophosphatemia inherited? X-linked hypophosphatemia (XLH) is caused by mutations in the PHEX gene, and is inherited in an X-linked dominant manner. This means that the gene responsible for the condit...

**Answer**:

The provided context does not mention X-linked lymphoproliferative syndrome 1 or its symptoms. Therefore, based on the given information, it is not possible to answer the question. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.830, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000307-1__dup1` (score=0.0297) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...
2. `medquad_ans_0000312-5` (score=0.0164) - These resources address the diagnosis or management of early-onset primary dystonia: - Gene Review: Gene Review: DYT1 Early-Onset Primary Dystonia - Genetic Testing Registry: Dystonia 1 - MedlinePlus Encyclopedia: Mov...
3. `medquad_ans_0004750-4` (score=0.0161) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is pass...
4. `medquad_ans_0001903-1` (score=0.0161) - Dopa-responsive dystonia (DRD) is an inherited type of dystonia that typically begins during childhood but may begin in adolescence or adulthood. Depending on the specific type of DRD, specific symptoms can vary. Feat...
5. `medquad_ans_0003601-2` (score=0.0159) - How is limb-girdle muscular dystrophy inherited? Limb-girdle muscular dystrophy (LGMD) is most often inherited in an autosomal recessive manner; less commonly, rare sub-types may be inherited in an autosomal dominant ...

**Answer**:

Yes, early-onset primary dystonia can be inherited. The Gene Review: DYT1 Early-Onset Primary Dystonia addresses the diagnosis and management of early-onset primary dystonia, indicating that some cases are inherited [medquad_ans_0000312-5]. Additionally, dystonia can be inherited in some cases, as noted in the general description of dystonia [medquad_ans_0000307-1__dup1]. However, the specific inheritance pattern (e.g., autosomal dominant or recessive) is not detailed in the provided context.

**Supported citations**: `medquad_ans_0000312-5`, `medquad_ans_0000307-1__dup1`
**Unsupported citations**: _none_

### Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Scores**: context_relevance=0.857, groundedness=0.907, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000397-1__dup2` (score=0.0328) - Taeniasis in humans is a parasitic infection caused by the tapeworm species Taenia saginata (beef tapeworm), Taenia solium (pork tapeworm), and Taenia asiatica (Asian tapeworm). Humans can become infected with these t...
2. `medquad_ans_0000397-2__dup1` (score=0.0323) - The tapeworms that cause taeniasis (Taenia saginata, T. solium, and T. asiatica) are found worldwide. Eating raw or undercooked beef or pork is the primary risk factor for acquiring taeniasis. Persons who don't eat ra...
3. `medquad_ans_0000121-2__dup3` (score=0.0317) - Cysticercosis is an infection caused by the larvae of the parasite Taenia solium. This infection occurs after a person swallows tapeworm eggs. The larvae get into tissues such as muscle and brain, and form cysts there...
4. `medquad_ans_0000689-1__dup2` (score=0.0308) - Parasites are living things that use other living things - like your body - for food and a place to live. You can get them from contaminated food or water, a bug bite, or sexual contact. Some parasitic diseases are ea...
5. `medquad_ans_0000366-1__dup2` (score=0.0308) - Schistosomiasis, also known as bilharzia, is a disease caused by parasitic worms. Infection with Schistosoma mansoni, S. haematobium, and S. japonicum causes illness in humans; less commonly, S. mekongi and S. interca...

**Answer**:

People at risk for taeniasis include those who eat raw or undercooked beef (particularly infected with Taenia saginata) or pork (particularly infected with Taenia solium or Taenia asiatica). Individuals in under-developed communities with poor sanitation, especially in Latin America, Eastern Europe, sub-Saharan Africa, India, and Asia, are at higher risk. Additionally, people in the United States who are Latin American immigrants or live in areas with poor sanitation and concentrated cattle and human populations (such as around feed lots) are at increased risk [medquad_ans_0000397-1\_\_dup2][medquad_ans_0000397-2__dup1].

**Supported citations**: `medquad_ans_0000397-1__dup2`, `medquad_ans_0000397-2__dup1`
**Unsupported citations**: _none_

### Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.929, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000029-2` (score=0.0817) - Aicardi-Goutieres syndrome is a rare disorder. Its exact prevalence is unknown.
2. `medquad_ans_0000194-2__dup1` (score=0.0794) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
3. `medquad_ans_0000197-1__dup1` (score=0.0794) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000196-1__dup1` (score=0.0791) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
5. `medquad_ans_0000029-1` (score=0.0765) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. Howeve...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy. These frequencies are approximate and based on the Human Phenotype Ontology, which compiles data from sources such as Orphanet [medquad_ans_0000194-2__dup1].

**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

### Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003421-1` (score=0.0820) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0005502-1` (score=0.0784) - Schnitzler syndrome is a rare autoinflammatory condition. Signs and symptoms of the condition vary but may include urticaria; recurrent fevers; joint pain and inflammation; organomegaly (abnormally enlarged organs); a...
3. `medquad_ans_0003423-1` (score=0.0751) - What are the signs and symptoms of Kozlowski Celermajer Tink syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kozlowski Celermajer Tink syndrome. If the information is avail...
4. `medquad_ans_0005502-2` (score=0.0719) - What are the signs and symptoms of Schnitzler syndrome? The signs and symptoms of Schnitzler syndrome vary but may include: Red raised patches of skin (urticaria) that may become itchy Recurrent fevers Join pain and i...
5. `medquad_ans_0003418-1` (score=0.0704) - What are the signs and symptoms of Kosztolanyi syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kosztolanyi syndrome. If the information is available, the table below includ...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

### Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.817, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000069-1__dup3` (score=0.0311) - Espaol Hemophilia (heem-o-FILL-ee-ah) is a rare bleeding disorder in which the blood doesn't clot normally. If you have hemophilia, you may bleed for a longer time than others after an injury. You also may bleed insid...
2. `medquad_ans_0000455-1__dup1` (score=0.0298) - Hemophilia is a rare disorder in which the blood does not clot normally. It is usually inherited. Hemophilia usually occurs in males. If you have hemophilia, you have little or no clotting factor. Clotting factor is a...
3. `medquad_ans_0000067-8` (score=0.0290) - You can't prevent primary, or inherited, hemochromatosis. However, not everyone who inherits hemochromatosis genes develops symptoms or complications of the disease. In those who do, treatments can keep the disease fr...
4. `medquad_ans_0000069-3__dup1` (score=0.0284) - A defect in one of the genes that determines how the body makes blood clotting factor VIII or IX causes hemophilia. These genes are located on the X chromosomes (KRO-muh-somz). Chromosomes come in pairs. Females have ...
5. `medquad_ans_0000117-3__dup3` (score=0.0279) - Abnormal hemoglobin, called hemoglobin S, causes sickle cell disease (SCD). The problem in hemoglobin S is caused by a small defect in the gene that directs the production of the beta globin part of hemoglobin. This s...

**Answer**:

Yes, hemophilia is usually inherited. [medquad_ans_0000069-1__dup3]

**Supported citations**: `medquad_ans_0000069-1__dup3`
**Unsupported citations**: _none_

### Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.537, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004163-1` (score=0.0774) - Mucopolysaccharidosis type IVA (MPS IVA, also called Morquio syndrome, type A) is a metabolic condition that primarily affects the skeleton. The severity, age of onset, and associated symptoms vary significantly from ...
2. `medquad_ans_0003284-1` (score=0.0641) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often t...
3. `medquad_ans_0004163-2` (score=0.0572) - What are the signs and symptoms of Mucopolysaccharidosis type IVA? The Human Phenotype Ontology provides the following list of signs and symptoms for Mucopolysaccharidosis type IVA. If the information is available, th...
4. `medquad_ans_0003103-2` (score=0.0484) - What are the signs and symptoms of IBIDS syndrome? The most common symptoms of Tay syndrome are brittle hair (trichothiodystrophy); dry, thickened, scaling skin (ichthyosis); photosensitivity (abnormal light sensitivi...
5. `medquad_ans_0002992-2` (score=0.0471) - What are the signs and symptoms of Hyper-IgD syndrome? Hyper IgD syndrome is characterized by periodic high fevers accompanied by lymphadenopathy, abdominal pain, diarrhea, headache, joint pain, hepatomegaly and/or sp...

**Answer**:

The symptoms of IVIC syndrome include hearing impairment (90%), limitation of joint mobility (90%), short stature (90%), strabismus (90%), abnormal dermatoglyphics (50%), aplasia/hypoplasia of the thumb (50%), radioulnar synostosis (50%), scoliosis (50%), synostosis of carpal bones (50%), triphalangeal thumb (50%), abnormality of the clavicle (7.5%), arrhythmia (7.5%), leukocytosis (7.5%), preaxial hand polydactyly (7.5%), thrombocytopenia (7.5%), urogenital fistula (7.5%), absent thumb, anal atresia, carpal bone hypoplasia, carpal synostosis, external ophthalmoplegia, hypoplas

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000775-2` (score=0.0656) - Paroxysmal nocturnal hemoglobinuria is a rare disorder, estimated to affect between 1 and 5 per million people.
2. `medquad_ans_0000775-1` (score=0.0645) - Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. The disorder affects red blood cells (erythrocytes), which carry oxygen; white bloo...
3. `medquad_ans_0000775-5` (score=0.0635) - These resources address the diagnosis or management of paroxysmal nocturnal hemoglobinuria: - Duke University School of Medicine: Hemostasis & Thrombosis Center - Genetic Testing Registry: Paroxysmal nocturnal hemoglo...
4. `medquad_ans_0004747-1` (score=0.0620) - Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It can occur at any age, but is usually diagnosed in young adulthood. People ...
5. `medquad_ans_0004744-1` (score=0.0620) - What are the signs and symptoms of Paroxysmal cold hemoglobinuria? The Human Phenotype Ontology provides the following list of signs and symptoms for Paroxysmal cold hemoglobinuria. If the information is available, th...

**Answer**:

Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. It affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes), which protect the body from infection; and platelets (thrombocytes), which are involved in blood clotting. The disorder affects both sexes equally and can occur at any age, although it is most often diagnosed in young adulthood. People with paroxysmal nocturnal hemoglobinuria experience sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses on the body, such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to hemoglobinuria—dark-colored urine due to the presence of hemoglobin. In many, but not all cases, hemoglobinuria is most noticeable in the morning. The premature destruction

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.966, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000010-10` (score=0.0651) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the li...
2. `medquad_ans_0000221-1__dup2` (score=0.0640) - The colon and rectum are part of the large intestine. Colorectal cancer occurs when tumors form in the lining of the large intestine. It is common in both men and women. The risk of developing colorectal cancer rises ...
3. `medquad_ans_0000010-1__dup5` (score=0.0633) - How Tumors Form The body is made up of many types of cells. Normally, cells grow, divide, and produce more cells as needed to keep the body healthy and functioning properly. Sometimes, however, the process goes wrong ...
4. `medquad_ans_0000037_3-1` (score=0.0613) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...
5. `medquad_ans_0000037_4-1` (score=0.0613) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...

**Answer**:

Colorectal cancer is cancer of the colon or rectum, which are parts of the large intestine. It occurs when malignant tumors form in the lining of the large intestine, also called the large bowel. The risk of developing colorectal cancer rises after age 50 and is common in both men and women. It can spread from its original location in the colon to other parts of the body through the bloodstream or lymphatic system [medquad_ans_0000010-10][medquad_ans_0000221-1__dup2].

**Supported citations**: `medquad_ans_0000010-10`, `medquad_ans_0000221-1__dup2`
**Unsupported citations**: _none_

### Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Scores**: context_relevance=1.000, groundedness=0.949, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000544-2` (score=0.0656) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.
2. `medquad_ans_0000544-5` (score=0.0645) - These resources address the diagnosis or management of junctional epidermolysis bullosa: - Epidermolysis Bullosa Center, Cincinnati Children's Hospital Medical Center - Gene Review: Gene Review: Junctional Epidermolys...
3. `medquad_ans_0002121-1` (score=0.0635) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main ty...
4. `medquad_ans_0003317-2` (score=0.0609) - What are the signs and symptoms of Junctional epidermolysis bullosa? The Human Phenotype Ontology provides the following list of signs and symptoms for Junctional epidermolysis bullosa. If the information is available...
5. `medquad_ans_0003317-1` (score=0.0604) - Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz ...

**Answer**:

Junctional epidermolysis bullosa is a rare genetic skin disease that causes the skin to blister easily in response to minor injuries or friction, such as rubbing or scratching. It is one of four main types of epidermolysis bullosa, along with dystrophic epidermolysis bullosa, epidermolysis bullosa simplex, and Kindler syndrome [medquad_ans_0002121-1]. Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States [medquad_ans_0000544-2]. Common signs and symptoms include abnormal blistering of the skin (seen in 90% of patients), abnormalities of dental enamel, fingernails, and toenails (each in 90% of patients), aplasia/hypoplasia of the skin (90%),

**Supported citations**: `medquad_ans_0002121-1`, `medquad_ans_0000544-2`
**Unsupported citations**: _none_

### Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.919, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000668-2` (score=0.0656) - Molybdenum cofactor deficiency is a rare condition that is estimated to occur in 1 in 100,000 to 200,000 newborns worldwide. More than 100 cases have been reported in the medical literature, although it is thought tha...
2. `medquad_ans_0000668-3` (score=0.0645) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same ...
3. `medquad_ans_0000668-1` (score=0.0632) - Molybdenum cofactor deficiency is a rare condition characterized by brain dysfunction (encephalopathy) that worsens over time. Babies with this condition appear normal at birth, but within a week they have difficulty ...
4. `medquad_ans_0000668-5` (score=0.0627) - These resources address the diagnosis or management of molybdenum cofactor deficiency: - Genetic Testing Registry: Combined molybdoflavoprotein enzyme deficiency - Genetic Testing Registry: Molybdenum cofactor deficie...
5. `medquad_ans_0004143-3` (score=0.0613) - What causes Mowat-Wilson syndrome? Mowat-Wilson syndrome is caused by mutations in the ZEB2 (also known as ZFHX1B or SIP-1) gene. This gene provides instructions for making a protein that plays a critical role in the ...

**Answer**:

The genetic changes related to molybdenum cofactor deficiency are mutations in the MOCS1, MOCS2, or GPHN gene. These mutations cause type A, B, or C of the disorder, respectively. Type A is caused by mutations in the MOCS1 gene, type B by mutations in the MOCS2 gene, and type C by mutations in the GPHN gene. These genetic mutations reduce or eliminate the function of the associated protein, impairing the biosynthesis of the molybdenum cofactor [medquad_ans_0000668-3].

**Supported citations**: `medquad_ans_0000668-3`
**Unsupported citations**: _none_

### Example 12 (`0000756-1`)

**Question**: What is (are) ornithine translocase deficiency ?

**Scores**: context_relevance=0.667, groundedness=0.963, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000756-2` (score=0.0656) - Ornithine translocase deficiency is a very rare disorder. Fewer than 100 affected individuals have been reported worldwide.
2. `medquad_ans_0000756-5` (score=0.0645) - These resources address the diagnosis or management of ornithine translocase deficiency: - Baby's First Test - Gene Review: Gene Review: Hyperornithinemia-Hyperammonemia-Homocitrullinuria Syndrome - Gene Review: Gene ...
3. `medquad_ans_0000755-2` (score=0.0630) - Ornithine transcarbamylase deficiency is believed to occur in approximately 1 in every 80,000 people.
4. `medquad_ans_0000755-4` (score=0.0616) - Ornithine transcarbamylase deficiency is an X-linked disorder. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes. A characte...
5. `medquad_ans_0000755-5` (score=0.0614) - These resources address the diagnosis or management of ornithine transcarbamylase deficiency: - Baby's First Test - Gene Review: Gene Review: Ornithine Transcarbamylase Deficiency - Gene Review: Gene Review: Urea Cycl...

**Answer**:

Ornithine translocase deficiency is a very rare disorder, with fewer than 100 affected individuals reported worldwide [medquad_ans_0000756-2].

**Supported citations**: `medquad_ans_0000756-2`
**Unsupported citations**: _none_

### Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Scores**: context_relevance=0.857, groundedness=0.908, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000362-1__dup2` (score=0.0328) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
2. `medquad_ans_0000036-8` (score=0.0320) - Medicaid is a state health insurance program for those with low incomes and few assets. Each state runs its own program. The Federal Government requires that Medicaid programs cover a specific set of services; however...
3. `medquad_ans_0000586-1__dup2` (score=0.0313) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state ha...
4. `medquad_ans_0000036-16` (score=0.0308) - - Diabetes management and treatment is expensive. Many people who have diabetes need help paying for their care. For those who qualify, a variety of government and nongovernment programs can help cover health care exp...
5. `medquad_ans_0000045-9` (score=0.0306) - Medicare Part B is medical insurance. It helps cover medical services such as doctor's services, outpatient care and other medical services that Part A doesn't cover. Part B also covers some preventive services, such ...

**Answer**:

Yes, there is information about financial assistance. If you don't have health insurance or need help with costs that aren't covered, financial assistance might be available through certain government programs and nonprofit organizations. You can also discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2]. Additionally, Medicaid, a state health insurance program for low-income individuals, can provide financial assistance and may pay for services that Medicare does not cover [medquad_ans_0000036-8]. Eligibility for Medicaid depends on factors such as income, age, pregnancy, disability, or blindness, and individuals can apply through a city or county department of social services or with the help of a social worker [medquad_ans_0000586-1__dup2]. CHIP, a federal and state

**Supported citations**: `medquad_ans_0000362-1__dup2`, `medquad_ans_0000036-8`, `medquad_ans_0000586-1__dup2`
**Unsupported citations**: _none_

### Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0004105-1` (score=0.0656) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About hal...
2. `medquad_ans_0004105-2` (score=0.0645) - What are the symptoms of Mollaret meningitis? The symptoms of Mollaret meningitis are the same as those found in other types of meningitis. In Mollaret meningitis, however, the symptoms are recurring and are often acc...
3. `medquad_ans_0000115-1__dup1` (score=0.0630) - Meningitis is an infection of the meninges, the membranes that surround the brain and spinal cord. Encephalitis is inflammation of the brain itself. Causes of encephalitis and meningitis include viruses, bacteria, fun...
4. `medquad_ans_0000598-1__dup2` (score=0.0606) - Meningitis is inflammation of the thin tissue that surrounds the brain and spinal cord, called the meninges. There are several types of meningitis. The most common is viral meningitis, which you get when a virus enter...
5. `medquad_ans_0000115-2__dup1` (score=0.0605) - Anyone experiencing symptoms of meningitis or encephalitis should see a doctor immediately. Antibiotics for most types of meningitis can greatly reduce the risk of dying from the disease. Antiviral medications may be ...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches, separated by weeks or months of no symptoms. About half of affected individuals may also experience long-term abnormalities of the nervous system that come and go, such as seizures, double vision, abnormal reflexes, paralysis of a cranial nerve (paresis), hallucinations, or coma. The exact cause remains unknown, but recent data suggests that herpes simplex virus (HSV-2 and, less frequently, HSV-1) may cause some, if not most cases. Other potential causes include trauma and viral infections other than herpes simplex [medquad_ans_0004105-1]. The symptoms of Mollaret meningitis are similar to those of other types of meningitis and include high fever, severe headache, nausea, vomiting, stiff neck, photophobia (s

**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

### Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.785, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003319-1` (score=0.0817) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. U...
2. `medquad_ans_0003331-1` (score=0.0804) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
3. `medquad_ans_0003331-3` (score=0.0789) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...
4. `medquad_ans_0000551-2` (score=0.0786) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.
5. `medquad_ans_0003319-2` (score=0.0772) - What are the signs and symptoms of juvenile amyotrophic lateral sclerosis? Signs and symptoms of juvenile ALS vary but include slowly to very slowly progressive muscle weakness, increased muscle tone, Babinski reflex,...

**Answer**:

The provided context does not specify the cause of juvenile amyotrophic lateral sclerosis (ALS). While it states that juvenile ALS is often genetic and may be inherited in an autosomal dominant or autosomal recessive fashion, it does not detail the specific genetic mutations or molecular mechanisms responsible [medquad_ans_0003319-1]. Therefore, based on the available information, the exact cause of juvenile ALS cannot be determined.

**Supported citations**: `medquad_ans_0003319-1`
**Unsupported citations**: _none_

### Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Scores**: context_relevance=0.778, groundedness=0.922, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000219-11` (score=0.0489) - - Erectile dysfunction (ED) is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last...
2. `medquad_ans_0000334-1__dup2` (score=0.0484) - Erectile dysfunction (ED) is a common type of male sexual dysfunction. It is when a man has trouble getting or keeping an erection. ED becomes more common as you get older. But it's not a natural part of aging. Some p...
3. `medquad_ans_0000219-1__dup4` (score=0.0476) - Erectile dysfunction\* is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last long ...
4. `medquad_ans_0000219-8` (score=0.0471) - Your doctor can offer you a number of treatments for ED. For many men, the answer is as simple as taking a pill. Other men have to try two or three options before they find a treatment that works for them. Dont give u...
5. `medquad_ans_0000219-4__dup3` (score=0.0459) - Erectile dysfunction affects men of all races and in all parts of the world. Men are more likely to have ED as they get older. For example, ED occurs in - about 12 percent of men younger than 60 - 22 percent of men ag...

**Answer**:

To know about Erectile Dysfunction (ED), you should understand that it is when you cannot get or keep an erection firm enough to have sex. This may include situations where you get an erection sometimes but not every time, get an erection that does not last long enough, or are unable to get an erection at all [medquad_ans_0000219-11]. An erection occurs due to increased blood flow into the penis, involving hormones, blood vessels, nerves, and muscles working together [medquad_ans_0000219-11]. ED often has multiple causes, and many diseases can damage nerves, arteries, and muscles [medquad_ans_0000219-11].

To determine the cause of ED, your doctor may take your medical and sexual history, ask about your mental health, perform a physical exam, test your blood, conduct a nighttime erection test, perform an injection

**Supported citations**: `medquad_ans_0000219-11`
**Unsupported citations**: _none_

### Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.946, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0006267-3` (score=0.0820) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
2. `medquad_ans_0001537-1` (score=0.0796) - Congenital muscular dystrophy (CMD) refers to a group of inherited conditions that affect the muscles and are present at birth or in early infancy. The severity of the condition, the associated signs and symptoms and ...
3. `medquad_ans_0000334-5` (score=0.0778) - These resources address the diagnosis or management of facioscapulohumeral muscular dystrophy: - Gene Review: Gene Review: Facioscapulohumeral Muscular Dystrophy - Genetic Testing Registry: Facioscapulohumeral muscula...
4. `medquad_ans_0001026-5` (score=0.0756) - These resources address the diagnosis or management of Walker-Warburg syndrome: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Genetic Testing Registry: Walker-Warburg congenital muscular dystrop...
5. `medquad_ans_0000576-5` (score=0.0752) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...

**Answer**:

Physical therapy, including early mobilization, regular stretching, and splinting, is the main focus of supportive care for Ullrich congenital muscular dystrophy. Respiratory support and night-time ventilation may become necessary in the first or second decade of life. Prevention of chest infections can be achieved with the use of antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that cyclosporin A may benefit individuals with Ullrich congenital muscular dystrophy, although more studies are needed to confirm its efficacy [medquad_ans_0006267-3].

**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

### Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004005-1` (score=0.0328) - Microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) is a genetic condition that is mainly characterized by intrauterine and post-natal growth retardation; an abnormally small head size (microcephaly); abn...
2. `medquad_ans_0004006-1` (score=0.0323) - Microcephalic osteodysplastic primordial dwarfism type 2 (MOPD2) is a condition characterized by short stature (dwarfism), skeletal abnormalities and an unusually small head size (microcephaly). Other signs and sympto...
3. `medquad_ans_0004005-5` (score=0.0317) - How might microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) be treated? At this time there are no specific treatments for MOPD1. Treatment is generally supportive. The prognosis is poor for affected ind...
4. `medquad_ans_0000653-1` (score=0.0312) - Microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is a condition characterized by short stature (dwarfism) with other skeletal abnormalities (osteodysplasia) and an unusually small head size (microcep...
5. `medquad_ans_0004006-2` (score=0.0308) - What are the signs and symptoms of Microcephalic osteodysplastic primordial dwarfism type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephalic osteodysplastic primordial ...

**Answer**:

Yes, microcephalic osteodysplastic primordial dwarfism type II (MOPD2) is inherited in an autosomal recessive manner [medquad_ans_0004006-1].

**Supported citations**: `medquad_ans_0004006-1`
**Unsupported citations**: _none_

### Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.749, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000742-2` (score=0.0328) - Ochoa syndrome is a rare disorder. About 150 cases have been reported in the medical literature.
2. `medquad_ans_0000742-5` (score=0.0323) - These resources address the diagnosis or management of Ochoa syndrome: - Gene Review: Gene Review: Urofacial Syndrome - Genetic Testing Registry: Ochoa syndrome - National Institute of Diabetes and Digestive and Kidne...
3. `medquad_ans_0006597-4` (score=0.0315) - Can 48,XXYY syndrome be inherited?
4. `medquad_ans_0000742-1` (score=0.0315) - Ochoa syndrome is a disorder characterized by urinary problems and unusual facial expressions. The urinary problems associated with Ochoa syndrome typically become apparent in early childhood or adolescence. People wi...
5. `medquad_ans_0000883-4` (score=0.0305) - This condition is typically inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutatio...

**Answer**:

The provided context does not specify whether Ochoa syndrome is inherited. While it describes the symptoms and associated conditions, such as urinary problems and facial expressions, and mentions that some disorders like Romano-Ward syndrome are inherited in an autosomal dominant pattern, there is no direct information about the inheritance pattern of Ochoa syndrome [medquad_ans_0000742-1][medquad_ans_0000883-4]. Therefore, based on the given context, the inheritance of Ochoa syndrome cannot be determined.

Answer: Insufficient information to determine if Ochoa syndrome is inherited.

**Supported citations**: `medquad_ans_0000742-1`, `medquad_ans_0000883-4`
**Unsupported citations**: _none_

### Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.756, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000531-2` (score=0.0820) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
2. `medquad_ans_0000531-3` (score=0.0806) - Isodicentric chromosome 15 syndrome results from the presence of an abnormal extra chromosome, called an isodicentric chromosome 15, in each cell. An isodicentric chromosome contains mirror-image segments of genetic m...
3. `medquad_ans_0000531-1` (score=0.0786) - Isodicentric chromosome 15 syndrome is a developmental disorder with a broad spectrum of features. The signs and symptoms vary among affected individuals. Poor muscle tone is commonly seen in individuals with isodicen...
4. `medquad_ans_0000531-5` (score=0.0781) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management...
5. `medquad_ans_0001226-1` (score=0.0774) - Chromosome 15q deletion is a chromosome abnormality that occurs when there is a missing copy of the genetic material located on the long arm (q) of chromosome 15. The severity of the condition and the signs and sympto...

**Answer**:

The provided context does not mention any treatments for isodicentric chromosome 15 syndrome. Therefore, based on the available information, it is not possible to determine the treatments for this condition. [medquad_ans_0000531-1][medquad_ans_0000531-3][medquad_ans_0000531-2]

**Supported citations**: `medquad_ans_0000531-1`, `medquad_ans_0000531-3`, `medquad_ans_0000531-2`
**Unsupported citations**: _none_

### Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.015, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000115-1__dup4` (score=0.0709) - Mntriers disease causes the ridges along the inside of the stomach wallcalled rugaeto enlarge, forming giant folds in the stomach lining. The rugae enlarge because of an overgrowth of mucous cells in the stomach wall....
2. `medquad_ans_0000115-2__dup2` (score=0.0654) - Scientists are unsure about what causes Mntriers disease; however, researchers think that most people acquire, rather than inherit, the disease. In extremely rare cases, siblings have developed Mntriers disease as chi...
3. `medquad_ans_0004414-3` (score=0.0479) - What causes Nicolaides-Baraitser syndrome? Nicolaides-Baraitser syndrome (NCBRS) is caused by mutations in the SMARCA2 gene, which is located on the small arm of chromosome 9. All mutations that have been identified i...
4. `medquad_ans_0002439-1` (score=0.0464) - Friedreich ataxia is an inherited condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features...
5. `medquad_ans_0001143-1` (score=0.0462) - CHILD syndrome, also known as congenital hemidysplasia with ichthyosiform erythroderma and limb defects, is a genetic condition that is typically characterized by large patches of skin that are red and inflamed (eryth...

**Answer**:

The provided context does not mention Troyer syndrome or its causes. Therefore, based on the given information, it is not possible to answer the question. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.136, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000133-3__dup2` (score=0.0539) - Wilson disease is caused by an inherited autosomal recessive mutation, or change, in the ATP7B gene. In an autosomal recessive disease, the child has to inherit the gene mutation from both parents to have an increased...
2. `medquad_ans_0000181-5__dup1` (score=0.0463) - Alagille syndrome occurs in about one of every 30,000 live births.3 The disorder affects both sexes equally and shows no geographical, racial, or ethnic preferences. JAG1 and NOTCH2 gene mutations are inherited in an ...
3. `medquad_ans_0000181-3__dup2` (score=0.0372) - Alagille syndrome is caused by a gene mutation, or defect. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the DNA sequence that makes up a gene. DNA, or deoxyribon...
4. `medquad_ans_0000983-4` (score=0.0328) - This condition is considered to have an autosomal dominant pattern of inheritance, which means one copy of the altered CACNA1C gene in each cell is sufficient to cause the disorder. Most cases result from new mutation...
5. `medquad_ans_0003672-1` (score=0.0323) - Timothy syndrome is a type of long QT syndrome. It affects many parts of the body including the heart, fingers, toes, face, and the nervous system. It is characterized by long QT syndrome, although some people with Ti...

**Answer**:

The provided context does not contain any information about the genetic changes related to Timothy syndrome. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000875-1` (score=0.0807) - Rheumatoid arthritis is a disease that causes chronic abnormal inflammation, primarily affecting the joints. The most common signs and symptoms are pain, swelling, and stiffness of the joints. Small joints in the hand...
2. `medquad_ans_0000061-11` (score=0.0805) - Rheumatoid arthritis can be difficult to diagnose in its early stages because the full range of symptoms develops over time, and only a few symptoms may be present in the early stages. As part of the diagnosis, your d...
3. `medquad_ans_0000061-7__dup1` (score=0.0794) - Rheumatoid arthritis is an inflammatory disease that causes pain, swelling, stiffness, and loss of function in the joints. It can cause mild to severe symptoms. People with rheumatoid arthritis may feel sick, tired, a...
4. `medquad_ans_0000061-3__dup3` (score=0.0770) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by i...
5. `medquad_ans_0000061-10` (score=0.0769) - Rheumatoid arthritis is characterized by inflammation of the joint lining. This inflammation causes warmth, redness, swelling, and pain around the joints. The pain of rheumatoid arthritis varies greatly from person to...

**Answer**:

The symptoms of rheumatoid arthritis include pain, swelling, and stiffness of the joints, with small joints in the hands and feet being most commonly affected [medquad_ans_0000875-1]. The joint pain and stiffness are often worse when getting out of bed in the morning or after a long rest. The disease typically affects joints in a symmetrical pattern, such as both hands or both knees [medquad_ans_0000875-1][medquad_ans_0000061-7__dup1]. Additional symptoms may include a loss of energy, low fever, weight loss, and anemia [medquad_ans_0000875-1]. In some cases, individuals develop rheumatoid nodules, which are firm, noncancerous lumps under the skin or elsewhere in the body [medquad_ans_0000875

**Supported citations**: `medquad_ans_0000875-1`, `medquad_ans_0000061-7__dup1`
**Unsupported citations**: _none_

### Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Scores**: context_relevance=1.000, groundedness=0.911, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0002796-2` (score=0.0648) - What are the signs and symptoms of Henoch-Schonlein purpura? The Human Phenotype Ontology provides the following list of signs and symptoms for Henoch-Schonlein purpura. If the information is available, the table belo...
2. `medquad_ans_0002796-4` (score=0.0638) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...
3. `medquad_ans_0002796-1` (score=0.0638) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is no...
4. `medquad_ans_0000149-1__dup3` (score=0.0630) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, d...
5. `medquad_ans_0000149-2__dup2` (score=0.0618) - Henoch-Schnlein purpura is caused by an abnormal immune system response in which the bodys immune system attacks the bodys own cells and organs. Usually, the immune system makes antibodies, or proteins, to protect the...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive issues, and other symptoms. It is characterized by signs and symptoms such as abdominal pain, arthralgia, bruising susceptibility, hematuria, nausea and vomiting, skin rash, and vasculitis, which are present in approximately 90% of patients [medquad_ans_0002796-1]. The condition is also associated with gastrointestinal infarctions, gastrointestinal hemorrhage, and glomerulopathy in some cases [medquad_ans_0002796-2]. There is no cure for HSP; treatment focuses on relieving symptoms using medications such as non-steroidal anti-inflammatory drugs (NSAIDs), corticosteroids, or immunosuppressive drugs in severe cases involving kidney damage [medquad_ans_0

**Supported citations**: `medquad_ans_0002796-1`, `medquad_ans_0002796-2`
**Unsupported citations**: _none_

### Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Scores**: context_relevance=1.000, groundedness=0.318, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000477-1__dup1` (score=0.0500) - HIV stands for human immunodeficiency virus. It kills or damages the body's immune system cells. AIDS stands for acquired immunodeficiency syndrome. It is the most advanced stage of infection with HIV. HIV most often ...
2. `medquad_ans_0003166-1` (score=0.0492) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information ...
3. `medquad_ans_0003162-1` (score=0.0481) - Hyper IgM syndrome is a type of primary immunodeficiency syndrome. Primary immunodeficiency occurs when part of a persons immune system is missing or does not work correctly. The bodies of people with primary immunode...
4. `medquad_ans_0003162-2` (score=0.0479) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 1? Symptoms and physical findings associated with hyper IgM syndrome usually become apparent in the first or second year of life. This condition ...
5. `medquad_ans_0003163-1` (score=0.0469) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information ...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [doc_id: medquad_ans_0003166-1].

**Supported citations**: _none_
**Unsupported citations**: `doc_id: medquad_ans_0003166-1`

### Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.975, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000896-3` (score=0.0487) - Senior-Lken syndrome can be caused by mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopic, finger...
2. `medquad_ans_0000896-1` (score=0.0479) - Senior-Lken syndrome is a rare disorder characterized by the combination of two specific features: a kidney condition called nephronophthisis and an eye condition known as Leber congenital amaurosis. Nephronophthisis ...
3. `medquad_ans_0005542-1` (score=0.0466) - Senior Loken syndrome is a rare disorder characterized by the combination of two specific features: a kidney condition called nephronophthisis and an eye condition known as Leber congenital amaurosis. It can be caused...
4. `medquad_ans_0003473-3` (score=0.0462) - What causes Laron syndrome? Laron syndrome is caused by changes (mutations) in the GHR gene. This gene encodes growth hormone receptor, which is a protein found on the outer membrane of cells throughout the body. Grow...
5. `medquad_ans_0000174-3__dup1` (score=0.0450) - What causes adult-onset vitelliform dystrophy? Historically, adult-onset vitelliform macular dystrophy (AVMD) was defined as a genetic disorder; however, recent studies have concluded that only a minority of cases hav...

**Answer**:

Genetic changes related to Senior-Lken syndrome include mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia, which are involved in signaling pathways between cells. Mutations in these genes likely lead to problems with the structure and function of cilia, disrupting important chemical signaling pathways within cells. However, it remains unclear how these defects specifically lead to nephronophthisis and Leber congenital amaurosis. In some cases, no mutations in these five known genes are identified, and the genetic cause of the disorder remains unknown [medquad_ans_0000896-3].

**Supported citations**: `medquad_ans_0000896-3`
**Unsupported citations**: _none_

### Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000610-2` (score=0.0492) - The prevalence of lymphedema-distichiasis syndrome is unknown. Because the extra eyelashes can be overlooked during a medical examination, researchers believe that some people with this condition may be misdiagnosed a...
2. `medquad_ans_0003700-2` (score=0.0484) - What are the signs and symptoms of Lymphedema-distichiasis syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lymphedema-distichiasis syndrome. If the information is available...
3. `medquad_ans_0003700-1` (score=0.0476) - Lymphedema distichiasis syndrome is a condition that affects the normal function of the lymphatic system (part of the immune system that produces and transports fluids and immune cells throughout the body). People wit...
4. `medquad_ans_0000610-1` (score=0.0469) - Lymphedema-distichiasis syndrome is a condition that affects the normal function of the lymphatic system, which is a part of the circulatory and immune systems. The lymphatic system produces and transports fluids and ...
5. `medquad_ans_0003100-1` (score=0.0457) - Hypotrichosis-lymphedema-telangiectasia syndrome (HLTS) is a rare condition that, as the name suggests, is associated with sparse hair (hypotrichosis), lymphedema, and telangiectasia, particularly on the palms of the ...

**Answer**:

The prevalence of lymphedema-distichiasis syndrome is unknown [medquad_ans_0000610-2].

**Supported citations**: `medquad_ans_0000610-2`
**Unsupported citations**: _none_

### Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Scores**: context_relevance=0.857, groundedness=0.397, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000013-22` (score=0.0328) - In a genetic test, a small sample of blood, saliva, or tissue is taken to examine a person's genes. Sometimes, genetic testing can detect diseases that may be preventable or treatable. This type of testing is availabl...
2. `medquad_ans_0000399-1__dup1` (score=0.0323) - Summary : Genetic tests are tests on blood and other tissue to find genetic disorders. Over 2000 tests are available. Doctors use genetic tests for several reasons. These include - Finding genetic diseases in unborn b...
3. `medquad_ans_0000013-25` (score=0.0317) - Genetic testing may be helpful whether the test identifies a mutation or not. Test results can - serve as a relief, eliminating some of the uncertainty about a health condition - help doctors make recommendations for ...
4. `medquad_ans_0000013-26` (score=0.0312) - Finding out your test results can affect you emotionally. Learning that you are someone in your family has or is at risk for a disease can be scary. Some people can also feel guilty, angry, anxious, or depressed when ...
5. `medquad_ans_0000397-1__dup1` (score=0.0303) - Summary : Genetic counseling provides information and support to people who have, or may be at risk for, genetic disorders. A genetic counselor meets with you to discuss genetic risks. The counseling may be for yourse...

**Answer**:

Yes, I have information about genetic testing. Genetic testing involves taking a small sample of blood, saliva, or tissue to examine a person's genes. It can detect genetic disorders and is available for thousands of conditions. Genetic tests are used for various purposes, including identifying genetic diseases in unborn babies, determining if individuals carry a gene that could be passed to their children, screening embryos for disease, testing adults before symptoms appear, making a diagnosis in someone with symptoms, and determining the best type or dose of medicine for a person. Test results can provide relief by reducing uncertainty, help doctors make treatment or monitoring recommendations, support decisions about personal and family health, encourage preventive actions such as earlier screenings or lifestyle changes, and assist in making future life decisions like family planning. However, test results can also have emotional impacts, such as fear, guilt, anger, anxiety, or depression, and the cost of testing can vary from less than $100 to higher amounts

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.513, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000149-1` (score=0.0656) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxy...
2. `medquad_ans_0000654-2` (score=0.0645) - Microcephaly-capillary malformation syndrome is rare. About a dozen people have been diagnosed with the disorder.
3. `medquad_ans_0000654-5` (score=0.0630) - These resources address the diagnosis or management of microcephaly-capillary malformation syndrome: - Gene Review: Gene Review: Microcephaly-Capillary Malformation Syndrome - Genetic Testing Registry: Microcephaly-ca...
4. `medquad_ans_0000654-1` (score=0.0623) - Microcephaly-capillary malformation syndrome is an inherited disorder characterized by an abnormally small head size (microcephaly) and abnormalities of small blood vessels in the skin called capillaries (capillary ma...
5. `medquad_ans_0000056-1__dup3` (score=0.0623) - Arteriovenous malformations (AVMs) are defects in your vascular system. The vascular system includes arteries, veins, and capillaries. Arteries carry blood away from the heart to other organs; veins carry blood back t...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin. These malformations appear as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs, and may be visible at birth or during childhood. While capillary malformations alone usually do not cause health problems, some individuals with CM-AVM also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs), which are abnormal connections between arteries, veins, and capillaries. These abnormalities can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure, some of which can be life-threatening. Complications of AVMs and AV

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001992-1` (score=0.0757) - What are the signs and symptoms of Dystonia 16? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 16. If the information is available, the table below includes how often the s...
2. `medquad_ans_0000312-1` (score=0.0748) - Early-onset primary dystonia is a condition characterized by progressive problems with movement, typically beginning in childhood. Dystonia is a movement disorder that involves involuntary tensing of the muscles (musc...
3. `medquad_ans_0001903-2` (score=0.0724) - What are the signs and symptoms of Dopa-responsive dystonia? The most common form of dopa-responsive dystonia (DRD) is autosomal dominant DRD (caused by a mutation in the GCH1 gene). This form of DRD is usually charac...
4. `medquad_ans_0002000-1` (score=0.0711) - What are the signs and symptoms of Dystonia 7, torsion? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 7, torsion. If the information is available, the table below includes...
5. `medquad_ans_0000112-3__dup1` (score=0.0708) - The initial symptoms can be very mild and may be noticeable only after prolonged exertion, stress, or fatigue. Dystonias often progress through various stages. Initially, dystonic movements are intermittent and appear...

**Answer**:

The symptoms of Dystonia 16 include cognitive impairment (seen in approximately 5% of patients), abnormal pyramidal signs, bradykinesia, delayed speech and language development, dysarthria, dysphagia, gait disturbance, hyperreflexia, involuntary movements, laryngeal dystonia, limb dystonia, lower limb pain, morphological abnormality of the pyramidal tract, motor delay, parkinsonism, postural tremor, progressive symptoms, retrocollis [medquad_ans_0001992-1].

**Supported citations**: `medquad_ans_0001992-1`
**Unsupported citations**: _none_

### Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Scores**: context_relevance=0.857, groundedness=0.980, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0005548-5` (score=0.0814) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not tr...
2. `medquad_ans_0000899-2` (score=0.0812) - Septo-optic dysplasia has a reported incidence of 1 in 10,000 newborns.
3. `medquad_ans_0005548-2` (score=0.0791) - What are the signs and symptoms of Septo-optic dysplasia? Symptoms may include blindness in one or both eyes, pupil dilation in response to light, nystagmus (a rapid, involuntary to-and-fro movement of the eyes), inwa...
4. `medquad_ans_0005548-1` (score=0.0777) - Septo-optic dysplasia is a disorder of early brain development. The signs and symptoms vary from person to person; however, underdevelopment (hypoplasia) of the optic nerve, abnormal formation of structures along the ...
5. `medquad_ans_0000001-3__dup1` (score=0.0762) - When the absence of the septum pellucidum is part of septo-optic dysplasia, the prognosis varies according to the presence and severity of associated symptoms. By itself, absence of the septum pellucidum is not life-t...

**Answer**:

Treatment for septo-optic dysplasia is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not treatable, and vision, physical, and occupational therapies may be required [medquad_ans_0005548-5].

**Supported citations**: `medquad_ans_0005548-5`
**Unsupported citations**: _none_

### Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Scores**: context_relevance=0.900, groundedness=0.834, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000208-1__dup4` (score=0.0656) - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. Benign prostatic hyperplasia is also called benign prostatic hypertrophy or benign prostatic ...
2. `medquad_ans_0000058-19` (score=0.0645) - Yes. Any of the symptoms caused by prostate cancer may also be due to enlargement of the prostate, which is not cancer. If you have any of the symptoms mentioned in question #10, see your doctor or a urologist to find...
3. `medquad_ans_0000036_3-1` (score=0.0635) - Key Points - Prostate cancer is a disease in which malignant (cancer) cells form in the tissues of the prostate. - Prostate cancer is the most common nonskin cancer among men in the United States. - Different factors ...
4. `medquad_ans_0000752-1__dup1` (score=0.0618) - The prostate is a gland in men. It helps make semen, the fluid that contains sperm. The prostate surrounds the tube that carries urine away from the bladder and out of the body. A young man's prostate is about the siz...
5. `medquad_ans_0000208-14` (score=0.0616) - - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. - The prostate is a walnut-shaped gland that is part of the male reproductive system. - The...

**Answer**:

The complications of prostate enlargement, specifically benign prostatic hyperplasia (BPH), include narrowing of the urethra and urinary retention—the inability to empty the bladder completely. This can lead to a weakened bladder wall and incomplete bladder emptying, which may result in residual urine remaining in the bladder [medquad_ans_0000208-1__dup4]. These issues contribute to the various symptoms associated with BPH [medquad_ans_0000208-1__dup4].

**Supported citations**: `medquad_ans_0000208-1__dup4`
**Unsupported citations**: _none_

### Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Scores**: context_relevance=0.444, groundedness=0.000, answer_relevance=0.711

**Top docs**:

1. `medquad_ans_0005548-5` (score=0.0704) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not tr...
2. `medquad_ans_0000145-3__dup1` (score=0.0633) - How might adenoameloblastoma be treated? Treatment may require the removal of the legion as well as the surrounding tissues. Once the treatment is complete, recurrence of the legion is very rare.
3. `medquad_ans_0004257-3` (score=0.0604) - How might myelomeningocele be treated? A child with meningomyelocele usually has surgery to close the myelomenigocele shortly after birth. This prevents infections and helps save the spinal cord from more damage.[5181...
4. `medquad_ans_0002962-5` (score=0.0594) - How might hydrocephalus due to congenital stenosis of aqueduct of sylvius be treated? The treatment of hydrocephalus due to congenital stenosis of aqueduct of sylvius (HSAS) is based on the signs and symptoms present ...
5. `medquad_ans_0004784-5` (score=0.0589) - How might Pelizaeus-Merzbacher disease be treated?

**Answer**:

The provided context does not contain information about the treatments for absence of the septum pellucidum. Therefore, a definitive answer cannot be given based on the available information. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Scores**: context_relevance=1.000, groundedness=0.511, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.0820) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (...
2. `medquad_ans_0002322-4` (score=0.0806) - How is febrile ulceronecrotic Mucha-Habermann disease definitively diagnosed? FUMHD is diagnosed based upon the clinical symptoms in the patient, with confirmation by skin biopsy. Skin biopsy findings suggestive of FU...
3. `medquad_ans_0002322-1` (score=0.0794) - Febrile ulceronecrotic Mucha-Habermann disease (FUMHD) is a rare and severe form of pityriasis lichenoides et varioliformis acuta (PLEVA). PLEVA is characterized by skin lesions that ulcerate, breakdown, form open sor...
4. `medquad_ans_0002322-5` (score=0.0776) - How is febrile ulceronecrotic Mucha-Habermann disease (FUMHD) treated? It is important that FUMHD is diagnosed and treated as soon as possible. While a number of treatments have been tried, it is hard to asses the ben...
5. `medquad_ans_0002322-3` (score=0.0767) - What causes febrile ulceronecrotic Mucha-Habermann disease? The cause of FUMHD is not known (idiopathic). A hypersensitivity to an infectious agent is suggested to be the main cause. Single cases of people with FUMHD ...

**Answer**:

The symptoms of Febrile Ulceronecrotic Mucha-Habermann disease (FUMHD) include red scaly skin lesions (papules) that ulcerate, break down, form open sores, and then develop a red-brown crust (i.e., PLEVA). These lesions can progress to large, destructive ulcers with extensive, painful loss of skin tissue. Skin lesions may become infected, leading to pus and a putrid odor. FUMHD is often associated with high fever (up to 104°F), which may be persistent or intermittent. Other symptoms may include feeling ill, sore throat, congestion, muscle soreness or pain, joint pain, diarrhea, central nervous system symptoms, abdominal pain, enlarged spleen, arthritis, megaloblastic anemia, interstitial pneumonitis, lymphocytic (viral) myocarditis, and sepsis. The disease can become life threatening

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Scores**: context_relevance=0.500, groundedness=0.196, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003951-3` (score=0.0328) - Is Merkel cell carcinoma inherited? To our knowledge, there currently is no evidence that Merkel cell carcinoma (MCC) is inherited. While DNA changes (mutations) found in the cells of MCC tumors can lead to MCC, these...
2. `medquad_ans_0000167-5__dup1` (score=0.0323) - Autosomal dominant PKD is the most common form of PKD and the most common inherited disorder of the kidneys.3 The term autosomal dominant means a child can get the disorder by inheriting the gene mutation from only on...
3. `medquad_ans_0000615-4` (score=0.0317) - Maffucci syndrome is not inherited. The mutations that cause this disorder are somatic, which means they occur during a person's lifetime. A somatic mutation occurs in a single cell. As that cell continues to grow and...
4. `medquad_ans_0000167-2__dup3` (score=0.0312) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up...
5. `medquad_ans_0001090-4` (score=0.0287) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain information about hereditary leiomyomatosis and renal cell cancer (HLRCC). Therefore, based on the given context, it is not possible to determine whether HLRCC is inherited.

Answer: Insufficient context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000389-2` (score=0.0820) - Frontometaphyseal dysplasia is a rare disorder; only a few dozen cases have been reported worldwide.
2. `medquad_ans_0000389-1` (score=0.0799) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which al...
3. `medquad_ans_0000389-5` (score=0.0789) - These resources address the diagnosis or management of frontometaphyseal dysplasia: - Gene Review: Gene Review: Otopalatodigital Spectrum Disorders - Genetic Testing Registry: Frontometaphyseal dysplasia These resourc...
4. `medquad_ans_0002445-1` (score=0.0778) - What are the signs and symptoms of Frontometaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontometaphyseal dysplasia. If the information is available, the tabl...
5. `medquad_ans_0000390-2` (score=0.0767) - Frontonasal dysplasia is likely a rare condition; at least 100 cases have been reported in the scientific literature.

**Answer**:

Symptoms of Frontometaphyseal dysplasia include abnormalities in skeletal development, joint contractures that restrict joint movement, bowed limbs, abnormal curvature of the spine (scoliosis), and abnormalities of the fingers and hands. Characteristic facial features may include prominent brow ridges, wide-set and downward-slanting eyes, a very small lower jaw and chin (micrognathia), and small, missing, or misaligned teeth. Hearing loss may occur due to malformations of the ossicles in the ears. Some individuals may also have obstruction of the ureters, heart defects, or constrictions in the bronchi leading to breathing problems. Males generally have more severe signs and symptoms than females, who may present with only the characteristic facial features. Abnormality of dental morphology is present in approximately 90% of patients [medquad_ans_0000389-1]medquad_ans_0

**Supported citations**: `medquad_ans_0000389-1`
**Unsupported citations**: _none_

### Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Scores**: context_relevance=0.818, groundedness=0.870, answer_relevance=0.527

**Top docs**:

1. `medquad_ans_0000128-7` (score=0.0805) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding,...
2. `medquad_ans_0000128-11` (score=0.0804) - - Diverticular disease is more common in people as they grow older. - A low-fiber diet is the most likely cause of the disease. - Most people are treated with a high-fiber diet and pain medication. - Add whole grain f...
3. `medquad_ans_0000128-5__dup1` (score=0.0774) - Many people get diverticular disease. Starting at age 40, the chance of getting it increases about every 10 years. About half of people between the ages of 60 and 80 have diverticular disease. Almost everyone over 80 ...
4. `medquad_ans_0000128-2__dup2` (score=0.0772) - Doctors are not sure what causes diverticular disease. Many think a diet low in fiber is the main cause. Fiber is a part of food that your body cannot digest. It is found in many fruits and vegetables. Fiber stays in ...
5. `medquad_ans_0000097-11` (score=0.0760) - - Diverticular disease is a condition that occurs when a person has problems from small pouches, or sacs, that have formed and pushed outward through weak spots in the colon wall. The problems that occur with divertic...

**Answer**:

Treatment for diverticular disease depends on the severity of the condition and whether the person has diverticulosis or diverticulitis. Most people improve with dietary changes, particularly by adopting a high-fiber diet that includes whole grain foods, high-fiber fruits, and vegetables. This helps soften stools and reduces straining during bowel movements, which can prevent the formation of diverticula or the development of diverticulitis [medquad_ans_0000128-11].

If a person experiences symptoms such as fever, chills, nausea, vomiting, abdominal pain, rectal bleeding, or a change in bowel habits, they should contact a doctor [medquad_ans_0000128-11]. In cases of rectal bleeding, a hospital visit is necessary so a doctor can identify the source and may use a special drug to stop the bleeding or, in some cases, perform surgery to remove

**Supported citations**: `medquad_ans_0000128-11`
**Unsupported citations**: _none_

### Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.608, answer_relevance=0.756

**Top docs**:

1. `medquad_ans_0004414-1` (score=0.0809) - Nicolaides-Baraitser syndrome (NCBRS) is a very rare condition characterized by severe intellectual disability and various physical features. Signs and symptoms may include seizures, short stature, sparse hair, distin...
2. `medquad_ans_0000730-2` (score=0.0763) - Nicolaides-Baraitser syndrome is likely a rare condition; approximately 75 cases have been reported in the scientific literature.
3. `medquad_ans_0004414-2` (score=0.0671) - What are the signs and symptoms of Nicolaides-Baraitser syndrome? Nicolaides-Baraitser syndrome (NCBRS) is typically characterized by intellectual disability, seizures, short stature, sparse hair, distinctive facial f...
4. `medquad_ans_0000895-2` (score=0.0650) - Schwartz-Jampel syndrome appears to be a rare condition. About 150 cases have been reported in the medical literature.
5. `medquad_ans_0000107-2` (score=0.0575) - Baraitser-Winter syndrome is a rare condition. Fewer than 50 cases have been reported in the medical literature.

**Answer**:

The provided context does not mention "Baraitser Brett Piesowicz syndrome." The context refers to "Nicolaides-Baraitser syndrome" (NCBRS), which is a rare condition characterized by intellectual disability, seizures, short stature, sparse hair, distinctive facial features, brachydactyly, and prominent interphalangeal joints [medquad_ans_0004414-1][medquad_ans_0004414-2]. There is no information about "Baraitser Brett Piesowicz syndrome" in the provided texts.

Answer: The context does not contain information about the symptoms of Baraitser Brett Piesowicz syndrome. [no context]

**Supported citations**: `medquad_ans_0004414-1`, `medquad_ans_0004414-2`
**Unsupported citations**: `no context`

### Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0003783-6` (score=0.0768) - How might mandibulofacial dysostosis with microcephaly be treated? Individualized treatment of craniofacial features is managed by a multidisciplinary team which may include various specialists. Surgery may be needed ...
2. `medquad_ans_0000986-5__dup1` (score=0.0635) - How might catamenial pneumothorax be treated? Treatment of choice is with surgery, with video-assisted thoracoscopic surgery (VATS). Conventional thoracotomy may be occasionally necessary, particularly in repeat opera...
3. `medquad_ans_0002753-4` (score=0.0628) - What treatment is available for meningeal hemangiopericytoma? Radical surgical resection with removal of all meningeal attachments is typically the preferred treatment. However this treatment option is generally possi...
4. `medquad_ans_0002549-2` (score=0.0570) - What treatment might be available for someone who has had recurrence of gigantomastia following a breast reduction? Breast reduction with or without hormonal therapy is often the first line of treatment for women who ...
5. `medquad_ans_0003746-3` (score=0.0569) - How might Maffucci syndrome be treated? Management aims at relief of symptoms and early detection of malignancies. Individuals with Maffucci syndrome may benefit from consultations with the following specialists: Radi...

**Answer**:

The provided context does not contain information about the treatments for meningioma. Therefore, a specific answer cannot be generated based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Scores**: context_relevance=1.000, groundedness=0.160, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000186-1__dup2` (score=0.0709) - Brain tumors are abnormal growths inside the skull. They are among the most common types of childhood cancers. Some are benign tumors, which aren't cancer. They can still be serious. Malignant tumors are cancerous. Ch...
2. `medquad_ans_0000144-1__dup2` (score=0.0690) - Cancer begins in the cells, which are the building blocks of your body. Normally, new cells form as you need them, replacing old cells that die. Sometimes, this process goes wrong. New cells form when you don't need t...
3. `medquad_ans_0000127-1__dup3` (score=0.0659) - A brain tumor is a growth of abnormal cells in the tissues of the brain. Brain tumors can be benign, with no cancer cells, or malignant, with cancer cells that grow quickly. Some are primary brain tumors, which start ...
4. `medquad_ans_0000006_7-2` (score=0.0648) - Key Points - Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and s...
5. `medquad_ans_0000006_7-1` (score=0.0638) - Key Points - Childhood central nervous system (CNS) germ cell tumors form from germ cells. - There are different types of childhood CNS germ cell tumors. - Germinomas - Nongerminomas - The cause of most childhood CNS ...

**Answer**:

The provided context does not mention Childhood Central Nervous System Germ Cell Tumors or their specific treatments. Therefore, it is not possible to answer the question based on the given information. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Scores**: context_relevance=1.000, groundedness=0.457, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0000729-2` (score=0.0656) - Neutral lipid storage disease with myopathy is a rare condition; its incidence is unknown.
2. `medquad_ans_0000729-1` (score=0.0645) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the ...
3. `medquad_ans_0004401-1` (score=0.0635) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (my...
4. `medquad_ans_0004401-5` (score=0.0625) - How might neutral lipid storage disease with myopathy be treated? To date, there is no treatment for the underlying metabolic problem. Current therapies include adhering to strict dietary guidelines and utilizing trea...
5. `medquad_ans_0004401-3` (score=0.0611) - What causes neutral lipid storage disease with myopathy? Neutral lipid storage disease with myopathy is caused by mutations in the PNPLA2 gene. This gene provides instructions for making an enzyme called adipose trigl...

**Answer**:

Neutral lipid storage disease with myopathy is a rare condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. This abnormal accumulation leads to muscle weakness (myopathy) due to fat buildup in muscle tissue. Other possible features include a fatty liver, weakened and enlarged heart (cardiomyopathy), inflammation of the pancreas (pancreatitis), reduced thyroid activity (hypothyroidism), and type 2 diabetes mellitus. The condition is caused by mutations in the PNPLA2 gene, which results in impaired function of the adipose triglyceride lipase (ATGL) enzyme, leading to reduced breakdown of triglycerides and their subsequent accumulation in tissues. It is inherited in an autosomal recessive pattern, and there is currently no treatment to correct the underlying metabolic problem. Management focuses on dietary guidelines and symptom-related treatments, with emerging evidence suggesting potential benefit from bezafibrate in reducing lipid accumulation and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Scores**: context_relevance=0.800, groundedness=0.941, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0006546-1` (score=0.0653) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. ...
2. `medquad_ans_0000981-1__dup1` (score=0.0643) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been repo...
3. `medquad_ans_0006546-2` (score=0.0633) - How might a Zika virus infection be treated? There is no vaccine to prevent Zika virus infections, nor is there a specific medicine to treat Zika. Individuals infected with the Zika virus should get plenty of rest, dr...
4. `medquad_ans_0000193-1__dup2` (score=0.0632) - Microcephaly is a medical condition in which the circumference of the head is smaller than normal because the brain has not developed properly or has stopped growing. Microcephaly can be present at birth or it may dev...
5. `medquad_ans_0000174-1__dup3` (score=0.0615) - Chikungunya is a virus that spread by the same kinds of mosquitoes that spread dengue and Zika virus. Rarely, it can spread from mother to newborn around the time of birth. It may also possibly spread through infected...

**Answer**:

Zika virus is a virus spread primarily through the bite of an infected mosquito. It can also be transmitted from a pregnant mother to her child during pregnancy or around the time of birth, through sexual contact with an affected male partner, and through blood transfusions. The illness associated with Zika virus infection is usually mild, with symptoms such as fever, rash, joint pain, and conjunctivitis, which typically last for several days to a week. In rare cases, Zika virus infection has been linked to serious birth defects, including microcephaly and other brain anomalies, as well as Guillain-Barre syndrome. There is currently no vaccine or medicine to treat Zika virus infection, and prevention measures include avoiding mosquito bites, postponing travel to affected areas, and taking precautions for sexual transmission [medquad_ans_0006546-1][medquad_ans_0000981-1__dup1].

**Supported citations**: `medquad_ans_0006546-1`, `medquad_ans_0000981-1__dup1`
**Unsupported citations**: _none_

### Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000019-1__dup2` (score=0.0817) - Broken heart syndrome is a condition in which extreme stress can lead to heart muscle failure. The failure is severe, but often short-term. Most people who experience broken heart syndrome think they may be having a h...
2. `medquad_ans_0000019-3__dup2` (score=0.0804) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack...
3. `medquad_ans_0000019-4__dup2` (score=0.0787) - Broken heart syndrome affects women more often than men. Often, people who experience broken heart syndrome have previously been healthy. Research shows that the traditional risk factors for heart disease may not appl...
4. `medquad_ans_0000019-5__dup1` (score=0.0786) - All chest pain should be checked by a doctor. Because symptoms of broken heart syndrome are similar to those of a heart attack, it is important to seek help right away. Your doctor may not be able to diagnose broken h...
5. `medquad_ans_0000019-8` (score=0.0767) - Researchers are still learning about broken heart syndrome, and no treatments have been shown to prevent it. For people who have experienced the condition, the risk of recurrence is low. An emotionally upsetting or se...

**Answer**:

The cause of broken heart syndrome is not fully known. In most cases, symptoms are triggered by extreme emotional or physical stress, such as intense grief, anger, or surprise. Researchers think that the stress releases hormones that stun the heart and affect its ability to pump blood to the body. [medquad_ans_0000019-1__dup2]

**Supported citations**: `medquad_ans_0000019-1__dup2`
**Unsupported citations**: _none_

### Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Scores**: context_relevance=0.800, groundedness=0.962, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000005-9__dup1` (score=0.0635) - - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. - Hashimotos disease often leads to reduced thyroid function, or hypothyroidism. Hypothyroidism is...
2. `medquad_ans_0000005-1__dup5` (score=0.0619) - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. An autoimmune disease is a disorder in which the bodys immune system attacks the bodys own cells and...
3. `medquad_ans_0000009-3__dup4` (score=0.0602) - Hypothyroidism has several causes, including - Hashimotos disease - thyroiditis, or inflammation of the thyroid - congenital hypothyroidism, or hypothyroidism that is present at birth - surgical removal of part or all...
4. `medquad_ans_0000005-3__dup4` (score=0.0580) - Many people with Hashimotos disease have no symptoms at first. As the disease slowly progresses, the thyroid usually enlarges and may cause the front of the neck to look swollen. The enlarged thyroid, called a goiter,...
5. `medquad_ans_0000009-4__dup4` (score=0.0566) - Hypothyroidism has many symptoms that can vary from person to person. Some common symptoms of hypothyroidism are - fatigue - weight gain - a puffy face - cold intolerance - joint and muscle pain - constipation - dry s...

**Answer**:

Hashimoto thyroiditis is an autoimmune disease in which the body's immune system attacks the thyroid gland, causing inflammation and interfering with its ability to produce thyroid hormones. It is also known as chronic lymphocytic thyroiditis or autoimmune thyroiditis. The immune system normally protects the body from infection by identifying and destroying foreign substances, but in Hashimoto thyroiditis, lymphocytes accumulate in the thyroid and produce antibodies that initiate the autoimmune process [medquad_ans_0000005-1__dup5]. This condition often leads to reduced thyroid function, or hypothyroidism, which occurs when the thyroid does not make enough thyroid hormone for the body's needs [medquad_ans_0000005-9__dup1]. Hashimoto thyroiditis is the most common cause of hypothyroidism in the United States [medquad_ans_0000005-1__dup5].

**Supported citations**: `medquad_ans_0000005-1__dup5`, `medquad_ans_0000005-9__dup1`
**Unsupported citations**: _none_

### Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Scores**: context_relevance=0.800, groundedness=0.946, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001043-1` (score=0.0656) - Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). In affected individuals, harmful amounts of lipids accumulate in the spleen, liver, ...
2. `medquad_ans_0006463-1` (score=0.0645) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include...
3. `medquad_ans_0000002-3__dup1` (score=0.0630) - Wolmans disease is usually fatal by age 1. The onset and course of cholesteryl ester storage disease varies, and individuals may live into adulthood.
4. `medquad_ans_0001043-5` (score=0.0625) - These resources address the diagnosis or management of Wolman disease: - Genetic Testing Registry: Lysosomal acid lipase deficiency These resources from MedlinePlus offer information about the diagnosis and management...
5. `medquad_ans_0006463-3` (score=0.0613) - How can I find additional comprehensive information on the treatment of Wolman disease? You can find relevant journal articles on Wolman syndrome and its treatment through a service called PubMed, a searchable databas...

**Answer**:

Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). It is a type of lysosomal storage disorder caused by mutations in the LIPA gene and is inherited in an autosomal recessive manner. The condition leads to a buildup of lipids in body organs such as the spleen, liver, bone marrow, small intestine, adrenal glands, and lymph nodes, along with calcium deposits in the adrenal glands. Infants are typically healthy at birth but soon develop symptoms including enlarged liver and spleen (hepatosplenomegaly), poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor nutrient absorption. The disease is severe and life-threatening, with most affected individuals not surviving past early childhood [medquad_ans_0001043-1]medquad_ans_0006463

**Supported citations**: `medquad_ans_0001043-1`
**Unsupported citations**: _none_

### Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002753-1` (score=0.0812) - Hemangiopericytoma is a term used to described a group of tumors that are derived from pericytes, the cells normally arranged along specific types of blood vessels called capillaries and venules. These types of tumors...
2. `medquad_ans_0002753-3` (score=0.0809) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma a...
3. `medquad_ans_0002750-1` (score=0.0794) - The term hemangioendothelioma describes several types of vascular neosplasms and includes both non-cancerous (benign) and cancerous (malignant) growths. The term has also been applied to those that show "borderline" b...
4. `medquad_ans_0002751-1` (score=0.0777) - Hemangioma thrombocytopenia syndrome is characterized by profound thrombocytopenia in association with two rare vascular tumors: kaposiform hemangioendotheliomas and tufted angiomas. The profound thrombocytopenia can ...
5. `medquad_ans_0002753-4` (score=0.0769) - What treatment is available for meningeal hemangiopericytoma? Radical surgical resection with removal of all meningeal attachments is typically the preferred treatment. However this treatment option is generally possi...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

### Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Scores**: context_relevance=0.625, groundedness=0.052, answer_relevance=0.725

**Top docs**:

1. `medquad_ans_0000127-4__dup2` (score=0.0542) - Family history and ancestry are the two risk factors for thalassemias. Family History Thalassemias are inheritedthat is, the genes for the disorders are passed from parents to their children. If your parents have miss...
2. `medquad_ans_0000396-1__dup2` (score=0.0512) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...
3. `medquad_ans_0000013-12` (score=0.0418) - People can't change the genes they inherit from their parents, but they can change other things to prevent diseases that run in the family. This is good news because many diseases result from a combination of a person...
4. `medquad_ans_0000181-3__dup2` (score=0.0412) - Alagille syndrome is caused by a gene mutation, or defect. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the DNA sequence that makes up a gene. DNA, or deoxyribon...
5. `medquad_ans_0000181-5__dup1` (score=0.0394) - Alagille syndrome occurs in about one of every 30,000 live births.3 The disorder affects both sexes equally and shows no geographical, racial, or ethnic preferences. JAG1 and NOTCH2 gene mutations are inherited in an ...

**Answer**:

The provided context does not contain information about genetic changes related to microphthalmia. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Scores**: context_relevance=0.833, groundedness=0.957, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000028-13` (score=0.0814) - Gout is a form of arthritis that frequently affects joints in the lower part of the body such as the knees, ankles, or toes. The affected joint may become swollen, red, or warm. Attacks usually occur at night. Sometim...
2. `medquad_ans_0000028-18` (score=0.0794) - The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH has more information about gout. Check out the information here. Also, see "Key Words" related to gout.
3. `medquad_ans_0000028-8` (score=0.0791) - Gout is a form of arthritis that causes sudden onset of intense pain and swelling in the joints, which also may be warm and red. Attacks frequently occur at night and can be triggered by stressful events, alcohol or d...
4. `medquad_ans_0000028-1__dup4` (score=0.0779) - Sudden, Intense Joint Pain Gout is a form of arthritis that causes the sudden onset of intense pain and swelling in the joints, which also may be warm and red. Attacks frequently occur at night and can be triggered by...
5. `medquad_ans_0000409-1__dup1` (score=0.0770) - Gout is a common, painful form of arthritis. It causes swollen, red, hot and stiff joints. Gout happens when uric acid builds up in your body. Uric acid comes from the breakdown of substances called purines. Purines a...

**Answer**:

The symptoms of gout include sudden onset of intense pain and swelling in the joints, which may also be warm and red. Attacks frequently occur at night and can be triggered by stressful events, alcohol or drugs, or the presence of another illness. The affected joint may become swollen, red, or warm. Hyperuricemia — high levels of uric acid in the body — is present, as well as the presence of uric acid crystals in joint fluid. More than one attack of acute arthritis may occur, and arthritis may develop in a day, producing a swollen, red, and warm joint. Attacks often affect only one joint, commonly the toe, ankle, or knee. Gout frequently affects joints in the lower part of the body such as the knees, ankles, or toes [medquad_ans_0000028-13][medquad_ans_0000028-8][medquad*ans*

**Supported citations**: `medquad_ans_0000028-13`, `medquad_ans_0000028-8`
**Unsupported citations**: _none_

### Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Scores**: context_relevance=0.333, groundedness=0.032, answer_relevance=0.333

**Top docs**:

1. `medquad_ans_0001903-1` (score=0.0795) - Dopa-responsive dystonia (DRD) is an inherited type of dystonia that typically begins during childhood but may begin in adolescence or adulthood. Depending on the specific type of DRD, specific symptoms can vary. Feat...
2. `medquad_ans_0003335-1` (score=0.0757) - Juvenile-onset dystonia is a form of dystonia, which is a movement disorder characterized by involuntary muscle contractions that cause repetitive movements and/or abnormal postures. The severity and frequency of the ...
3. `medquad_ans_0000704-2` (score=0.0754) - The prevalence of myoclonus-dystonia is unknown. This condition has been described in people worldwide.
4. `medquad_ans_0000312-5` (score=0.0743) - These resources address the diagnosis or management of early-onset primary dystonia: - Gene Review: Gene Review: DYT1 Early-Onset Primary Dystonia - Genetic Testing Registry: Dystonia 1 - MedlinePlus Encyclopedia: Mov...
5. `medquad_ans_0000309-1` (score=0.0741) - Dystonia 6 is one of many forms of dystonia, which is a group of conditions characterized by involuntary movements, twisting (torsion) and tensing of various muscles, and unusual positioning of affected body parts. Dy...

**Answer**:

The provided context does not mention the cause of Dystonia 8. Therefore, based on the given information, the cause of Dystonia 8 cannot be determined. [doc_id: not available]

**Supported citations**: _none_
**Unsupported citations**: `doc_id: not available`

### Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Scores**: context_relevance=0.889, groundedness=0.956, answer_relevance=0.444

**Top docs**:

1. `medquad_ans_0000612-2` (score=0.0489) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unk...
2. `medquad_ans_0003719-1` (score=0.0487) - Lysinuric protein intolerance is a metabolic disorder caused by the body's inability to digest and use the amino acids lysine, arginine, and ornithine. Because the body cannot effectively break down these amino acids,...
3. `medquad_ans_0000974-1` (score=0.0476) - Tetrahydrobiopterin deficiency is a rare disorder characterized by a shortage (deficiency) of a molecule called tetrahydrobiopterin or BH4. This condition alters the levels of several substances in the body, including...
4. `medquad_ans_0000974-2` (score=0.0462) - This condition is rare, affecting an estimated 1 in 500,000 to 1 in 1 million newborns. In most parts of the world, tetrahydrobiopterin deficiency accounts for 1 to 3 percent of all cases of elevated phenylalanine lev...
5. `medquad_ans_0000451-2__dup1` (score=0.0455) - What are the signs and symptoms of Aromatic L-amino acid decarboxylase deficiency? Symptoms, which typically present during the first year of life, include severe developmental delay, weak muscle tone (hypotonia), mus...

**Answer**:

Lysinuric protein intolerance affects an estimated 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations, the condition occurs less frequently, but the exact incidence is unknown [medquad_ans_0000612-2].

**Supported citations**: `medquad_ans_0000612-2`
**Unsupported citations**: _none_

### Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.978, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000170-1__dup3` (score=0.0804) - Renal artery stenosis is the narrowing of one or both renal arteries. Renal means kidney and stenosis means narrowing. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood ...
2. `medquad_ans_0000170-10` (score=0.0801) - - Renal artery stenosis (RAS) is the narrowing of one or both renal arteries. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood vessel that carries blood from the heart ...
3. `medquad_ans_0004939-2` (score=0.0696) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their abilit...
4. `medquad_ans_0005299-2` (score=0.0684) - What are the signs and symptoms of renal nutcracker syndrome? The signs and symptoms of renal nutcracker syndrome and the disease severity can vary from person to person. Some affected people may be asymptomatic while...
5. `medquad_ans_0000038-14` (score=0.0587) - Kidney disease is often called a "silent" disease, because most people have no symptoms in early kidney disease. In fact, you might feel just fine until your kidneys have almost stopped working. Do NOT wait for sympto...

**Answer**:

The first symptoms of renal artery stenosis (RAS) are usually either high blood pressure or decreased kidney function, or both. RAS is often overlooked as a cause of high blood pressure [medquad_ans_0000170-10].

**Supported citations**: `medquad_ans_0000170-10`
**Unsupported citations**: _none_

### Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Scores**: context_relevance=0.800, groundedness=0.782, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000712-1` (score=0.0814) - Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis (NFJS/DPR) represents a rare type of ectodermal dysplasia, a group of about 150 conditions characterized by abnormal development of ectoder...
2. `medquad_ans_0000712-5` (score=0.0794) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis -...
3. `medquad_ans_0005502-5` (score=0.0788) - How might Schnitzler syndrome be treated? The treatment of Schnitzler syndrome is aimed at alleviating the signs and symptoms associated with the condition. The following medications have been used with variable succe...
4. `medquad_ans_0004303-1` (score=0.0782) - Naegeli syndrome belongs to a group of disorders known as ectodermal dysplasias. This condition is characterized by absent fingerprints, thickening of the palms and soles (palmoplantar keratoderma), decreased sweating...
5. `medquad_ans_0004303-4` (score=0.0724) - Is there a treatment for Naegeli syndrome? Treatment for Naegeli syndrome is based on an individual's symptoms. Dry skin can be moisturized with creams. To avoid overheating, affected individuals should wear appropria...

**Answer**:

The provided context does not contain information about the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis (NFJS/DPR). While the context describes the signs and features of the condition, including reticulate hyperpigmentation, palmoplantar keratoderma, nail abnormalities, and other dermatological manifestations, it does not mention any specific treatments. Therefore, based on the available information, the treatments for NFJS/DPR cannot be determined.

[medquad_ans_0000712-5] addresses resources for diagnosis and management of NFJS/DPR, such as genetic testing and MedlinePlus, but does not specify treatments. Thus, the context is insufficient to answer the question about treatments.

**Supported citations**: `medquad_ans_0000712-5`
**Unsupported citations**: _none_

### Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.0804) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0000353-5` (score=0.0801) - These resources address the diagnosis or management of familial hyperaldosteronism: - Genetic Testing Registry: Familial hyperaldosteronism type 1 - Genetic Testing Registry: Familial hyperaldosteronism type 3 - Hormo...
3. `medquad_ans_0000353-1` (score=0.0799) - Familial hyperaldosteronism is a group of inherited conditions in which the adrenal glands, which are small glands located on top of each kidney, produce too much of the hormone aldosterone. Aldosterone helps control ...
4. `medquad_ans_0000353-2` (score=0.0789) - The prevalence of familial hyperaldosteronism is unknown. Familial hyperaldosteronism type II appears to be the most common variety. All types of familial hyperaldosteronism combined account for fewer than 1 out of 10...
5. `medquad_ans_0005138-2` (score=0.0757) - What are the signs and symptoms of Pseudohypoaldosteronism type 2? The most common symptom of pseudohypoaldosteronism type 2 is high blood pressure in adolescents or young adults. In its most severe form, it is associ...

**Answer**:

The symptoms of Familial hyperaldosteronism type III include hypercalciuria (5%), metabolic acidosis (5%), polydipsia (5%), polyuria (5%), adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

**Supported citations**: `medquad_ans_0002259-1`
**Unsupported citations**: _none_

### Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Scores**: context_relevance=1.000, groundedness=0.915, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004530-2` (score=0.0164) - Is olivopontocerebellar atrophy inherited? Olivopontocerebellar atrophy (OPCA) may be associated with conditions that are inherited (genetic), or it may occur sporadically. Genetic forms of OPCA may be inherited in an...
2. `medquad_ans_0005761-1` (score=0.0164) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal...
3. `medquad_ans_0004750-4` (score=0.0161) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is pass...
4. `medquad_ans_0000941-5` (score=0.0161) - These resources address the diagnosis or management of spinocerebellar ataxia type 36: - Ataxia Center at the University of Minnesota: Dominant Spinocerebellar Ataxias - Baylor College of Medicine: Parkinson's Disease...
5. `medquad_ans_0002854-3` (score=0.0159) - How is hereditary sensory neuropathy type IE inherited? Hereditary sensory neuropathy type IE (HSNIE) is inherited in an autosomal dominant manner. This means that having only one changed (mutated) copy of the respons...

**Answer**:

Yes, Spinocerebellar ataxia type 11 (SCA11) is inherited in an autosomal dominant manner [medquad_ans_0005761-1].

**Supported citations**: `medquad_ans_0005761-1`
**Unsupported citations**: _none_

### Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Scores**: context_relevance=1.000, groundedness=0.938, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000503-2` (score=0.0820) - Hypochromic microcytic anemia with iron overload is likely a rare disorder; at least five affected families have been reported in the scientific literature.
2. `medquad_ans_0000503-1` (score=0.0799) - Hypochromic microcytic anemia with iron overload is a condition that impairs the normal transport of iron in cells. Iron is an essential component of hemoglobin, which is the substance that red blood cells use to carr...
3. `medquad_ans_0003040-1` (score=0.0794) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload....
4. `medquad_ans_0000503-5` (score=0.0789) - These resources address the diagnosis or management of hypochromic microcytic anemia with iron overload: - Genetic Testing Registry: Hypochromic microcytic anemia with iron overload These resources from MedlinePlus of...
5. `medquad_ans_0000529-1` (score=0.0765) - Iron-refractory iron deficiency anemia is one of many types of anemia, which is a group of conditions characterized by a shortage of healthy red blood cells. This shortage prevents the blood from carrying an adequate ...

**Answer**:

The symptoms of Hypochromic microcytic anemia with iron overload include pale skin (pallor), tiredness (fatigue), and slow growth. Additionally, the condition leads to abnormally small (microcytic) and pale (hypochromic) red blood cells due to impaired iron transport in cells. Iron that is not used by red blood cells accumulates in the liver, which can impair liver function over time, with liver problems typically becoming apparent in adolescence or early adulthood [medquad_ans_0000503-1].

**Supported citations**: `medquad_ans_0000503-1`
**Unsupported citations**: _none_

### Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Scores**: context_relevance=0.600, groundedness=0.735, answer_relevance=0.720

**Top docs**:

1. `medquad_ans_0000698-1__dup2` (score=0.0328) - Pemphigus is an autoimmune disorder. If you have it, your immune system attacks healthy cells in your skin and mouth, causing blisters and sores. No one knows the cause. Pemphigus does not spread from person to person...
2. `medquad_ans_0000685-4` (score=0.0313) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000349-4` (score=0.0312) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000990-4` (score=0.0299) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000101-4` (score=0.0293) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention "benign chronic pemphigus" or specify whether it is inherited. While the context states that pemphigus is an autoimmune disorder and does not appear to be inherited, it also notes that "some people's genes put them more at risk for pemphigus" [medquad_ans_0000698-1__dup2]. However, this does not confirm inheritance of a specific form like "benign chronic pemphigus." Therefore, based on the available information, it cannot be determined if benign chronic pemphigus is inherited.

Answer: Insufficient context to determine if benign chronic pemphigus is inherited. [medquad_ans_0000698-1__dup2]

**Supported citations**: `medquad_ans_0000698-1__dup2`
**Unsupported citations**: _none_

### Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.955, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000139-2__dup2` (score=0.0199) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...
2. `medquad_ans_0000916-4__dup1` (score=0.0164) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is...
3. `medquad_ans_0000145-2` (score=0.0164) - The prevalence of Camurati-Engelmann disease is unknown. Approximately 200 cases have been reported worldwide.
4. `medquad_ans_0000915-1__dup1` (score=0.0161) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
5. `medquad_ans_0000916-5__dup1` (score=0.0161) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene...

**Answer**:

Yes, Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a mutation in only one copy of the responsible gene in each cell is enough to cause the condition. In some cases, the mutation is inherited from an affected parent, and in other cases, it occurs for the first time in an individual with no family history of the condition (called a de novo mutation) [medquad_ans_0000915-1__dup1].

**Supported citations**: `medquad_ans_0000915-1__dup1`
**Unsupported citations**: _none_

### Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Scores**: context_relevance=0.700, groundedness=0.012, answer_relevance=0.780

**Top docs**:

1. `medquad_ans_0002794-1` (score=0.0489) - Hennekam syndrome is a rare condition that affects the lymphatic system. Signs and symptoms of the condition are generally noticeable at birth and vary significantly from person to person, even within the same family....
2. `medquad_ans_0003466-1` (score=0.0487) - Langerhans cell histiocytosis (LCH) is a disorder that primarily affects children, but is also found in adults of all ages. People with LCH produce too many Langerhans cells or histiocytes, a form of white blood cell ...
3. `medquad_ans_0004454-1` (score=0.0466) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004451-1` (score=0.0464) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0002049-3` (score=0.0455) - What causes Ehlers-Danlos syndrome? Ehlers-Danlos syndrome can be caused by changes (mutations) in several different genes (COL5A1, COL5A2, COL1A1, COL3A1, TNXB, PLOD1, COL1A2, and ADAMTS2). However, the underlying ge...

**Answer**:

The provided context does not contain information about the genetic changes related to histiocytosis-lymphadenopathy plus syndrome. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.052, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000261-4` (score=0.0325) - In most cases of cyclic vomiting syndrome, affected people have no known history of the disorder in their family. However, many affected individuals have a family history of related conditions, such as migraines, irri...
2. `medquad_ans_0000139-2__dup2` (score=0.0323) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...
3. `medquad_ans_0000646-4` (score=0.0320) - Menkes syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome)...
4. `medquad_ans_0000557-4` (score=0.0312) - This condition is generally not inherited but arises from mutations in the body's cells that occur after conception. This alteration is called a somatic mutation and is present only in certain cells. Rarely, this cond...
5. `medquad_ans_0003743-4` (score=0.0308) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition w...

**Answer**:

The provided context does not contain information about whether Noonan syndrome is inherited. Therefore, based on the given documents, it is not possible to determine if Noonan syndrome is inherited. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Scores**: context_relevance=0.625, groundedness=0.772, answer_relevance=0.725

**Top docs**:

1. `medquad_ans_0001093-2` (score=0.0820) - 3-methylglutaconyl-CoA hydratase deficiency is a rare disorder; at least 20 cases have been reported in the scientific literature.
2. `medquad_ans_0001093-5` (score=0.0801) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Aci...
3. `medquad_ans_0001093-1` (score=0.0796) - 3-methylglutaconyl-CoA hydratase deficiency is an inherited condition that causes neurological problems. Beginning in infancy to early childhood, children with this condition often have delayed development of mental a...
4. `medquad_ans_0001093-3` (score=0.0784) - Mutations in the AUH gene cause 3-methylglutaconyl-CoA hydratase deficiency. This gene provides instructions for producing 3-methylglutaconyl-CoA hydratase, an enzyme that is involved in breaking down a protein buildi...
5. `medquad_ans_0001090-2` (score=0.0762) - The exact incidence of 3-hydroxyacyl-CoA dehydrogenase deficiency is unknown; it has been reported in only a small number of people worldwide.

**Answer**:

The provided context does not mention any specific treatments for 3-methylglutaconyl-CoA hydratase deficiency. While it discusses diagnostic resources and symptoms, including elevated levels of 3-methylglutaconic acid and metabolic acidosis, there is no information on therapeutic interventions or management strategies. [medquad_ans_0001093-5] lists resources related to diagnosis and management, but does not specify treatments. Therefore, based on the given context, the treatments for this condition cannot be determined.

**Supported citations**: `medquad_ans_0001093-5`
**Unsupported citations**: _none_

### Example 61 (`0003000-1`)

**Question**: What is (are) Hyperkalemic periodic paralysis ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.0656) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0000494-5` (score=0.0645) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis ...
3. `medquad_ans_0000494-1` (score=0.0625) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to...
4. `medquad_ans_0000126-1__dup1` (score=0.0623) - Familial periodic paralyses are a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. They are characterized by episodes in which the affect...
5. `medquad_ans_0003000-2` (score=0.0620) - What are the signs and symptoms of Hyperkalemic periodic paralysis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hyperkalemic periodic paralysis. If the information is available, ...

**Answer**:

Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms and legs. Episodes tend to increase in frequency until mid-adulthood, after which they occur less frequently. Factors that can trigger attacks include rest after exercise, potassium-rich foods such as bananas and potatoes, stress, fatigue, alcohol, pregnancy, exposure to cold temperatures, certain medications, and periods without food (fasting). Muscle strength usually returns to normal between attacks, although many affected people continue to experience mild stiffness (myotonia), particularly in muscles of the face and hands. Most people with hyperkalemic periodic paralysis have increased levels of potassium in their blood (hyperkalemia) during attacks, which results when the weak or paralyzed muscles release potassium ions into the bloodstream. In other cases, attacks are associated with normal blood potassium levels

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Scores**: context_relevance=0.800, groundedness=0.791, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000225-5` (score=0.0820) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of...
2. `medquad_ans_0000225-1` (score=0.0804) - Congenital fibrosis of the extraocular muscles is a disorder that affects the muscles that surround the eyes. These muscles control eye movement and the position of the eyes (for example, looking straight ahead). Cong...
3. `medquad_ans_0000225-2` (score=0.0796) - CFEOM1 is the most common form of congenital fibrosis of the extraocular muscles, affecting at least 1 in 230,000 people. CFEOM1 and CFEOM3 have been reported worldwide, whereas CFEOM2 has been seen in only a few fami...
4. `medquad_ans_0000375-1` (score=0.0777) - Fibrodysplasia ossificans progressiva (FOP) is a disorder in which muscle tissue and connective tissue such as tendons and ligaments are gradually replaced by bone (ossified), forming bone outside the skeleton (extra-...
5. `medquad_ans_0000374-1` (score=0.0756) - Fibrochondrogenesis is a very severe disorder of bone growth. Affected infants have a very narrow chest, which prevents the lungs from developing normally. Most infants with this condition are stillborn or die shortly...

**Answer**:

The provided context does not specify any treatments for congenital fibrosis of the extraocular muscles. While it mentions resources related to diagnosis and management, such as genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, palliative care, and strabismus management [medquad_ans_0000225-5], it does not detail specific therapeutic interventions for congenital fibrosis of the extraocular muscles. Therefore, based on the available information, the treatments are not explicitly stated.

Answer: The provided context does not specify treatments for congenital fibrosis of the extraocular muscles. [medquad_ans_0000225-5]

**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

### Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Scores**: context_relevance=0.800, groundedness=0.534, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000491-1__dup2` (score=0.0636) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms...
2. `medquad_ans_0000152-3__dup1` (score=0.0476) - Huntingtons disease causes disability that gets worse over time. People with this disease usually die within 15 to 20 years following diagnosis. At this time, no treatment is available to slow, stop or reverse the cou...
3. `medquad_ans_0000052-31` (score=0.0476) - Genetics is one of the most exciting areas of Parkinson's disease research. Studying the genes responsible for inherited cases can help researchers understand both inherited and sporadic cases of the disease. Sporadic...
4. `medquad_ans_0000152-1__dup1` (score=0.0474) - Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. Symptoms of the disease, which gets progressiv...
5. `medquad_ans_0000489-2` (score=0.0469) - Huntington disease affects an estimated 3 to 7 per 100,000 people of European ancestry. The disorder appears to be less common in some other populations, including people of Japanese, Chinese, and African descent.

**Answer**:

Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. It is caused by a mutation in the gene for a protein called huntingtin. Symptoms, which get progressively worse, include uncontrolled movements (called chorea), abnormal body postures, changes in behavior, emotion, judgment, and cognition, impaired coordination, slurred speech, and difficulty feeding and swallowing. The disease typically begins between ages 30 and 50, although an earlier onset form called juvenile HD can occur under age 20. People are born with the defective gene, but symptoms usually don't appear until middle age. If one of your parents has Huntington's disease, you have a 50 percent chance of getting it. There is no cure; medicines can help manage symptoms but cannot slow down or stop the disease. People with HD usually die

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.556

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.0492) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
2. `medquad_ans_0000416-5` (score=0.0479) - These resources address the diagnosis or management of glutamate formiminotransferase deficiency: - Baby's First Test - Genetic Testing Registry: Glutamate formiminotransferase deficiency These resources from MedlineP...
3. `medquad_ans_0002601-2` (score=0.0479) - What are the signs and symptoms of Glutamate formiminotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Glutamate formiminotransferase deficiency. If the inform...
4. `medquad_ans_0000416-1` (score=0.0471) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with ...
5. `medquad_ans_0002601-1` (score=0.0462) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...

**Answer**:

Approximately 20 affected individuals have been identified with glutamate formiminotransferase deficiency [medquad_ans_0000416-2].

**Supported citations**: `medquad_ans_0000416-2`
**Unsupported citations**: _none_

### Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Scores**: context_relevance=0.625, groundedness=0.190, answer_relevance=0.625

**Top docs**:

1. `medquad_ans_0000085-1__dup3` (score=0.0429) - Behcet's syndrome is a disease that involves vasculitis, which is inflammation of the blood vessels. It causes problems in many parts of the body. The most common symptoms are - Sores in the mouth - Sores on the sex o...
2. `medquad_ans_0000089-3__dup2` (score=0.0378) - People at highest risk for Marfan syndrome are those who have a family history of the condition. If you have Marfan syndrome, you have a 50 percent chance of passing the altered gene to each of your children. Marfan s...
3. `medquad_ans_0000115-3__dup3` (score=0.0359) - Mntriers disease is rare. The disease is more common in men than in women. The average age at diagnosis is 55.2
4. `medquad_ans_0000934-1__dup1` (score=0.0359) - Usher syndrome is an inherited disease that causes serious hearing loss and retinitis pigmentosa, an eye disorder that causes your vision to get worse over time. It is the most common condition that affects both heari...
5. `medquad_ans_0000018-13` (score=0.0348) - Sjgren's Syndrome Clinic National Institute of Dental and Craniofacial Research Building 10, Room 1N113 10 Center Drive MSC 1190 Bethesda, MD 20892-1190 301-435-8528 http://www.nidcr.nih.gov/Research/NIDCRLaboratories...

**Answer**:

The provided context does not contain any information about Noonan syndrome or the number of people affected by it. Therefore, the question cannot be answered based on the given context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Scores**: context_relevance=0.571, groundedness=0.989, answer_relevance=0.571

**Top docs**:

1. `medquad_ans_0000339-5__dup1` (score=0.0489) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on t...
2. `medquad_ans_0000339-1__dup3` (score=0.0487) - Also called crab lice or "crabs," pubic lice are parasitic insects found primarily in the pubic or genital area of humans. Pubic lice infestation is found worldwide and occurs in all races, ethnic groups, and levels o...
3. `medquad_ans_0000339-2__dup1` (score=0.0476) - Pubic ("crab") lice infestation is found worldwide and occurs in all races and ethnic groups and in all levels of society. Pubic lice usually are spread through sexual contact and are most common in adults. Occasional...
4. `medquad_ans_0000557-1__dup2` (score=0.0469) - Summary : Lice are parasitic insects that can be found on people's heads and bodies. They survive by feeding on human blood. Lice found on each area of the body are different from each other. The three types of lice t...
5. `medquad_ans_0000339-7` (score=0.0462) - Pubic ("crab") lice most commonly are spread directly from person to person by sexual contact. Pubic lice very rarely may be spread by clothing, bedding, or a toilet seat. The following are steps that can be taken to ...

**Answer**:

Pubic lice infestation is diagnosed by finding a “crab” louse or eggs (nits) on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp) [medquad_ans_0000339-5__dup1]. Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

### Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005571-3` (score=0.0817) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
2. `medquad_ans_0005571-1` (score=0.0809) - Shapiro syndrome is a rare disease affecting about 50 people worldwide that is typically characterized by recurrent episodes of excessive sweating and hypothermia and the agenesis of the corpus callosum. The duration ...
3. `medquad_ans_0005571-2` (score=0.0794) - What are the signs and symptoms of Shapiro syndrome? Shapiro syndrome generally consists of three findings: spontaneous periodic hypothermia, excessive sweating, and agenesis of the corpus callosum. However, there has...
4. `medquad_ans_0000886-1` (score=0.0651) - Rubinstein-Taybi syndrome is a condition characterized by short stature, moderate to severe intellectual disability, distinctive facial features, and broad thumbs and first toes. Additional features of the disorder ca...
5. `medquad_ans_0000650-1` (score=0.0648) - Methylmalonic acidemia is an inherited disorder in which the body is unable to process certain proteins and fats (lipids) properly. The effects of methylmalonic acidemia, which usually appear in early infancy, vary fr...

**Answer**:

Treatment options for Shapiro syndrome that have been attempted include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, and sympathectomy. However, these treatments have been unsuccessful or of doubtful efficacy due to the small number of individuals diagnosed with the condition [medquad_ans_0005571-3]. It is recommended that treatment options be discussed with a healthcare provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

### Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.964, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002359-1` (score=0.0656) - FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. People wi...
2. `medquad_ans_0000373-1` (score=0.0635) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects ...
3. `medquad_ans_0002359-3` (score=0.0632) - How might FG syndrome be treated? Treatment is aimed at addressing the individual symptoms present in each case. This often involves care by a team of providers which may include pediatricians, neurologists, cardiolog...
4. `medquad_ans_0000373-2` (score=0.0630) - The prevalence of FG syndrome is unknown, although several hundred cases have been reported worldwide. Researchers suspect that FG syndrome may be overdiagnosed because many of its signs and symptoms are also seen wit...
5. `medquad_ans_0000373-5` (score=0.0620) - These resources address the diagnosis or management of FG syndrome: - Gene Review: Gene Review: MED12-Related Disorders - Genetic Testing Registry: FG syndrome - Genetic Testing Registry: FG syndrome 2 - Genetic Testi...

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. It is characterized by intellectual disability ranging from mild to severe, hypotonia, constipation and/or anal anomalies, a distinctive facial appearance, broad thumbs and great toes, relative macrocephaly, and abnormalities of the corpus callosum. Medical problems such as heart defects, seizures, undescended testicles, and inguinal hernia have also been reported. The condition is inherited in an X-linked recessive pattern, and mutations in genes such as MED12 (FGS1), FLNA (FGS2), CASK (FGS4), UPF3B (FGS6), and BRWD3 (FGS7) are associated with the disorder [medquad_ans_0002359-1]. Affected individuals often

**Supported citations**: `medquad_ans_0002359-1`
**Unsupported citations**: _none_

### Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006242-1` (score=0.0656) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0000883-2__dup1` (score=0.0597) - What are the signs and symptoms of Buschke Ollendorff syndrome? Buschke Ollendorff syndrome (BOS) is an association of connective tissue nevi and osteopoikilosis (small, round areas of increased bone density). The nev...
3. `medquad_ans_0005502-2` (score=0.0589) - What are the signs and symptoms of Schnitzler syndrome? The signs and symptoms of Schnitzler syndrome vary but may include: Red raised patches of skin (urticaria) that may become itchy Recurrent fevers Join pain and i...
4. `medquad_ans_0000335-2__dup1` (score=0.0563) - What are the signs and symptoms of Andersen-Tawil syndrome? Anderson-Tawil syndrome causes episodes of muscle weakness (periodic paralysis), changes in heart rhythm (arrhythmia), and developmental abnormalities. The m...
5. `medquad_ans_0001335-1` (score=0.0466) - What are the signs and symptoms of Chudley Rozdilsky syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Chudley Rozdilsky syndrome. If the information is available, the table ...

**Answer**:

The symptoms of Tukel syndrome include carpal bone aplasia, carpal synostosis, compensatory chin elevation, congenital fibrosis of extraocular muscles, nonprogressive restrictive external ophthalmoplegia, postaxial oligodactyly, ptosis, restrictive external ophthalmoplegia, and syndactyly [medquad_ans_0006242-1].

**Supported citations**: `medquad_ans_0006242-1`
**Unsupported citations**: _none_

### Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Scores**: context_relevance=0.500, groundedness=0.670, answer_relevance=0.633

**Top docs**:

1. `medquad_ans_0000199-1__dup1` (score=0.0651) - The mucolipidoses (ML) are a group of inherited metabolic diseases that affect the bodys ability to carry out the normal turnover of various materials within cells. In ML, abnormal amounts of carbohydrates and fatty m...
2. `medquad_ans_0000674-5` (score=0.0645) - These resources address the diagnosis or management of mucolipidosis II alpha/beta: - Gene Review: Gene Review: Mucolipidosis II - Genetic Testing Registry: I cell disease - MedlinePlus Encyclopedia: Clubfoot - Medlin...
3. `medquad_ans_0004153-1` (score=0.0633) - Mucolipidosis type 4 is a metabolic condition that affects the body's ability to process certain carbohydrates and fats. As a result, these materials accumulate in cells leading to the various signs and symptoms of th...
4. `medquad_ans_0000677-5` (score=0.0633) - These resources address the diagnosis or management of mucolipidosis type IV: - Gene Review: Gene Review: Mucolipidosis IV - Genetic Testing Registry: Ganglioside sialidase deficiency - MedlinePlus Encyclopedia: Gastr...
5. `medquad_ans_0000676-5` (score=0.0615) - These resources address the diagnosis or management of mucolipidosis III gamma: - Gene Review: Gene Review: Mucolipidosis III Gamma - Genetic Testing Registry: Mucolipidosis III Gamma - MedlinePlus Encyclopedia: Cloud...

**Answer**:

The provided context does not mention any specific treatments for mucolipidoses. While it describes the disease mechanisms, symptoms, and genetic basis, it does not include information on therapeutic interventions or management strategies. Therefore, based on the given context, treatments for mucolipidoses cannot be determined.

[medquad_ans_0000199-1__dup1]

**Supported citations**: `medquad_ans_0000199-1__dup1`
**Unsupported citations**: _none_

### Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.974, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000118-5` (score=0.0807) - These resources address the diagnosis or management of benign recurrent intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Benign recurrent intrahepatic cholestasis 1 -...
2. `medquad_ans_0005084-1` (score=0.0785) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching...
3. `medquad_ans_0000118-1` (score=0.0781) - Benign recurrent intrahepatic cholestasis (BRIC) is characterized by episodes of liver dysfunction called cholestasis. During these episodes, the liver cells have a reduced ability to release a digestive fluid called ...
4. `medquad_ans_0000829-5` (score=0.0777) - These resources address the diagnosis or management of progressive familial intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Progressive familial intrahepatic cholest...
5. `medquad_ans_0001157-1` (score=0.0754) - What are the signs and symptoms of Cholestasis, progressive familial intrahepatic 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Cholestasis, progressive familial intrahepatic 4....

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 (PFIC2) include severe itching, jaundice, failure to thrive, portal hypertension, and hepatosplenomegaly. [medquad_ans_0005084-1]

**Supported citations**: `medquad_ans_0005084-1`
**Unsupported citations**: _none_

### Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Scores**: context_relevance=0.800, groundedness=0.741, answer_relevance=0.720

**Top docs**:

1. `medquad_ans_0000211-1__dup3` (score=0.0487) - Urinary incontinence is the loss of bladder control, resulting in the accidental leakage of urine from the body. For example, a man may feel a strong, sudden need, or urgency, to urinate just before losing a large amo...
2. `medquad_ans_0000211-6` (score=0.0487) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he i...
3. `medquad_ans_0000099-2__dup2` (score=0.0469) - Nearly 18 million U.S. adultsabout one in 12have fecal incontinence.1 People of any age can have a bowel control problem, though fecal incontinence is more common in older adults. Fecal incontinence is slightly more c...
4. `medquad_ans_0000070-7` (score=0.0467) - Urinary incontinence means a person leaks urine by accident. Urinary incontinence is a common bladder problem as people age. Women are more likely than men to leak urine. If this problem is happening to you, there is ...
5. `medquad_ans_0000211-4__dup2` (score=0.0462) - Urinary incontinence in men results when the brain does not properly signal the bladder, the sphincters do not squeeze strongly enough, or both. The bladder muscle may contract too much or not enough because of a prob...

**Answer**:

Urinary incontinence occurs in 11 to 34 percent of older men [medquad_ans_0000211-6]. However, the context does not provide a specific total number of people affected by urinary incontinence in men, only the percentage among older men. Therefore, a precise number of individuals affected cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000211-6`
**Unsupported citations**: _none_

### Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.544, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000882-2` (score=0.0645) - Both the autosomal recessive and autosomal dominant forms of Robinow syndrome are rare. Fewer than 200 people with autosomal recessive Robinow syndrome have been described in the medical literature. This form of the c...
2. `medquad_ans_0000882-4` (score=0.0643) - As discussed above, Robinow syndrome can have either an autosomal recessive or an autosomal dominant pattern of inheritance. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. T...
3. `medquad_ans_0000882-1` (score=0.0640) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
4. `medquad_ans_0000882-5` (score=0.0628) - These resources address the diagnosis or management of Robinow syndrome: - Gene Review: Gene Review: Autosomal Dominant Robinow Syndrome - Gene Review: Gene Review: ROR2-Related Robinow Syndrome - Genetic Testing Regi...
5. `medquad_ans_0005415-1` (score=0.0618) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...

**Answer**:

Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome, distinguished by the severity of signs and symptoms and by their pattern of inheritance—autosomal recessive or autosomal dominant. Autosomal recessive Robinow syndrome is characterized by skeletal abnormalities such as shortening of the long bones in the arms and legs, abnormally short fingers and toes (brachydactyly), wedge-shaped spinal bones (hemivertebrae) leading to spinal curvature (kyphoscoliosis), fused or missing ribs, and short stature. Affected individuals also have distinctive facial features, including a broad forehead, prominent and widely spaced eyes, a short nose with an upturned tip, a wide nasal bridge, and a broad, triangle-shaped mouth, which are sometimes described as "fetal facies." Other features include underdeveloped genitalia in both males

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 74 (`0000551-1`)

**Question**: What is (are) juvenile primary lateral sclerosis ?

**Scores**: context_relevance=0.857, groundedness=0.993, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000551-2` (score=0.0656) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.
2. `medquad_ans_0003331-1` (score=0.0643) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
3. `medquad_ans_0003319-1` (score=0.0632) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. U...
4. `medquad_ans_0000551-5` (score=0.0623) - These resources address the diagnosis or management of juvenile primary lateral sclerosis: - Gene Review: Gene Review: ALS2-Related Disorders - Genetic Testing Registry: Juvenile primary lateral sclerosis These resour...
5. `medquad_ans_0000551-1` (score=0.0618) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. The features of this disorder are caused by damage to moto...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. It is caused by damage to motor neurons, which are specialized nerve cells in the brain and spinal cord that control muscle movement. Symptoms begin in early childhood and progress slowly over many years. Early symptoms include clumsiness, muscle weakness and spasticity in the legs, and difficulty with balance. As symptoms progress, spasticity spreads to the arms and hands, and individuals may develop slurred speech, drooling, difficulty swallowing, and an inability to walk. Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene and is inherited in an autosomal recessive pattern [medquad_ans_0003331-1].

**Supported citations**: `medquad_ans_0003331-1`
**Unsupported citations**: _none_

### Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Scores**: context_relevance=0.750, groundedness=0.958, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000040-9` (score=0.0646) - Leukemia is a cancer of the blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white ...
2. `medquad_ans_0000010-1__dup4` (score=0.0640) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. Thes...
3. `medquad_ans_0000188-1__dup2` (score=0.0628) - Leukemia is cancer of the white blood cells. It is the most common type of childhood cancer. Your blood cells form in your bone marrow. White blood cells help your body fight infection. In leukemia, the bone marrow pr...
4. `medquad_ans_0000040-1__dup3` (score=0.0611) - Leukemia is a cancer of the blood cells. It is the most common type of blood cancer and affects 10 times as many adults as children. Most people diagnosed with leukemia are over 50 years old. Leukemia Starts in Bone M...
5. `medquad_ans_0000009-1__dup3` (score=0.0604) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. Thes...

**Answer**:

Leukemia is a cancer of the white blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white blood cells builds up in the blood, they crowd out healthy blood cells, making it difficult for the blood to carry out its normal functions [medquad_ans_0000040-9]. Leukemia is the most common type of childhood cancer and can develop quickly or slowly, with acute leukemia being a fast-growing type and chronic leukemia growing slowly [medquad_ans_0000188-1__dup2]. In acute myeloid leukemia (AML), there are too many myeloblasts, a specific type of white blood cell, and AML is the most common type of acute leukemia in adults [medquad_ans_0000010-1\_\_

**Supported citations**: `medquad_ans_0000040-9`, `medquad_ans_0000188-1__dup2`
**Unsupported citations**: _none_

### Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Scores**: context_relevance=0.750, groundedness=0.986, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000643-1__dup2` (score=0.0551) - Neuroblastoma is a cancer that forms in your nerve tissue. It usually begins in the adrenal glands, which sit atop your kidneys. It may also begin in your neck, chest or spinal cord. The cancer often begins in early c...
2. `medquad_ans_0000127-1__dup3` (score=0.0528) - A brain tumor is a growth of abnormal cells in the tissues of the brain. Brain tumors can be benign, with no cancer cells, or malignant, with cancer cells that grow quickly. Some are primary brain tumors, which start ...
3. `medquad_ans_0002485-1` (score=0.0492) - Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. Gangliocytomas may occur in all age groups but most often occur in people between the ages of 10 and 30. The most common si...
4. `medquad_ans_0000326-1__dup1` (score=0.0484) - Anaplastic ganglioglioma (AGG) is a very rare type of brain tumor that is a type of ganglioglioma. In general, gangliogliomas are classified as grade I or low grade tumors, meaning that they grow slowly and are consid...
5. `medquad_ans_0002573-1` (score=0.0476) - Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue th...

**Answer**:

Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. It may occur in all age groups but most often occurs in people between the ages of 10 and 30. The most common site is the temporal lobe of the brain, but it can arise anywhere in the CNS, including the cerebellum, brainstem, floor of the third ventricle, and spinal cord. It is among the most frequent tumors associated with epilepsy. Signs and symptoms depend on the tumor's location and may include seizures (most commonly), increased brain pressure, endocrine disorders, and focal symptoms. Gangliocytomas are generally slow-growing and usually do not become malignant. Treatment involves surgical removal of the tumor [medquad_ans_0002485-1].

**Supported citations**: `medquad_ans_0002485-1`
**Unsupported citations**: _none_

### Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.962, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000130-1__dup1` (score=0.0656) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of ...
2. `medquad_ans_0004068-1` (score=0.0633) - Miller syndrome is a rare condition that mainly affects the development of the face and limbs. Characteristic features include underdeveloped cheek bones, a very small lower jaw, cleft lip and/or palate, abnormalities...
3. `medquad_ans_0000658-2` (score=0.0633) - Miller syndrome is a rare disorder; it is estimated to affect fewer than 1 in 1 million newborns. At least 30 cases have been reported in the medical literature.
4. `medquad_ans_0000130-2__dup1` (score=0.0623) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportiv...
5. `medquad_ans_0000130-3__dup1` (score=0.0618) - The prognosis for most individuals with Miller Fisher syndrome is good. In most cases, recovery begins within 2 to 4 weeks of the onset of symptoms, and may be almost complete within 6 months. Some individuals are lef...

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of tendon reflexes. Like Guillain-Barr syndrome, symptoms may be preceded by a viral illness. Additional symptoms include generalized muscle weakness and respiratory failure. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder [medquad_ans_0000130-1__dup1]. Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis and supportive care [medquad_ans_0000130-2__dup1]. The prognosis for most individuals with Miller Fisher syndrome is good, with recovery beginning within 2 to 4 weeks of the onset of symptoms and possibly being almost complete within

**Supported citations**: `medquad_ans_0000130-1__dup1`, `medquad_ans_0000130-2__dup1`
**Unsupported citations**: _none_

### Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0001084-1` (score=0.0812) - 22q11.2 deletion syndrome (which is also known by several other names, listed below) is a disorder caused by the deletion of a small piece of chromosome 22. The deletion occurs near the middle of the chromosome at a l...
2. `medquad_ans_0006570-3` (score=0.0802) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location desig...
3. `medquad_ans_0001087-3` (score=0.0786) - 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at a location designated 2q37. The size...
4. `medquad_ans_0001086-2` (score=0.0782) - At least 500 cases of 22q13.3 deletion syndrome are known.
5. `medquad_ans_0006579-3` (score=0.0768) - What causes 2q37 deletion syndrome? 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at ...

**Answer**:

22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in the region designated q11.2 [medquad_ans_0001084-1].

**Supported citations**: `medquad_ans_0001084-1`
**Unsupported citations**: _none_

### Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Scores**: context_relevance=1.000, groundedness=0.885, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000062-12` (score=0.0812) - Shingles is caused by a virus called the varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores and other symptoms healed, but the virus remained....
2. `medquad_ans_0000062-2__dup3` (score=0.0802) - Caused By A Virus Shingles is caused by a virus called varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores healed and the other symptoms went a...
3. `medquad_ans_0000148-1__dup2` (score=0.0786) - Shingles (herpes zoster) is an outbreak of rash or blisters on the skin that is caused by the same virus that causes chickenpox the varicella-zoster virus. The first sign of shingles is often burning or tingling pain ...
4. `medquad_ans_0000062-7__dup1` (score=0.0781) - Shingles -- also called varicella-zoster -- is a painful skin disease caused by a reactivation of the chickenpox virus. It is distinctive because it affects only one side of the body. The early signs of shingles usual...
5. `medquad_ans_0000062-11` (score=0.0772) - The most common complication of shingles is pain -- a condition called post-herpetic neuralgia (PHN). People with PHN have severe pain in the areas where they had the shingles rash, even after the rash clears up. In m...

**Answer**:

Shingles is caused by the varicella-zoster virus, the same virus that causes chickenpox. After recovering from chickenpox, the virus remains in the body for life, hiding in nerve cells, usually in the spine. It can become active again when the immune system weakens, which may occur due to aging, certain medical conditions (such as cancer, HIV/AIDS, or transplant operations), or stress. When reactivated, the virus travels along a nerve path (dermatome) to the skin, causing a painful rash [medquad_ans_0000062-12][medquad_ans_0000062-2__dup3].

**Supported citations**: `medquad_ans_0000062-12`, `medquad_ans_0000062-2__dup3`
**Unsupported citations**: _none_

### Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.887, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0002456-1` (score=0.0820) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the ta...
2. `medquad_ans_0000333-1` (score=0.0806) - Fabry disease is an inherited disorder that results from the buildup of a particular type of fat, called globotriaosylceramide, in the body's cells. Beginning in childhood, this buildup causes signs and symptoms that ...
3. `medquad_ans_0000333-2` (score=0.0770) - Fabry disease affects an estimated 1 in 40,000 to 60,000 males. This disorder also occurs in females, although the prevalence is unknown. Milder, late-onset forms of the disorder are probably more common than the clas...
4. `medquad_ans_0002458-1` (score=0.0770) - Fryns syndrome is a condition that affects the development of many parts of the body. Signs and symptoms vary widely among affected individuals. Many affected individuals have a defect in the diaphragm muscle such as ...
5. `medquad_ans_0000123-1__dup1` (score=0.0753) - Fabry disease is caused by the lack of or faulty enzyme needed to metabolize lipids, fat-like substances that include oils, waxes, and fatty acids. The disease is also called alpha-galactosidase-A deficiency. A mutati...

**Answer**:

The symptoms of Fryns Hofkens Fabry syndrome include ulnar deviation of the finger (seen in 90% of patients), distal ulnar hypoplasia, dysplastic radii, hypoplasia of the radius, mesomelic arm shortening, radial bowing, and ulnar deviation of the hand. These signs and symptoms are derived from the Human Phenotype Ontology and are based on observations in patients with this condition [medquad_ans_0002456-1].

Note: There is no evidence in the provided context that "Fryns Hofkens Fabry syndrome" is a recognized or valid medical condition. The context separately describes "Fabry disease," which is a distinct inherited disorder involving globotriaosylceramide buildup and symptoms such as acroparesthesias, angiokeratomas, hypohidrosis, corneal opacity, gastrointestinal issues, tinnitus

**Supported citations**: `medquad_ans_0002456-1`
**Unsupported citations**: _none_

### Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Scores**: context_relevance=0.600, groundedness=0.965, answer_relevance=0.360

**Top docs**:

1. `medquad_ans_0000050-8` (score=0.0492) - Scientists are pursuing a wide range of basic and clinical studies on osteoporosis. Significant advances in preventing and treating osteoporosis continue to be made. Such advances are the direct result of research foc...
2. `medquad_ans_0000552-5` (score=0.0484) - These resources address the diagnosis or management of juvenile primary osteoporosis: - Lucile Packard Children's Hospital at Stanford: Juvenile Osteoporosis - MedlinePlus Encyclopedia: Bone Mineral Density Test - Mer...
3. `medquad_ans_0000050-36` (score=0.0464) - Millions of Americans are able to lead healthy, productive lives while living with osteoporosis. If you have been feeling symptoms of depressionsuch as loss of appetite, hopelessness, feeling useless and helpless, or ...
4. `medquad_ans_0000050-29` (score=0.0459) - Diagnosing osteoporosis involves several steps, starting with a physical exam and a careful medical history, blood and urine tests, and possibly a bone mineral density assessment. When recording information about your...
5. `medquad_ans_0000552-1` (score=0.0455) - Juvenile primary osteoporosis is a skeletal disorder characterized by thinning of the bones (osteoporosis) that begins in childhood. Osteoporosis is caused by a shortage of calcium and other minerals in bones (decreas...

**Answer**:

Scientists are pursuing a wide range of basic and clinical studies on osteoporosis, focusing on determining the causes and consequences of bone loss at the cellular and tissue levels, assessing risk factors, developing new strategies to maintain and even enhance bone density and reduce fracture risk, and exploring the roles of genetics, hormones, calcium, vitamin D, drugs, and exercise on bone mass [medquad_ans_0000050-8]. Additional information on ongoing research can be obtained from the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH [medquad_ans_0000050-8].

**Supported citations**: `medquad_ans_0000050-8`
**Unsupported citations**: _none_

### Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Scores**: context_relevance=0.857, groundedness=0.897, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000430-1__dup1` (score=0.0328) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. ...
2. `medquad_ans_0000338-1__dup1` (score=0.0323) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the s...
3. `medquad_ans_0000927-1__dup2` (score=0.0317) - Summary : It seems to happen almost every day - you hear about the results of a new medical research study. Sometimes the results of one study seem to disagree with the results of another study. It's important to be c...
4. `medquad_ans_0000011-14` (score=0.0306) - Some parents did not tell their children about receiving treatment with pituitary hGH and the possible risk of CJD. These children are now adults. Although the HHS no longer sends annual information about the problem ...
5. `medquad_ans_0000705-1__dup1` (score=0.0305) - Summary : You've probably seen your chart at your doctor's office. In fact, you may have charts at several doctors' offices. If you've been in the hospital, you have a chart there, too. These charts are your medical r...

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. These scams often promise quick fixes, miracle cures, ancient remedies, or secret ingredients, which may sound too good to be true. Such scams can be dangerous, waste money, and prevent individuals from accessing effective treatments. They commonly target people with serious health conditions like cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. To protect yourself, recognize red flags and consult a doctor or healthcare professional before taking unproven treatments [medquad_ans_0000430-1__dup1].

**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

### Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Scores**: context_relevance=0.917, groundedness=0.337, answer_relevance=0.583

**Top docs**:

1. `medquad_ans_0000275-1__dup1` (score=0.0814) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
2. `medquad_ans_0000016-6` (score=0.0807) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
3. `medquad_ans_0000015-17` (score=0.0784) - Diabetes is a very serious disease. Over time, diabetes that is not well managed causes serious damage to the eyes, kidneys, nerves, and heart, gums and teeth. If you have diabetes, you are more likely than someone wh...
4. `medquad_ans_0000016-24` (score=0.0765) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even ...
5. `medquad_ans_0000016-8` (score=0.0754) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a com...

**Answer**:

The provided context does not include information on symptoms of preventing diabetes problems or specifically on how to keep eyes healthy in terms of prevention. While it describes symptoms of diabetic retinopathy (such as blurry or double vision, rings or flashing lights, dark or floating spots, pain or pressure in the eyes, and trouble seeing out of the corners of the eyes), these are symptoms of the condition, not preventive measures.

Therefore, based on the given context, there is no information to answer the question about symptoms of preventing diabetes problems or keeping eyes healthy.

Answer: Insufficient context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.042, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0004549-4` (score=0.0164) - How is Opitz G/BBB syndrome inherited? Opitz G/BBB syndrome often has an X-linked pattern of inheritance. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome,...
2. `medquad_ans_0004784-4` (score=0.0164) - How is Pelizaeus-Merzbacher disease inherited?
3. `medquad_ans_0000585-1__dup1` (score=0.0161) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
4. `medquad_ans_0006446-4` (score=0.0161) - Is Williams syndrome inherited?
5. `medquad_ans_0001143-1` (score=0.0159) - CHILD syndrome, also known as congenital hemidysplasia with ichthyosiform erythroderma and limb defects, is a genetic condition that is typically characterized by large patches of skin that are red and inflamed (eryth...

**Answer**:

The provided context does not contain information about the inheritance of Aarskog-Scott syndrome. Therefore, it is not possible to determine how Aarskog-Scott syndrome is inherited based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Scores**: context_relevance=0.800, groundedness=0.958, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0006525-1` (score=0.0653) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin. People with X-linked sideroblastic anemia have mature red blood cells that...
2. `medquad_ans_0001063-2` (score=0.0643) - X-linked sideroblastic anemia and ataxia is a rare disorder; only a few affected families have been reported.
3. `medquad_ans_0001062-1` (score=0.0633) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin, which is the protein that carries oxygen in the blood. People with X-linke...
4. `medquad_ans_0001063-1` (score=0.0625) - X-linked sideroblastic anemia and ataxia is a rare condition characterized by a blood disorder called sideroblastic anemia and movement problems known as ataxia. This condition occurs only in males. Sideroblastic anem...
5. `medquad_ans_0001062-4` (score=0.0616) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...

**Answer**:

The genetic changes related to X-linked sideroblastic anemia are mutations in the ALAS2 gene. In rare cases, mutations in both the HFE gene and the ALAS2 gene are found, resulting in a more severe form of the disorder. [medquad_ans_0006525-1]

**Supported citations**: `medquad_ans_0006525-1`
**Unsupported citations**: _none_

### Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Scores**: context_relevance=0.833, groundedness=0.934, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0003601-1` (score=0.0656) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
2. `medquad_ans_0000602-2` (score=0.0628) - It is difficult to determine the prevalence of limb-girdle muscular dystrophy because its features vary and overlap with those of other muscle disorders. Prevalence estimates range from 1 in 14,500 to 1 in 123,000 ind...
3. `medquad_ans_0001935-1` (score=0.0614) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystroph...
4. `medquad_ans_0000602-1` (score=0.0593) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifi...
5. `medquad_ans_0000306-1` (score=0.0584) - Muscular dystrophies are a group of genetic conditions characterized by progressive muscle weakness and wasting (atrophy). The Duchenne and Becker types of muscular dystrophy are two related conditions that primarily ...

**Answer**:

Limb-girdle muscular dystrophy is a group of disorders that affect the voluntary muscles around the hips and shoulders, leading to progressive loss of muscle strength and bulk over time. The conditions primarily impact the proximal muscles, including those in the shoulders, upper arms, pelvic area, and thighs. Onset may occur in childhood, adolescence, young adulthood, or later in life, and the severity, age of onset, and features vary among the many subtypes. Most forms are inherited in an autosomal recessive manner, while several rare forms follow an autosomal dominant pattern. Males and females are affected equally. There is no treatment that directly reverses muscle weakness, but supportive care can help reduce complications. There are at least 20 different types of limb-girdle muscular dystrophy [medquad_ans_0003601-1]. [medquad_ans_0001935-1]

**Supported citations**: `medquad_ans_0003601-1`, `medquad_ans_0001935-1`
**Unsupported citations**: _none_

### Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Scores**: context_relevance=1.000, groundedness=0.907, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000016-6` (score=0.0650) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
2. `medquad_ans_0000275-1__dup1` (score=0.0648) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
3. `medquad_ans_0000055-3__dup2` (score=0.0635) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision -...
4. `medquad_ans_0000016-8` (score=0.0623) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a com...
5. `medquad_ans_0000016-1__dup3` (score=0.0609) - Can Cause Vision Loss, Blindness Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina in the back of the eye. A ...

**Answer**:

Diabetic eye problems include diabetic retinopathy, cataracts, and glaucoma. Diabetic retinopathy is a condition that occurs when diabetes damages the tiny blood vessels in the retina, the light-sensitive tissue at the back of the eye, and is a leading cause of blindness in American adults [medquad_ans_0000275-1__dup1]. Cataracts involve a clouding of the lens of the eye, which can be treated with surgery to restore clear vision [medquad_ans_0000275-1__dup1]. Glaucoma occurs when pressure builds up in the eye, damaging the main nerve, and can be managed with eye drops or surgery [medquad_ans_0000275-1__dup1]. All people with diabetes, both type 1 and type 2, are at risk for these conditions, and a comprehensive dilated eye

**Supported citations**: `medquad_ans_0000275-1__dup1`
**Unsupported citations**: _none_

### Example 88 (`0006449-1`)

**Question**: What is (are) Wilson disease ?

**Scores**: context_relevance=1.000, groundedness=0.887, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000133-6` (score=0.0606) - The signs and symptoms of Wilson disease vary, depending on what organs of the body are affected. Wilson disease is present at birth; however, the signs and symptoms of the disease do not appear until the copper build...
2. `medquad_ans_0000133-12` (score=0.0600) - A person cannot prevent Wilson disease; however, people with a family history of Wilson disease, especially those with an affected sibling or parent, should talk with a health care provider about testing. A health car...
3. `medquad_ans_0000133-1__dup3` (score=0.0587) - Wilson disease is a genetic disease that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filt...
4. `medquad_ans_0000133-3__dup2` (score=0.0574) - Wilson disease is caused by an inherited autosomal recessive mutation, or change, in the ATP7B gene. In an autosomal recessive disease, the child has to inherit the gene mutation from both parents to have an increased...
5. `medquad_ans_0000133-8` (score=0.0557) - People who have Wilson disease that is not treated or diagnosed early can have serious complications, such as - cirrhosisscarring of the liver - kidney damageas liver function decreases, the kidneys may be damaged - p...

**Answer**:

Wilson disease is a genetic disorder present at birth characterized by the accumulation of copper in the liver, brain, or other organs due to impaired copper excretion. The signs and symptoms vary depending on which organs are affected and may include liver-related issues such as weakness, fatigue, loss of appetite, nausea, vomiting, jaundice, and spider angiomas; central nervous system-related symptoms like tremors, muscle stiffness, and difficulties with speech or coordination; mental health-related symptoms such as personality changes, depression, anxiety, and psychosis; and other symptoms including anemia and edema [medquad_ans_0000133-6]. Symptoms may not appear until copper builds up significantly, and some individuals may remain asymptomatic until diagnosed during a routine physical exam, blood test, or illness.

**Supported citations**: `medquad_ans_0000133-6`
**Unsupported citations**: _none_

### Example 89 (`0001503-2`)

**Question**: What are the symptoms of Congenital dyserythropoietic anemia type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0001503-2` (score=0.0820) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to abs...
2. `medquad_ans_0001503-1` (score=0.0799) - Congenital dyserythropoietic anemia type 2 (CDA II) is an inherited blood disorder characterized by mild to severe anemia. It is usually diagnosed in adolescence or early adulthood. Many affected individuals have yell...
3. `medquad_ans_0000307-2` (score=0.0777) - Dyserythropoietic anemia and thrombocytopenia is a rare condition; its prevalence is unknown. Occasionally, individuals with this disorder are mistakenly diagnosed as having more common blood disorders, making it even...
4. `medquad_ans_0000307-1` (score=0.0774) - Dyserythropoietic anemia and thrombocytopenia is a condition that affects blood cells and primarily occurs in males. A main feature of this condition is a type of anemia called dyserythropoietic anemia, which is chara...
5. `medquad_ans_0000223-1` (score=0.0751) - Congenital dyserythropoietic anemia (CDA) is an inherited blood disorder that affects the development of red blood cells. This disorder is one of many types of anemia, which is a condition characterized by a shortage ...

**Answer**:

The symptoms of Congenital dyserythropoietic anemia type 2 (CDA II) include jaundice, gallstones, an enlarged liver and spleen (hepatosplenomegaly), and anemia of inadequate production. The condition also causes iron overload, which can lead to complications such as an abnormal heart rhythm (arrhythmia), congestive heart failure, diabetes, and chronic liver disease (cirrhosis). Rarely, mediastinal tumors may occur [medquad_ans_0001503-2]. Additionally, reticulocytosis and reduced activity of N-acetylglucosaminyltransferase II are associated with the condition [medquad_ans_0001503-2].

**Supported citations**: `medquad_ans_0001503-2`
**Unsupported citations**: _none_

### Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.911, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000038-11` (score=0.0656) - Kidney disease -- also known as chronic kidney disease (CKD) -- occurs when kidneys can no longer remove wastes and extra water from the blood or perform other functions as they should. According to the Centers for Di...
2. `medquad_ans_0000140-2__dup2` (score=0.0638) - Chronic kidney disease is kidney damage that occurs slowly over many years, often due to diabetes or high blood pressure. Once damaged, the kidneys cant filter blood as they should. This damage can cause wastes to bui...
3. `medquad_ans_0000038-12` (score=0.0623) - Kidney disease is most often caused by diabetes or high blood pressure. Each kidney contains about one million tiny filters made up of blood vessels. These filters are called glomeruli. Diabetes and high blood pressur...
4. `medquad_ans_0000038-3__dup2` (score=0.0623) - Kidney Disease Kidney disease is often called a "silent" disease, because most people have no symptoms with early kidney disease. In fact, you might feel just fine until your kidneys have almost stopped working. Do NO...
5. `medquad_ans_0000033-1__dup4` (score=0.0618) - Diabetic kidney disease, also called diabetic nephropathy, is kidney disease caused by diabetes. Even when well controlled, diabetes can lead to chronic kidney disease (CKD) and kidney failure, described as end-stage ...

**Answer**:

Kidney disease, also known as chronic kidney disease (CKD), occurs when kidneys can no longer remove wastes and extra water from the blood or perform other functions as they should [medquad_ans_0000038-11]. It is often called a "silent" disease because most people have no symptoms in the early stages [medquad_ans_0000038-3__dup2]. Chronic kidney disease is characterized by slow kidney damage over many years, typically due to diabetes or high blood pressure [medquad_ans_0000140-2__dup2]. This damage affects the glomeruli, the tiny filters in the kidneys made up of blood vessels, impairing the kidneys' ability to filter blood effectively [medquad_ans_0000038-12]. As more filters are damaged, the kidneys eventually stop working, leading to kidney failure, where less

**Supported citations**: `medquad_ans_0000038-11`, `medquad_ans_0000038-3__dup2`, `medquad_ans_0000140-2__dup2`, `medquad_ans_0000038-12`
**Unsupported citations**: _none_

### Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Scores**: context_relevance=0.857, groundedness=0.953, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000058-23` (score=0.0787) - There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But ...
2. `medquad_ans_0000058-6__dup1` (score=0.0782) - Choosing Treatment There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade...
3. `medquad_ans_0000058-26` (score=0.0771) - Surgery is a common treatment for early stage prostate cancer. It is used to remove the cancer. The surgeon may remove the entire prostate -- a type of surgery called radical prostatectomy -- or, in some cases, remove...
4. `medquad_ans_0000750-1__dup1` (score=0.0724) - The prostate is the gland below a man's bladder that produces fluid for semen. Prostate cancer is common among older men. It is rare in men younger than 40. Risk factors for developing prostate cancer include being ov...
5. `medquad_ans_0000058-24` (score=0.0718) - Surgery, radiation therapy, and hormonal therapy all have the potential to disrupt sexual desire or performance for a short while or permanently. Discuss your concerns with your health care provider. Several options a...

**Answer**:

The treatments for prostate cancer may include watchful waiting, surgery, radiation therapy, or hormonal therapy. Some men receive a combination of these therapies. The choice of treatment depends on the stage of the disease and the grade of the tumor, as well as the man's age, general health, and feelings about the treatments and their possible side effects [medquad_ans_0000058-23]. A cure is probable for men whose prostate cancer is diagnosed early [medquad_ans_0000058-23]. Surgery, such as radical prostatectomy, may be used to remove the cancer, and can sometimes preserve sexual function [medquad_ans_0000058-6__dup1]. Radiation therapy and hormonal therapy also have potential side effects, including impacts on sexual desire or performance and urinary function [medquad_ans_0000058-6__dup1].

**Supported citations**: `medquad_ans_0000058-23`, `medquad_ans_0000058-6__dup1`
**Unsupported citations**: _none_

### Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Scores**: context_relevance=0.857, groundedness=0.913, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000431-1__dup1` (score=0.0328) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...
2. `medquad_ans_0000036-2__dup3` (score=0.0323) - Health insurance helps pay for medical care, including the cost of diabetes care. Health insurance options include the following: - private health insurance, which includes group and individual health insurance - gove...
3. `medquad_ans_0000036-4__dup3` (score=0.0317) - Insurance companies sell private health insurance plans. Two types of private health insurance are - Group health insurance. People may be eligible to purchase group health insurance through their employer or union or...
4. `medquad_ans_0000045-9` (score=0.0312) - Medicare Part B is medical insurance. It helps cover medical services such as doctor's services, outpatient care and other medical services that Part A doesn't cover. Part B also covers some preventive services, such ...
5. `medquad_ans_0000705-1__dup1` (score=0.0305) - Summary : You've probably seen your chart at your doctor's office. In fact, you may have charts at several doctors' offices. If you've been in the hospital, you have a chart there, too. These charts are your medical r...

**Answer**:

Yes, there is information about health insurance. Health insurance helps protect individuals from high medical care costs by providing coverage for expenses when medical care is needed. It is a contract between the individual and an insurance company, where the company agrees to pay part of the expenses [medquad_ans_0000431-1__dup1]. Health insurance options include private health insurance (such as group and individual plans) and government health insurance (e.g., Medicare, Medicaid, CHIP, TRICARE, and veterans health care programs) [medquad_ans_0000036-2__dup3]. Private health insurance can be obtained through employers, unions, professional organizations, or purchased individually [medquad_ans_0000036-4__dup3]. The Affordable Care Act (ACA) prevents insurers from denying coverage or charging higher premiums to individuals with preexisting conditions like diabetes and requires most people to have

**Supported citations**: `medquad_ans_0000431-1__dup1`, `medquad_ans_0000036-2__dup3`, `medquad_ans_0000036-4__dup3`
**Unsupported citations**: _none_

### Example 93 (`0003860-1`)

**Question**: What is (are) MECP2 duplication syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.329, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003860-1` (score=0.0656) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...
2. `medquad_ans_0000635-2` (score=0.0643) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of...
3. `medquad_ans_0000635-1` (score=0.0632) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infan...
4. `medquad_ans_0000635-5` (score=0.0621) - These resources address the diagnosis or management of MECP2 duplication syndrome: - Cincinnati Children's Hospital: MECP2-Related Disorders - Cleveland Clinic: Spasticity - Gene Review: Gene Review: MECP2 Duplication...
5. `medquad_ans_0000635-3` (score=0.0618) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...

**Answer**:

MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia, delayed motor milestones (e.g., sitting up, crawling), recurrent infections, poor or absent speech, seizures, and/or spasticity. The condition is caused by an extra copy (duplication) of the MECP2 gene in each cell, which results from a duplication of genetic material on the long (q) arm of the X chromosome. It is inherited in an X-linked manner and is estimated to be responsible for 1 to 2 percent of all cases of intellectual disability caused by changes in the X chromosome [medquad_ans_0003860-1, medquad_ans_0000635-1, medquad_ans_0000635-3]. Approximately one third of individuals cannot walk without assistance

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0003860-1, medquad_ans_0000635-1, medquad_ans_0000635-3`

### Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Scores**: context_relevance=0.833, groundedness=0.137, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000594-1__dup2` (score=0.0648) - Melanoma is the most serious type of skin cancer. Often the first sign of melanoma is a change in the size, shape, color, or feel of a mole. Most melanomas have a black or black-blue area. Melanoma may also appear as ...
2. `medquad_ans_0000028_2-1` (score=0.0633) - Key Points - Melanoma is a disease in which malignant (cancer) cells form in melanocytes (cells that color the skin). - There are different types of cancer that start in the skin. - Melanoma can occur anywhere on the ...
3. `medquad_ans_0000063-15` (score=0.0628) - Melanoma is the fastest growing and most invasive type of skin cancer. This cancer arises from overgrowth of melanocytes. Melanocytes are specialized skin cells that produce a pigment called melanin.
4. `medquad_ans_0000063-6__dup1` (score=0.0624) - Many Treatment Options There are many treatment options for people with melanoma and non-melanoma skin cancer. The choice of treatment depends on your age and general health, the site of the cancer, the stage of the c...
5. `medquad_ans_0000063-30` (score=0.0604) - The National Cancer Institute has developed a comprehensive online cancer database called the Physician Data Query (PDQ) to present evidence from the most recent research on melanoma and other skin cancers. Click here...

**Answer**:

The provided context does not contain information about the outlook for melanoma, such as prognosis or chance of recovery. Therefore, it is not possible to answer the question based on the given text. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.719, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000132-2` (score=0.0328) - Boomerang dysplasia is a rare disorder; its exact prevalence is unknown. Approximately 10 affected individuals have been identified.
2. `medquad_ans_0000132-5` (score=0.0323) - These resources address the diagnosis or management of boomerang dysplasia: - Gene Review: Gene Review: FLNB-Related Disorders - Genetic Testing Registry: Boomerang dysplasia These resources from MedlinePlus offer inf...
3. `medquad_ans_0000132-1` (score=0.0317) - Boomerang dysplasia is a disorder that affects the development of bones throughout the body. Affected individuals are born with inward- and upward-turning feet (clubfeet) and dislocations of the hips, knees, and elbow...
4. `medquad_ans_0001959-1` (score=0.0312) - Dyggve-Melchior-Clausen (DMC) syndrome is a rare, progressive genetic condition characterized by abnormal skeletal development, microcephaly, and intellectual disability. Only about 100 cases have been reported to dat...
5. `medquad_ans_0004514-4` (score=0.0308) - How is oculopharyngeal muscular dystrophy inherited?

**Answer**:

The provided context does not specify whether boomerang dysplasia is inherited. While it mentions that boomerang dysplasia is a rare disorder affecting bone development and lists resources for diagnosis and management, there is no information regarding its mode of inheritance [medquad_ans_0000132-1][medquad_ans_0000132-2][medquad_ans_0000132-5]. Therefore, based on the given context, the inheritance pattern of boomerang dysplasia cannot be determined.

**Supported citations**: `medquad_ans_0000132-1`, `medquad_ans_0000132-2`, `medquad_ans_0000132-5`
**Unsupported citations**: _none_

### Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Scores**: context_relevance=0.800, groundedness=0.087, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000466-1` (score=0.0656) - Hereditary multiple osteochondromas is a condition in which people develop multiple benign (noncancerous) bone tumors called osteochondromas. The number of osteochondromas and the bones on which they are located vary ...
2. `medquad_ans_0000466-2` (score=0.0645) - The incidence of hereditary multiple osteochondromas is estimated to be 1 in 50,000 individuals. This condition occurs more frequently in some isolated populations: the incidence is approximately 1 in 1,000 in the Cha...
3. `medquad_ans_0002838-1` (score=0.0635) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called...
4. `medquad_ans_0002838-2` (score=0.0625) - What are the signs and symptoms of Hereditary multiple osteochondromas? The Human Phenotype Ontology provides the following list of signs and symptoms for Hereditary multiple osteochondromas. If the information is ava...
5. `medquad_ans_0000466-5` (score=0.0615) - These resources address the diagnosis or management of hereditary multiple osteochondromas: - Gene Review: Gene Review: Hereditary Multiple Osteochondromas - Genetic Testing Registry: Multiple congenital exostosis - G...

**Answer**:

The provided context does not mention specific genetic changes related to hereditary multiple osteochondromas. Therefore, based on the given information, the genetic changes are not specified [medquad_ans_0000466-1, medquad_ans_0002838-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000466-1, medquad_ans_0002838-1`

### Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.753, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0004510-1` (score=0.0489) - Oculofaciocardiodental syndrome is a genetic syndrome that affects the eyes, heart, face, and teeth. Common signs and symptoms include abnormally small deep-set eyes, cataracts, long narrow face, a broad nasal tip tha...
2. `medquad_ans_0000746-5` (score=0.0487) - These resources address the diagnosis or management of oculofaciocardiodental syndrome: - Genetic Testing Registry: Oculofaciocardiodental syndrome These resources from MedlinePlus offer information about the diagnosi...
3. `medquad_ans_0004510-2` (score=0.0476) - What are the signs and symptoms of Oculofaciocardiodental syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Oculofaciocardiodental syndrome. If the information is available, ...
4. `medquad_ans_0000746-1` (score=0.0466) - Oculofaciocardiodental (OFCD) syndrome is a condition that affects the development of the eyes (oculo-), facial features (facio-), heart (cardio-) and teeth (dental). This condition occurs only in females. The eye abn...
5. `medquad_ans_0000151-2` (score=0.0453) - Cardiofaciocutaneous syndrome is a very rare condition whose incidence is unknown. Researchers estimate that 200 to 300 people worldwide have this condition.

**Answer**:

The provided context does not specify the total number of people affected by oculofaciocardiodental syndrome. While it lists the frequency of various signs and symptoms in individuals with the condition, such as "90% with abnormality of the cardiac septa," it does not provide a population estimate or prevalence figure. Therefore, the number of people affected cannot be determined from the given information.

Answer: Insufficient context to determine the number of people affected by oculofaciocardiodental syndrome. [medquad_ans_0004510-2]

**Supported citations**: `medquad_ans_0004510-2`
**Unsupported citations**: _none_

### Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Scores**: context_relevance=1.000, groundedness=0.975, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000189-1` (score=0.0653) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused b...
2. `medquad_ans_0000189-5` (score=0.0638) - These resources address the diagnosis or management of citrullinemia: - Baby's First Test: Citrullinemia, Type I - Baby's First Test: Citrullinemia, Type II - Gene Review: Gene Review: Citrin Deficiency - Gene Review:...
3. `medquad_ans_0001350-1` (score=0.0638) - Citrullinemia type I is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. This condition, also known as classic citrullinemia, belongs to a class of genetic diseases call...
4. `medquad_ans_0001350-5` (score=0.0628) - What happens when citrullinemia type I is not treated? Untreated individuals with the severe form of citrullinemia type I have hyperammonemia (plasma ammonia concentration 1000-3000 mol/L). Without prompt intervention...
5. `medquad_ans_0004334-1` (score=0.0620) - Neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) is a liver condition is also known as neonatal-onset type II citrullinemia. NICCD blocks the flow of bile (a digestive fluid produced by the liver)...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused by mutations in different genes [medquad_ans_0000189-1]. Type I citrullinemia (also known as classic citrullinemia) usually becomes evident in the first few days of life, with symptoms including lethargy, poor feeding, vomiting, seizures, and loss of consciousness, which are life-threatening in many cases. A milder, later-onset form can occur in childhood or adulthood, associated with intense headaches, partial loss of vision, ataxia, and lethargy. Type II citrullinemia chiefly affects the nervous system, causing confusion, restlessness, memory loss, abnormal behaviors, seizures, and coma, and may appear during adulthood. In some cases

**Supported citations**: `medquad_ans_0000189-1`
**Unsupported citations**: _none_

### Example 99 (`0000718-1__dup1`)

**Question**: What is (are) Bilateral frontal polymicrogyria ?

**Scores**: context_relevance=0.833, groundedness=0.968, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000718-1__dup1` (score=0.0656) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
2. `medquad_ans_0000720-1__dup1` (score=0.0645) - Bilateral generalized polymicrogyria is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). This is the most widespread form of polymicrogyria and typically affects the enti...
3. `medquad_ans_0004956-1` (score=0.0632) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. Specifically, the surface of the brain develops too many folds which are unusually small. The signs and symptoms associate...
4. `medquad_ans_0000719-1__dup1` (score=0.0627) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides ...
5. `medquad_ans_0000807-1` (score=0.0613) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. The surface of the brain normally has many ridges or folds, called gyri. In people with polymicrogyria, the brain develops...

**Answer**:

Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form that affects only the frontal lobes of the brain, without involving the areas located behind the Sylvius fissure or the Rolando sulcus. Signs and symptoms include delayed motor and language milestones, spastic hemiparesis or quadriparesis, and mild to moderate intellectual disability. Seizures may also be present. The cause of this form is still not known, unlike the frontoparietal form, which is associated with mutations in the GPR56 gene. Treatment is based on the signs and symptoms present in each individual [medquad_ans_0000718-1__dup1].

**Supported citations**: `medquad_ans_0000718-1__dup1`
**Unsupported citations**: _none_

### Example 100 (`0000944-5`)

**Question**: What are the treatments for spondylocostal dysostosis ?

**Scores**: context_relevance=0.714, groundedness=0.971, answer_relevance=0.486

**Top docs**:

1. `medquad_ans_0000944-2` (score=0.0820) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.
2. `medquad_ans_0000949-5` (score=0.0792) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from Medli...
3. `medquad_ans_0005820-1` (score=0.0784) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
4. `medquad_ans_0005821-1` (score=0.0770) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
5. `medquad_ans_0005819-1` (score=0.0756) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...

**Answer**:

Treatment for spondylocostal dysostosis is symptomatic and supportive and may include respiratory support and surgery to correct inguinal hernia and scoliosis [medquad_ans_0005820-1].

**Supported citations**: `medquad_ans_0005820-1`
**Unsupported citations**: _none_

### Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.700, answer_relevance=0.543

**Top docs**:

1. `medquad_ans_0002652-2` (score=0.0762) - What are the signs and symptoms of Gordon syndrome? Gordon syndrome belongs to a group of conditions known as the distal arthrogryposes, which are characterized by stiffness and impaired mobility of certain joints of ...
2. `medquad_ans_0000786-1` (score=0.0650) - Perry syndrome is a progressive brain disease that is characterized by four major features: a pattern of movement abnormalities known as parkinsonism, psychiatric changes, weight loss, and abnormally slow breathing (h...
3. `medquad_ans_0000786-2` (score=0.0645) - Perry syndrome is very rare; about 50 affected individuals have been reported worldwide.
4. `medquad_ans_0000786-5` (score=0.0616) - These resources address the diagnosis or management of Perry syndrome: - Gene Review: Gene Review: Perry Syndrome - Genetic Testing Registry: Perry syndrome - MedlinePlus Encyclopedia: Major Depression - MedlinePlus E...
5. `medquad_ans_0000245-1` (score=0.0596) - Costello syndrome is a disorder that affects many parts of the body. This condition is characterized by delayed development and intellectual disability, loose folds of skin (which are especially noticeable on the hand...

**Answer**:

The provided context does not contain any information about Perry syndrome. Therefore, it is not possible to answer the question based on the given context. [medquad_ans_0002652-2]

**Supported citations**: `medquad_ans_0002652-2`
**Unsupported citations**: _none_

### Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Scores**: context_relevance=1.000, groundedness=0.907, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0001028-2` (score=0.0801) - The prevalence of warfarin sensitivity is unknown. However, it appears to be more common in people who are older, those with lower body weights, and individuals of Asian ancestry. Of the approximately 2 million people...
2. `medquad_ans_0001027-2` (score=0.0797) - Warfarin resistance is thought to be a rare condition, although its prevalence is unknown.
3. `medquad_ans_0006398-2` (score=0.0794) - What are the signs and symptoms of Warfarin sensitivity? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin sensitivity. If the information is available, the table below includ...
4. `medquad_ans_0006398-1` (score=0.0782) - Warfarin sensitivity is a condition that is characterized by a reduced tolerance for a "blood-thinning" medication called warfarin. Warfarin is an anticoagulant that is often prescribed to people who are at an increas...
5. `medquad_ans_0006399-1` (score=0.0780) - What are the signs and symptoms of Warfarin syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin syndrome. If the information is available, the table below includes how...

**Answer**:

The symptoms of Warfarin sensitivity include excessive internal bleeding, which is often seen when individuals with warfarin sensitivity are given too much of the medication. This can manifest as abnormal bleeding in the brain, gastrointestinal tract, or other tissues. The most common sign is excessive internal bleeding [medquad_ans_0001028-2]. Additionally, the Human Phenotype Ontology lists signs and symptoms related to warfarin sensitivity, such as abnormality of blood and blood-forming tissues and abnormality of metabolism/homeostasis, though specific frequencies are not always available [medquad_ans_0006398-2].

**Supported citations**: `medquad_ans_0001028-2`, `medquad_ans_0006398-2`
**Unsupported citations**: _none_

### Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.533, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000701-2` (score=0.0656) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.
2. `medquad_ans_0004262-1` (score=0.0645) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, a...
3. `medquad_ans_0000701-1` (score=0.0635) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most...
4. `medquad_ans_0004262-2` (score=0.0623) - What are the signs and symptoms of Myhre syndrome? Myhre syndrome is a condition with features affecting many systems and functions of the body. Associated findings might include: Delayed development of language and m...
5. `medquad_ans_0000701-5` (score=0.0611) - These resources address the diagnosis or management of Myhre syndrome: - Centers for Disease Control and Prevention: Types of Hearing Loss - Genetic Testing Registry: Myhre syndrome - National Institute on Deafness an...

**Answer**:

Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement of the muscles (muscle hypertrophy), and/or joint stiffness. The condition is caused by mutations in the SMAD4 gene and is inherited in an autosomal dominant pattern, with most cases due to a new mutation. Affected individuals often have delayed development of language and motor skills, such as crawling and walking, and may have behavioral issues like features of autism or related developmental disorders. Common physical features include hearing loss (sensorineural, conductive, or mixed), skin stiffness, thickened skull bones, flattened spine bones (platyspondyly), broad ribs, underdeveloped pelvic wings (hypoplastic iliac wings), short fingers and toes (brachydactyly), joint stiffness, and limited mobility. Typical facial features include narrow

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Scores**: context_relevance=0.857, groundedness=0.950, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0006057-1` (score=0.0487) - Thanatophoric dysplasia is a severe skeletal disorder characterized by extremely short limbs and folds of extra skin on the arms and legs. Other features of this condition include a narrow chest, short ribs, underdeve...
2. `medquad_ans_0000976-5` (score=0.0477) - These resources address the diagnosis or management of thanatophoric dysplasia: - Gene Review: Gene Review: Thanatophoric Dysplasia - Genetic Testing Registry: Thanatophoric dysplasia type 1 - Genetic Testing Registry...
3. `medquad_ans_0000976-1` (score=0.0476) - Thanatophoric dysplasia is a severe skeletal disorder characterized by extremely short limbs and folds of extra (redundant) skin on the arms and legs. Other features of this condition include a narrow chest, short rib...
4. `medquad_ans_0006057-3` (score=0.0455) - What causes thanatophoric dysplasia? Thanatophoric dysplasia is caused by mutations in the FGFR3 gene. This gene provides instructions for making a protein that is involved in the development and maintenance of bone a...
5. `medquad_ans_0000976-2` (score=0.0427) - This condition occurs in 1 in 20,000 to 50,000 newborns. Type I thanatophoric dysplasia is more common than type II.

**Answer**:

The symptoms of Thanatophoric dysplasia include extremely short limbs, folds of extra (redundant) skin on the arms and legs, a narrow chest, short ribs, underdeveloped lungs, an enlarged head with a large forehead, and prominent, wide-spaced eyes. [medquad_ans_0006057-1] [medquad_ans_0000976-1]

**Supported citations**: `medquad_ans_0006057-1`, `medquad_ans_0000976-1`
**Unsupported citations**: _none_

### Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000381-1__dup1` (score=0.0806) - Fragile X syndrome is the most common form of inherited developmental disability. A problem with a specific gene causes the disease. Normally, the gene makes a protein you need for brain development. But the problem c...
2. `medquad_ans_0002417-2` (score=0.0790) - What are the signs and symptoms of Fragile X syndrome? Fragile X syndrome is characterized by developmental problems including intellectual disability and delayed speech and language development. Males are usually mor...
3. `medquad_ans_0000381-1` (score=0.0789) - Fragile X syndrome is a genetic condition that causes a range of developmental problems including learning disabilities and cognitive impairment. Usually, males are more severely affected by this disorder than females...
4. `medquad_ans_0000381-2` (score=0.0773) - Fragile X syndrome occurs in approximately 1 in 4,000 males and 1 in 8,000 females.
5. `medquad_ans_0000381-5` (score=0.0767) - These resources address the diagnosis or management of fragile X syndrome: - Gene Review: Gene Review: FMR1-Related Disorders - GeneFacts: Fragile X Syndrome: Diagnosis - GeneFacts: Fragile X Syndrome: Management - Ge...

**Answer**:

Symptoms of Fragile X syndrome include intellectual disability and delayed speech and language development. Males are usually more severely affected than females. Additional features may include anxiety, attention deficit disorder (ADD), features of autism spectrum disorders affecting communication and social interaction, and seizures. Physical features that may become more apparent with age include a long and narrow face, large ears, a prominent jaw and forehead, unusually flexible fingers, flat feet, and in males, enlarged testicles (macroorchidism) after puberty. Cognitive impairment, joint hypermobility, macroorchidism, neurological speech impairment, otitis media, pes planus, abnormality of the pinna, attention deficit hyperactivity disorder, frontal bossing, intellectual disability, moderate, long face, macrocephaly, mandibular prognathia, muscular hypotonia, narrow face, sinusitis, abnormality of the mitral valve, autism, cerebral cortical atrophy, dilatation of

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Scores**: context_relevance=1.000, groundedness=0.495, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000166-2` (score=0.0653) - CADASIL is likely a rare condition; however, its prevalence is unknown.
2. `medquad_ans_0000166-5` (score=0.0645) - These resources address the diagnosis or management of CADASIL: - Butler Hospital: Treatment and Management of CADASIL - Gene Review: Gene Review: CADASIL - Genetic Testing Registry: Cerebral autosomal dominant arteri...
3. `medquad_ans_0000056-3__dup1` (score=0.0637) - Symptoms usually progress slowly. By age 65, the majority of persons with CADASIL have cognitive problems and dementia. Some will become dependent due to multiple strokes.
4. `medquad_ans_0000889-2__dup1` (score=0.0625) - What are the signs and symptoms of CADASIL? Strokes are the main feature of CADASIL and often occur repeatedly. Strokes may lead to severe disability such as an inability to walk and urinary incontinence. The average ...
5. `medquad_ans_0000166-1` (score=0.0578) - Cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy, usually called CADASIL, is an inherited condition that causes stroke and other impairments. This condition affects blood flow...

**Answer**:

CADASIL is a rare condition characterized by recurrent strokes, which are the main feature, often occurring repeatedly and leading to severe disability such as inability to walk and urinary incontinence. It typically presents with a slow decline in cognitive function, with over half of affected individuals experiencing a decline in thinking ability, beginning as early as 35 years of age. Approximately 75% of people with CADASIL eventually develop dementia, including significant difficulties with reasoning and memory. Migraines with aura occur in about 35% of individuals, with the first attack typically occurring at an average age of 26 years. Psychiatric issues, such as personality changes or severe depression, affect about 30% of people. Other common signs and symptoms include abnormality of temperature regulation (90%), abnormality of the retinal vasculature (90%), and memory impairment (50%). The condition is associated with cerebral autosomal dominant arter

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Scores**: context_relevance=0.833, groundedness=0.960, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000010-8` (score=0.0602) - - Multiple endocrine neoplasia type 1 (MEN1) is an inherited disorder that causes hormone-secreting tumors in the duodenum and the endocrine glands-most often the parathyroid, pancreas, and pituitary. - Overactive par...
2. `medquad_ans_0000010-1__dup6` (score=0.0576) - MEN1 is an inherited disorder that causes tumors in the endocrine glands and the duodenum, the first part of the small intestine. MEN1 is sometimes called multiple endocrine adenomatosis or Wermer's syndrome, after on...
3. `medquad_ans_0000014-5__dup1` (score=0.0569) - In about 80 percent of people with primary hyperparathyroidism, a benign, or noncancerous, tumor called an adenoma has formed in one of the parathyroid glands.2 The tumor causes the gland to become overactive. In most...
4. `medquad_ans_0000710-1__dup1` (score=0.0537) - Pheochromocytoma is a rare tumor that usually starts in the cells of one of your adrenal glands. Although they are usually benign, pheochromocytomas often cause the adrenal gland to make too many hormones. This can le...
5. `medquad_ans_0000688-1` (score=0.0492) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate...

**Answer**:

Multiple endocrine neoplasia (MEN1) is an inherited disorder that causes hormone-secreting tumors in the duodenum and the endocrine glands, most often the parathyroid, pancreas, and pituitary [medquad_ans_0000010-8]. It is also known as multiple endocrine adenomatosis or Wermer's syndrome [medquad_ans_0000010-1__dup6]. MEN1 is rare, occurring in about one in 30,000 people, and affects both sexes equally without geographical, racial, or ethnic preferences [medquad_ans_0000010-1__dup6]. In people with MEN1, multiple endocrine glands form tumors and become hormonally overactive, often simultaneously [medquad_ans_0000010-1__dup6]. Although many tumors associated with

**Supported citations**: `medquad_ans_0000010-8`, `medquad_ans_0000010-1__dup6`
**Unsupported citations**: _none_

### Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Scores**: context_relevance=0.750, groundedness=0.751, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000200-1__dup2` (score=0.0810) - The mucopolysaccharidoses are a group of inherited metabolic diseases in which a defective or missing enzyme causes large amounts of complex sugar molecules to accumulate in harmful amounts in the body's cells and tis...
2. `medquad_ans_0004160-1` (score=0.0794) - Mucopolysaccharidosis type IIIC (MPS IIIC) is an genetic disorder that makes the body unable to break down large sugar molecules called glycosaminoglycans (GAGs, formerly called mucopolysaccharides). Specifically, peo...
3. `medquad_ans_0000682-5` (score=0.0787) - These resources address the diagnosis or management of mucopolysaccharidosis type VI: - Emory University Lysosomal Storage Disease Center - Genetic Testing Registry: Mucopolysaccharidosis type VI - MedlinePlus Encyclo...
4. `medquad_ans_0000680-2` (score=0.0784) - MPS III is the most common type of mucopolysaccharidosis; the estimated incidence of all four types combined is 1 in 70,000 newborns. MPS IIIA and MPS IIIB are much more common than MPS IIIC and MPS IIID.
5. `medquad_ans_0004155-1` (score=0.0768) - Mucopolysaccharidosis I (MPS I) is a condition that affects many parts of the body. It is a progressively debilitating disorder; however, the rate of progression varies among affected individuals. MPS I is caused by m...

**Answer**:

The provided context does not specify the cause of Mucopolysaccharidosis type I. While it describes the general mechanism of mucopolysaccharidoses—defective or missing enzymes leading to accumulation of complex sugar molecules—it does not detail the specific enzyme deficiency or genetic cause for MPS I. Therefore, based on the given information, the cause of MPS I cannot be determined.

[medquad_ans_0004155-1] mentions that MPS I is a progressively debilitating condition affecting many parts of the body, but does not state the underlying cause. Other documents discuss MPS IIIC and MPS III, not MPS I. Thus, the cause of MPS I is not explicitly provided in the context.

**Supported citations**: `medquad_ans_0004155-1`
**Unsupported citations**: _none_

### Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Scores**: context_relevance=0.700, groundedness=0.016, answer_relevance=0.780

**Top docs**:

1. `medquad_ans_0000104-4__dup2` (score=0.0608) - Pleurisy The main symptom of pleurisy is a sharp or stabbing pain in your chest that gets worse when you breathe in deeply or cough or sneeze. The pain may stay in one place or it may spread to your shoulders or back....
2. `medquad_ans_0000107-3__dup2` (score=0.0533) - Primary ciliary dyskinesia (PCD) is a rare, inherited disease. "Inherited" means the disease is passed from parents to children through genes. With PCD, this process is very complex. Researchers are still learning how...
3. `medquad_ans_0000720-1__dup2` (score=0.0453) - Your pleura is a large, thin sheet of tissue that wraps around the outside of your lungs and lines the inside of your chest cavity. Between the layers of the pleura is a very thin space. Normally it's filled with a sm...
4. `medquad_ans_0000104-1__dup3` (score=0.0446) - Pleurisy (PLUR-ih-se) is a condition in which the pleura is inflamed. The pleura is a membrane that consists of two large, thin layers of tissue. One layer wraps around the outside of your lungs. The other layer lines...
5. `medquad_ans_0000109-4__dup2` (score=0.0422) - Pulmonary hypertension (PH) begins with inflammation and changes in the cells that line your pulmonary arteries. Other factors also can affect the pulmonary arteries and cause PH. For example, the condition may develo...

**Answer**:

The provided context does not contain information about genetic changes related to primary spontaneous pneumothorax. Therefore, it is not possible to answer the question based on the given text. [medquad_ans_0000104-4__dup2, medquad_ans_0000107-3__dup2, medquad_ans_0000720-1__dup2]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000104-4__dup2, medquad_ans_0000107-3__dup2, medquad_ans_0000720-1__dup2`

### Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Scores**: context_relevance=0.750, groundedness=0.497, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000066-13` (score=0.0630) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blo...
2. `medquad_ans_0000124-1__dup2` (score=0.0625) - A stroke occurs if the flow of oxygen-rich blood to a portion of the brain is blocked. Without oxygen, brain cells start to die after a few minutes. Sudden bleeding in the brain also can cause a stroke if it damages b...
3. `medquad_ans_0000527-1__dup2` (score=0.0620) - A stroke is a medical emergency. There are two types - ischemic and hemorrhagic. Ischemic stroke is the most common type. It is usually caused by a blood clot that blocks or plugs a blood vessel in the brain. This kee...
4. `medquad_ans_0000857-1__dup2` (score=0.0613) - A stroke is a medical emergency. Strokes happen when blood flow to your brain stops. Within minutes, brain cells begin to die. There are two kinds of stroke. The more common kind, called ischemic stroke, is caused by ...
5. `medquad_ans_0000261-1__dup2` (score=0.0612) - A stroke occurs when the blood supply to part of the brain is suddenly interrupted or when a blood vessel in the brain bursts, spilling blood into the spaces surrounding brain cells. Brain cells die when they no longe...

**Answer**:

A stroke is a serious medical condition that occurs when the flow of oxygen-rich blood to a portion of the brain is blocked, leading to brain cell death within minutes due to lack of oxygen. It can also result from sudden bleeding in the brain that damages brain cells. There are two main types: ischemic stroke, which is the most common and caused by a blood clot blocking a blood vessel in the brain, and hemorrhagic stroke, which occurs when a blood vessel in the brain breaks and bleeds, damaging brain cells. Stroke symptoms include sudden weakness, paralysis or numbness of the face, arms, or legs, trouble speaking or understanding speech, and trouble seeing. A stroke can cause lasting brain damage, long-term disability, or even death, and requires immediate emergency care. Every minute during a stroke counts, and if someone is suspected of having a stroke, 911 should be called immediately. A transient ischemic attack (TIA), or

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.780, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000417-1__dup1` (score=0.0529) - You lose up to 100 hairs from your scalp every day. That's normal, and in most people, those hairs grow back. But many men -- and some women -- lose hair as they grow older. You can also lose your hair if you have cer...
2. `medquad_ans_0006276-5` (score=0.0487) - How is uncombable hair syndrome diagnosed? A diagnosis of uncombable hair syndrome (UHS) is made by observing the characteristic symptoms of the condition, as well observing the hair shaft under a special microscope. ...
3. `medquad_ans_0006276-2` (score=0.0484) - What are the signs and symptoms of Uncombable hair syndrome? Uncombable hair syndrome (UHS) may first become apparent any time between the ages of 3 months and 12 years. It only affects the scalp hair. The quantity of...
4. `medquad_ans_0006276-1` (score=0.0481) - Uncombable hair syndrome (UHS) is a rare disorder of the hair shaft of the scalp. It usually is characterized by silvery-blond or straw-colored hair that is disorderly; stands out from the scalp; and cannot be combed ...
5. `medquad_ans_0006276-6` (score=0.0466) - How might uncombable hair syndrome be treated? There is no definitive treatment for uncombable hair syndrome, but the condition usually improves or resolves on its own with the onset of puberty. Gentle hair care is ge...

**Answer**:

The provided context does not specify the causes of Uncombable hair syndrome. It describes the signs and symptoms, diagnostic methods, and inheritance pattern (autosomal dominant), but does not identify the underlying causes [medquad_ans_0006276-2][medquad_ans_0006276-5]. Therefore, based on the given information, the causes cannot be determined.

**Supported citations**: `medquad_ans_0006276-2`, `medquad_ans_0006276-5`
**Unsupported citations**: _none_

### Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000139-2__dup2` (score=0.0287) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...
2. `medquad_ans_0000770-4` (score=0.0265) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000858-4` (score=0.0255) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000801-4` (score=0.0252) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000898-4` (score=0.0249) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain information about Russell-Silver syndrome or its inheritance pattern. Therefore, it is not possible to determine whether Russell-Silver syndrome is inherited based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Scores**: context_relevance=1.000, groundedness=0.959, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.0328) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0000718-1__dup2` (score=0.0320) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts...
3. `medquad_ans_0000094-5__dup1` (score=0.0320) - Obesity hypoventilation syndrome (OHS) is diagnosed based on your medical history, signs and symptoms, and test results. Specialists Involved A critical care specialist, pulmonologist (lung specialist), and/or sleep s...
4. `medquad_ans_0000018-1__dup2` (score=0.0312) - After any operation, you'll have some side effects. There is usually some pain with surgery. There may also be swelling and soreness around the area that the surgeon cut. Your surgeon can tell you which side effects t...
5. `medquad_ans_0000022-5` (score=0.0308) - These resources address the diagnosis or management of adiposis dolorosa: - Genetic Testing Registry: Lipomatosis dolorosa - Merck Manual Consumer Version: Lipomas These resources from MedlinePlus offer information ab...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. It often limits the amount of food you can take in, and some types also affect how you digest and absorb nutrients. All types of weight loss surgery have risks and complications, such as infections, hernias, and blood clots. Many people lose weight quickly, but some regain weight later; however, following diet and exercise recommendations can help keep most of the weight off. Ongoing medical follow-up is required for the rest of your life [medquad_ans_0000969-1__dup2].

**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

### Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.824, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000191-8` (score=0.0703) - - Primary sclerosing cholangitis (PSC) inflames, scars, and blocks bile ducts inside and outside the liver. - When bile ducts become blocked, bile builds up in the liver and damages liver cells. - PSC can lead to vita...
2. `medquad_ans_0000889-2` (score=0.0492) - Saethre-Chotzen syndrome has an estimated prevalence of 1 in 25,000 to 50,000 people.
3. `medquad_ans_0000889-5` (score=0.0484) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Sae...
4. `medquad_ans_0000889-4` (score=0.0476) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
5. `medquad_ans_0000889-1` (score=0.0469) - Saethre-Chotzen syndrome is a genetic condition characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the ...

**Answer**:

The provided context does not specify treatments for Saethre-Chotzen syndrome. It mentions resources that address the diagnosis or management of the condition, such as Genetic Testing Registry, MedlinePlus Encyclopedia, and information on diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care [medquad_ans_0000889-5]. However, no explicit details on specific treatments are given. Therefore, based on the available information, the treatments for Saethre-Chotzen syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000889-5`
**Unsupported citations**: _none_

### Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=1.000, groundedness=0.489, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000512-1` (score=0.0648) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effect...
2. `medquad_ans_0000512-2` (score=0.0645) - Idiopathic pulmonary fibrosis has an estimated prevalence of 13 to 20 per 100,000 people worldwide. About 100,000 people are affected in the United States, and 30,000 to 40,000 new cases are diagnosed each year. Famil...
3. `medquad_ans_0000512-5` (score=0.0643) - These resources address the diagnosis or management of idiopathic pulmonary fibrosis: - Gene Review: Gene Review: Pulmonary Fibrosis, Familial - Genetic Testing Registry: Idiopathic fibrosing alveolitis, chronic form ...
4. `medquad_ans_0000078-1__dup2` (score=0.0620) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...
5. `medquad_ans_0000758-1__dup2` (score=0.0620) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...

**Answer**:

Idiopathic pulmonary fibrosis is a chronic, progressive lung disease that causes scar tissue (fibrosis) to build up in the lungs, making them unable to transport oxygen into the bloodstream effectively. The disease typically affects people between the ages of 50 and 70. Common signs and symptoms include shortness of breath, a persistent dry, hacking cough, loss of appetite, and gradual weight loss. Some individuals may also develop widened and rounded tips of the fingers and toes (clubbing) due to oxygen shortage. Over time, lung scarring worsens until the lungs can no longer provide sufficient oxygen to the body's organs and tissues. The disease can lead to serious complications such as lung cancer, pulmonary emboli, pneumonia, or pulmonary hypertension. Most affected individuals survive 3 to 5 years after diagnosis, although the disease course is highly variable, with some living for a decade or longer. Idiopathic pulmonary fibrosis is usually sporadic

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.862, answer_relevance=0.617

**Top docs**:

1. `medquad_ans_0000213-3__dup1` (score=0.0492) - Early identification of and treatment for individuals with neuroleptic malignant syndrome improves outcome. If clinically indicated, a low potency neuroleptic can be reintroduced very slowly when the individual recove...
2. `medquad_ans_0004379-1` (score=0.0481) - Neuroleptic malignant syndrome is a rare neurological condition that is caused by an adverse reaction to neuroleptic (tranquilizer) or antipsychotic drugs. These drugs are commonly prescribed for the treatment of schi...
3. `medquad_ans_0000213-4__dup1` (score=0.0479) - The NINDS supports research on neurological disorders such as neuroleptic malignant syndrome. Much of this research focuses on finding ways to prevent and treat the disorder.
4. `medquad_ans_0000273-4__dup1` (score=0.0469) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts TSC research in its laboratories at the National Institutes of Health (NIH) and also supports TSC research through grants to major medical i...
5. `medquad_ans_0000022-4__dup1` (score=0.0462) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system, and to use that knowledge to reduce the burden of neurological di...

**Answer**:

The National Institute of Neurological Disorders and Stroke (NINDS) supports research on neurological disorders such as neuroleptic malignant syndrome, focusing on finding ways to prevent and treat the disorder [medquad_ans_0000213-4__dup1]. However, specific details about ongoing clinical trials or particular research studies on neuroleptic malignant syndrome are not provided in the context.

**Supported citations**: `medquad_ans_0000213-4__dup1`
**Unsupported citations**: _none_

### Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Scores**: context_relevance=0.714, groundedness=0.846, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000203-11` (score=0.0492) - - A kidney stone is a solid piece of material that forms in a kidney when substances that are normally found in the urine become highly concentrated. - Kidney stones are one of the most common disorders of the urinary...
2. `medquad_ans_0000540-1__dup2` (score=0.0481) - A kidney stone is a solid piece of material that forms in the kidney from substances in the urine. It may be as small as a grain of sand or as large as a pearl. Most kidney stones pass out of the body without help fro...
3. `medquad_ans_0000221-8` (score=0.0476) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other he...
4. `medquad_ans_0000221-12` (score=0.0471) - - A kidney stone is a solid piece of material that forms in a kidney when there are high levels of certain substances in the urine. These substances are normally found in the urine and do not cause problems at lower l...
5. `medquad_ans_0000203-7` (score=0.0448) - People with kidney stones may have pain while urinating, see blood in the urine, or feel a sharp pain in the back or lower abdomen. The pain may last for a short or long time. People may experience nausea and vomiting...

**Answer**:

To diagnose kidney stones in adults, a healthcare provider will perform a physical exam and take a medical history, including questions about family history, diet, digestive problems, and other health issues. The provider may also perform urine, blood, and imaging tests to complete the diagnosis. Urine tests can identify infections or substances in the urine that contribute to stone formation, blood tests can detect underlying conditions that may lead to stones, and imaging tests are used to visualize the kidneys and urinary tract [medquad_ans_0000203-11][medquad_ans_0000221-8][medquad_ans_0000540-1__dup2].

**Supported citations**: `medquad_ans_0000203-11`, `medquad_ans_0000221-8`, `medquad_ans_0000540-1__dup2`
**Unsupported citations**: _none_

### Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000041_1-1` (score=0.0656) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of ...
2. `medquad_ans_0006289-1` (score=0.0640) - Urachal cancer is a rare type of bladder cancer, making up less than 1% of all bladder cancers. Only about 350 cases have been described in the medical literature to date. The urachus is a primitive structure which be...
3. `medquad_ans_0000099-1__dup2` (score=0.0623) - The bladder is a hollow organ in your lower abdomen that stores urine. Bladder cancer occurs in the lining of the bladder. It is the sixth most common type of cancer in the United States. Symptoms include - Blood in y...
4. `medquad_ans_0000041_1-2` (score=0.0623) - Signs of urethral cancer include bleeding or trouble with urination. These and other signs and symptoms may be caused by urethral cancer or by other conditions. There may be no signs or symptoms in the early stages. C...
5. `medquad_ans_0000041_1-5` (score=0.0606) - Key Points - After urethral cancer has been diagnosed, tests are done to find out if cancer cells have spread within the urethra or to other parts of the body. - There are three ways that cancer spreads in the body. -...

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. It is a rare cancer that occurs more often in men than in women. There are different types of urethral cancer that begin in cells that line the urethra, named for the types of cells that become malignant: squamous cell carcinoma (the most common type), transitional cell carcinoma, and adenocarcinoma. Risk factors include a history of bladder cancer and conditions that cause chronic inflammation in the urethra, such as sexually transmitted diseases (including human papillomavirus type 16) and frequent urinary tract infections (UTIs). Signs of urethral cancer include bleeding or trouble with urination. Tests that examine the urethra and bladder are used to detect and diagnose the disease. Urethral cancer can metastasize quickly to tissues around the urethra and is often found in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Scores**: context_relevance=0.750, groundedness=0.033, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0004671-4` (score=0.0328) - How is pachyonychia congenita inherited? Pachyonychia congenita (PC) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of the responsible...
2. `medquad_ans_0006446-4` (score=0.0320) - Is Williams syndrome inherited?
3. `medquad_ans_0004750-4` (score=0.0318) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is pass...
4. `medquad_ans_0004784-4` (score=0.0306) - How is Pelizaeus-Merzbacher disease inherited?
5. `medquad_ans_0002316-4` (score=0.0305) - How is fatal familial insomnia inherited? Fatal familial insomnia (FFI) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of the responsi...

**Answer**:

The provided context does not contain information about the inheritance of paramyotonia congenita. Therefore, based on the given sources, it is not possible to determine whether paramyotonia congenita is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000735-1__dup1` (score=0.0806) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. The disorder may become apparent in the first few months of life, or later in childhood. The more severe form ...
2. `medquad_ans_0000735-2__dup1` (score=0.0804) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound bio...
3. `medquad_ans_0000125-2` (score=0.0797) - Profound or partial biotinidase deficiency occurs in approximately 1 in 60,000 newborns
4. `medquad_ans_0000125-1` (score=0.0794) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. If this condition is not recognized and treated, its signs and symptoms typically appear within the first few ...
5. `medquad_ans_0000484-1` (score=0.0755) - Holocarboxylase synthetase deficiency is an inherited disorder in which the body is unable to use the vitamin biotin effectively. This disorder is classified as a multiple carboxylase deficiency, a group of disorders ...

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), and reduced consciousness/confusion (7.5%) [medquad_ans_0000735-2__dup1]. In the more severe form, profound biotinidase deficiency, additional symptoms may include delayed development, breathing problems, hearing and vision loss, problems with movement and balance (ataxia), skin rashes, hair

**Supported citations**: `medquad_ans_0000735-2__dup1`
**Unsupported citations**: _none_

### Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Scores**: context_relevance=0.571, groundedness=0.880, answer_relevance=0.571

**Top docs**:

1. `medquad_ans_0000966-1__dup2` (score=0.0328) - Summary : We all need clean water. People need it to grow crops and to operate factories, and for drinking and recreation. Fish and wildlife depend on it to survive. Many different pollutants can harm our rivers, stre...
2. `medquad_ans_0000298-1__dup1` (score=0.0323) - Summary : We all need to drink water. How much you need depends on your size, activity level, and the weather where you live. The water you drink is a combination of surface water and groundwater. Surface water includ...
3. `medquad_ans_0000331-1__dup1` (score=0.0317) - Summary : Our environment affects our health. If parts of the environment, like the air, water, or soil become polluted, it can lead to health problems. For example, asthma pollutants and chemicals in the air or in th...
4. `medquad_ans_0000019-1__dup3` (score=0.0306) - Summary : Air pollution is a mixture of solid particles and gases in the air. Car emissions, chemicals from factories, dust, pollen and mold spores may be suspended as particles. Ozone, a gas, is a major part of air p...
5. `medquad_ans_0000505-1__dup1` (score=0.0303) - Summary : We usually think of air pollution as being outdoors, but the air in your house or office could also be polluted. Sources of indoor pollution include - Mold and pollen - Tobacco smoke - Household products and...

**Answer**:

Yes, there is information about water pollution. The three most common pollutants affecting rivers, streams, lakes, and oceans are soil, nutrients, and bacteria. Rain washes soil into water bodies, which can kill fish eggs, clog fish gills, and block light, causing plant death. Nutrients from fertilizers cause algae growth, turning water green, especially in lakes, ponds, and reservoirs. Bacteria from sewage spills can pollute both fresh and salt water. To help protect water supplies, one should not pour household products like cleansers, medicines, or grease down the drain, and instead dispose of them at hazardous waste sites. Excess grease, diapers, condoms, and pet waste should be thrown in the garbage. Pet waste contains nutrients and germs that can contribute to water pollution [medquad_ans_0000966-1__dup2].

**Supported citations**: `medquad_ans_0000966-1__dup2`
**Unsupported citations**: _none_

### Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Scores**: context_relevance=0.857, groundedness=0.973, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000132-3__dup1` (score=0.0650) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is...
2. `medquad_ans_0000132-2__dup1` (score=0.0636) - Treatment depends on the specific cause of foot drop. The most common treatment is to support the foot with light-weight leg braces and shoe inserts, called ankle-foot orthotics. Exercise therapy to strengthen the mus...
3. `medquad_ans_0000132-1__dup1` (score=0.0630) - Foot drop describes the inability to raise the front part of the foot due to weakness or paralysis of the muscles that lift the foot. As a result, individuals with foot drop scuff their toes along the ground or bend t...
4. `medquad_ans_0005814-2` (score=0.0607) - What are the signs and symptoms of Split hand/foot malformation X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Split hand/foot malformation X-linked. If the information is...
5. `medquad_ans_0000290-1` (score=0.0596) - Distal arthrogryposis type 1 is a disorder characterized by joint deformities (contractures) that restrict movement in the hands and feet. The term "arthrogryposis" comes from the Greek words for joint (arthro-) and c...

**Answer**:

The outlook for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1].

**Supported citations**: `medquad_ans_0000132-3__dup1`
**Unsupported citations**: _none_

### Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Scores**: context_relevance=1.000, groundedness=0.511, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000145-2` (score=0.0656) - The prevalence of Camurati-Engelmann disease is unknown. Approximately 200 cases have been reported worldwide.
2. `medquad_ans_0000916-5__dup1` (score=0.0643) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene...
3. `medquad_ans_0000145-5` (score=0.0637) - These resources address the diagnosis or management of Camurati-Engelmann disease: - Gene Review: Gene Review: Camurati-Engelmann Disease - Genetic Testing Registry: Diaphyseal dysplasia These resources from MedlinePl...
4. `medquad_ans_0000145-1` (score=0.0623) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and h...
5. `medquad_ans_0000916-2__dup1` (score=0.0618) - What are the signs and symptoms of Camurati-Engelmann disease? People with Camurati-Engelmann disease have increased bone density, particularly affecting the long bones of the arms and legs (tibia, femur, humerus, uln...

**Answer**:

Camurati-Engelmann disease is a condition that mainly affects the bones, characterized by increased bone density, particularly in the long bones of the arms and legs. It can also affect the skull and hip bones. The thickened bones may lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. Increased skull density can cause increased pressure on the brain, resulting in neurological problems such as headaches, hearing loss, vision problems, dizziness (vertigo), ringing in the ears (tinnitus), and facial paralysis. Additional features include abnormal curvature of the spine (scoliosis), joint deformities (contractures), knock knees, flat feet (pes planus), abnormally long limbs in proportion to height, decreased muscle mass and body fat, and delayed puberty. The age of symptom onset varies, with most individuals developing pain or weakness by adolescence. In some cases, individuals may have the gene mutation

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000316-2` (score=0.0820) - X-linked Emery-Dreifuss muscular dystrophy is the most common form of this condition, affecting an estimated 1 in 100,000 people. The autosomal recessive type of this disorder appears to be very rare; only a few cases...
2. `medquad_ans_0000316-1` (score=0.0804) - Emery-Dreifuss muscular dystrophy is a condition that chiefly affects muscles used for movement (skeletal muscles) and heart (cardiac) muscle. Among the earliest features of this disorder are joint deformities called ...
3. `medquad_ans_0000316-5` (score=0.0794) - These resources address the diagnosis or management of Emery-Dreifuss muscular dystrophy: - Gene Review: Gene Review: Emery-Dreifuss Muscular Dystrophy - Genetic Testing Registry: Emery-Dreifuss muscular dystrophy - G...
4. `medquad_ans_0001935-1` (score=0.0772) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystroph...
5. `medquad_ans_0000631-1__dup1` (score=0.0753) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked, include joint deformities called contractures, which restrict movement and become noticeable in early childhood, most often involving the elbows, ankles, and neck. Affected individuals also experience slowly progressive muscle weakness and wasting, beginning in the muscles of the upper arms and lower legs and progressing to the shoulders and hips. Almost all people with this condition have heart problems by adulthood, which often stem from cardiac conduction defects and arrhythmias. If untreated, these heart abnormalities can lead to bradycardia, syncope, increased risk of stroke, and sudden death [medquad_ans_0000316-1].

**Supported citations**: `medquad_ans_0000316-1`
**Unsupported citations**: _none_

### Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.833, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000072-2` (score=0.0820) - Arterial tortuosity syndrome is a rare disorder; its prevalence is unknown. About 100 cases have been reported in the medical literature.
2. `medquad_ans_0000072-5` (score=0.0801) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins Mc...
3. `medquad_ans_0000072-1` (score=0.0799) - Arterial tortuosity syndrome is a disorder that affects connective tissue. Connective tissue provides strength and flexibility to structures throughout the body, including blood vessels, skin, joints, and the gastroin...
4. `medquad_ans_0000072-3` (score=0.0781) - Arterial tortuosity syndrome is caused by mutations in the SLC2A10 gene. This gene provides instructions for making a protein called GLUT10. The level of GLUT10 appears to be involved in the regulation of a process ca...
5. `medquad_ans_0000126-6` (score=0.0593) - Tetralogy of Fallot is repaired with open-heart surgery, either soon after birth or later in infancy. The goal of surgery is to repair the four defects of tetralogy of Fallot so the heart can work as normally as possi...

**Answer**:

The provided context does not specify treatments for arterial tortuosity syndrome. While it mentions resources that address the diagnosis or management of the disorder, such as genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, and palliative care, it does not detail specific treatments [medquad_ans_0000072-5]. Therefore, based on the given information, the exact treatments are not available.

**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

### Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001093-1__dup1` (score=0.0812) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is ava...
2. `medquad_ans_0001092-1__dup1` (score=0.0799) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
3. `medquad_ans_0001094-1__dup1` (score=0.0763) - Charcot-Marie-Tooth disease type 1E (CMT1E) is a form of Charcot-Marie-Tooth disease, which is a group of rare conditions that affect the peripheral nerves. Signs and symptoms of CMT1E generally become apparent betwee...
4. `medquad_ans_0001100-1__dup1` (score=0.0752) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is a...
5. `medquad_ans_0001094-2__dup1` (score=0.0750) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1E? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1E. If the information is ava...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 1C include decreased motor nerve conduction velocity, distal amyotrophy, distal muscle weakness, distal sensory impairment, hypertrophic nerve changes, hyporeflexia, juvenile onset, onion bulb formation, pes cavus, and segmental peripheral demyelination/remyelination [medquad_ans_0001092-1__dup1].

**Supported citations**: `medquad_ans_0001092-1__dup1`
**Unsupported citations**: _none_

### Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000870-1` (score=0.0812) - Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) is a disorder that affects blood vessels in the eyes and heart. The condition generally becomes apparent in infancy or childhood. RAMSVPS d...
2. `medquad_ans_0000870-5` (score=0.0796) - These resources address the diagnosis or management of RAMSVPS: - Calgary Retina Consultants: Retinal Arterial Macroaneurysm - Genetic Testing Registry: Retinal arterial macroaneurysm with supravalvular pulmonic steno...
3. `medquad_ans_0005330-1` (score=0.0781) - Retinal vasculopathy with cerebral leukodystrophy (RVCL) is a rare, genetic condition that primarily affects the central nervous system. Symptoms begin in adulthood (usually in the 40s) and may include loss of vision,...
4. `medquad_ans_0000016-3__dup2` (score=0.0729) - Diabetic retinopathy often has no early warning signs. Don't wait for symptoms. Be sure to have a comprehensive dilated eye exam at least once a year to detect the disease before it causes damage to your vision. Visio...
5. `medquad_ans_0005330-2` (score=0.0714) - What are the signs and symptoms of Retinal vasculopathy with cerebral leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal vasculopathy with cerebral leukodystroph...

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) include vision loss due to bleeding from ruptured macroaneurysms in the retina, shortness of breath, a rapid heartbeat, fatigue, and swelling in the face, feet, or abdomen. These symptoms arise because the retinal macroaneurysms can rupture and spread into other areas of the eye, and the supravalvular pulmonic stenosis impairs blood flow into the lungs, leading to reduced oxygen delivery to the body's tissues [medquad_ans_0000870-1].

**Supported citations**: `medquad_ans_0000870-1`
**Unsupported citations**: _none_

### Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Scores**: context_relevance=1.000, groundedness=0.907, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001838-2` (score=0.0656) - What are the signs and symptoms of Dicarboxylic aminoaciduria? There are no common signs or symptoms of dicarboxylic aminoaciduria. Hypoglycemia, developmental and neurological abnormalities, and obsessive compulsive ...
2. `medquad_ans_0002894-1` (score=0.0638) - What are the signs and symptoms of Histidinuria renal tubular defect? The Human Phenotype Ontology provides the following list of signs and symptoms for Histidinuria renal tubular defect. If the information is availab...
3. `medquad_ans_0000050-1` (score=0.0635) - Aminoacylase 1 deficiency is an inherited disorder that can cause neurological problems; the pattern and severity of signs and symptoms vary widely among affected individuals. Individuals with this condition typically...
4. `medquad_ans_0000168-7` (score=0.0621) - - Proteinuria is a condition in which urine contains a detectable amount of protein. - Proteinuria is a sign of chronic kidney disease (CKD). - Groups at risk for proteinuria include African Americans, American Indian...
5. `medquad_ans_0002733-1` (score=0.0560) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...

**Answer**:

Dicarboxylic aminoaciduria is a rare genetic disorder characterized by the presence of dicarboxylic amino acids in the urine, typically due to autosomal recessive inheritance. It is associated with a range of signs and symptoms, though many individuals have no noticeable symptoms. Reported signs and symptoms include fasting hypoglycemia, intellectual disability, and in rare cases, developmental and neurological abnormalities or obsessive compulsive tendencies. However, in some diagnosed individuals, there are virtually no signs or symptoms. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which compiles information from sources such as Orphanet, and may vary depending on the study population [medquad_ans_0001838-2].

**Supported citations**: `medquad_ans_0001838-2`
**Unsupported citations**: _none_

### Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Scores**: context_relevance=1.000, groundedness=0.900, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000949-5` (score=0.0328) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from Medli...
2. `medquad_ans_0005859-1` (score=0.0323) - Spondylospinal thoracic dysostosis is an extremely rare skeletal disorder characterized by a short, curved spine and fusion of the spinous processes, short thorax with 'crab-like' configuration of the ribs, underdevel...
3. `medquad_ans_0005819-1` (score=0.0317) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
4. `medquad_ans_0005821-1` (score=0.0312) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
5. `medquad_ans_0005822-1` (score=0.0308) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...

**Answer**:

Yes, spondylothoracic dysostosis is inherited in an autosomal recessive manner. [medquad_ans_0005859-1]

**Supported citations**: `medquad_ans_0005859-1`
**Unsupported citations**: _none_

### Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.278, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000542-2` (score=0.0653) - Jervell and Lange-Nielsen syndrome is uncommon; it affects an estimated 1.6 to 6 per 1 million people worldwide. This condition has a higher prevalence in Denmark, where it affects at least 1 in 200,000 people.
2. `medquad_ans_0003296-1` (score=0.0643) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. I...
3. `medquad_ans_0000542-1` (score=0.0623) - Jervell and Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). This disorder is a form of long QT syndrome, which is a heart ...
4. `medquad_ans_0000542-5` (score=0.0620) - These resources address the diagnosis or management of Jervell and Lange-Nielsen syndrome: - Gene Review: Gene Review: Jervell and Lange-Nielsen Syndrome - Genetic Testing Registry: Jervell and Lange-Nielsen syndrome ...
5. `medquad_ans_0000579-2` (score=0.0599) - Langer-Giedion syndrome is a rare condition; its incidence is unknown.

**Answer**:

Jervell and Lange-Nielsen syndrome is a rare condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). It is a form of long QT syndrome, which results in the heart muscle taking longer than usual to recharge between beats. This disorder increases the risk of fainting (syncope) and sudden death, especially beginning in early childhood. It is inherited in an autosomal recessive fashion and has two types: type 1 and type 2. The condition is uncommon globally, affecting an estimated 1.6 to 6 per 1 million people, with a higher prevalence in Denmark, where it affects at least 1 in 200,000 people [medquad_ans_0000542-2, medquad_ans_0003296-1, medquad_ans_0000542-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000542-2, medquad_ans_0003296-1, medquad_ans_0000542-1`

### Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Scores**: context_relevance=1.000, groundedness=0.766, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000218-1` (score=0.0492) - Congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is a rare disorder that affects several parts of the body. It is characterized by a clouding of the lens of the eyes at birth (congenital cataracts) and...
2. `medquad_ans_0000218-5` (score=0.0484) - These resources address the diagnosis or management of CCFDN: - Gene Review: Gene Review: Congenital Cataracts, Facial Dysmorphism, and Neuropathy - Genetic Testing Registry: Congenital Cataracts, Facial Dysmorphism, ...
3. `medquad_ans_0000507-2` (score=0.0476) - The prevalence of hypomyelination and congenital cataract is unknown.
4. `medquad_ans_0000507-3` (score=0.0466) - Mutations in the FAM126A gene cause hypomyelination and congenital cataract. The FAM126A gene provides instructions for making a protein called hyccin, the function of which is not completely understood. Based on the ...
5. `medquad_ans_0000507-1` (score=0.0464) - Hypomyelination and congenital cataract is an inherited condition that affects the nervous system and the eyes. This disease is one of a group of genetic disorders called leukoencephalopathies. Leukoencephalopathies i...

**Answer**:

The provided context does not specify the number of people affected by congenital cataracts, facial dysmorphism, and neuropathy. [medquad_ans_0000218-1] describes the disorder's features and complications but does not include epidemiological data on prevalence or affected population size. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000218-1`
**Unsupported citations**: _none_

### Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Scores**: context_relevance=0.769, groundedness=0.100, answer_relevance=0.631

**Top docs**:

1. `medquad_ans_0000127-1__dup2` (score=0.0648) - Thalassemias (thal-a-SE-me-ahs) are inherited blood disorders. "Inherited" means that the disorder is passed from parents to children through genes. Thalassemias cause the body to make fewer healthy red blood cells an...
2. `medquad_ans_0000127-4__dup2` (score=0.0630) - Family history and ancestry are the two risk factors for thalassemias. Family History Thalassemias are inheritedthat is, the genes for the disorders are passed from parents to their children. If your parents have miss...
3. `medquad_ans_0000127-8` (score=0.0618) - You cant prevent thalassemias because theyre inherited (passed from parents to children through genes). However, prenatal tests can detect these blood disorders before birth. Family genetic studies may help find out w...
4. `medquad_ans_0000381-1__dup1` (score=0.0613) - Fragile X syndrome is the most common form of inherited developmental disability. A problem with a specific gene causes the disease. Normally, the gene makes a protein you need for brain development. But the problem c...
5. `medquad_ans_0000886-1__dup2` (score=0.0593) - Thalassemias are inherited blood disorders. If you have one, your body makes fewer healthy red blood cells and less hemoglobin. Hemoglobin is a protein that carries oxygen to the body. That leads to anemia. Thalassemi...

**Answer**:

The provided context does not mention "alpha thalassemia X-linked intellectual disability syndrome" or any genetic changes related to this condition. Therefore, based on the given information, it is not possible to answer the question.

Answer: Insufficient context [medquad_ans_0000127-1__dup2, medquad_ans_0000127-4__dup2]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000127-1__dup2, medquad_ans_0000127-4__dup2`

### Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Scores**: context_relevance=1.000, groundedness=0.941, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0002622-1` (score=0.0817) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepa...
2. `medquad_ans_0002622-2` (score=0.0783) - What are the signs and symptoms of Glycogen storage disease type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 6. If the information is available, ...
3. `medquad_ans_0000423-5` (score=0.0776) - These resources address the diagnosis or management of glycogen storage disease type III: - Gene Review: Gene Review: Glycogen Storage Disease Type III - Genetic Testing Registry: Glycogen storage disease type III The...
4. `medquad_ans_0000427-5` (score=0.0762) - These resources address the diagnosis or management of glycogen storage disease type VI: - Gene Review: Gene Review: Glycogen Storage Disease Type VI - Genetic Testing Registry: Glycogen storage disease, type VI These...
5. `medquad_ans_0002615-2` (score=0.0736) - What are the signs and symptoms of Glycogen storage disease type 13? Glycogen storage disease type 13 causes muscle pain (myalgia). Individuals with GSD13 also experience exercise intolerance, which means they have di...

**Answer**:

The symptoms of Glycogen storage disease type 6 include low blood sugar (hypoglycemia), an enlarged liver (hepatomegaly), and an increase in the amount of lactic acid in the blood (lactic acidosis), particularly when an individual does not eat for a long time. Short stature and postnatal growth retardation are also associated with the condition. Hypoglycemia and short stature are reported in approximately 90% of patients [medquad_ans_0002622-2]. [medquad_ans_0002622-1]

**Supported citations**: `medquad_ans_0002622-2`, `medquad_ans_0002622-1`
**Unsupported citations**: _none_

### Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Scores**: context_relevance=1.000, groundedness=0.952, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000057-1__dup5` (score=0.0790) - Too much glucose in the blood for a long time can cause diabetes problems. This high blood glucose, also called blood sugar, can damage many parts of the body, such as the heart, blood vessels, eyes, and kidneys. Hear...
2. `medquad_ans_0000015-2__dup2` (score=0.0780) - Diabetes is a serious, life-long disease. It can lead to problems such as heart disease, stroke, vision loss, kidney disease, and nerve damage. More than 8 million people in the United States have type 2 diabetes and ...
3. `medquad_ans_0000032-1__dup4` (score=0.0757) - Diabetes is a disorder of metabolismthe way our bodies use digested food for energy. Most of the food we eat is broken down into glucose, the form of sugar in the blood. Glucose is the body's main source of fuel. Afte...
4. `medquad_ans_0000027-7__dup1` (score=0.0731) - - Diabetes is a complex group of diseases with a variety of causes. Scientists believe genes and environmental factors interact to cause diabetes in most cases. - People with diabetes have high blood glucose, also cal...
5. `medquad_ans_0000037-7` (score=0.0726) - - Diabetes is a complex group of diseases with a variety of causes. Scientists believe genes and environmental factors interact to cause diabetes in most cases. - People with diabetes have high blood glucose, also cal...

**Answer**:

The causes of diabetes are not fully known, but for Type 1 diabetes, it is believed that both genetic factors and environmental factors, such as viruses, are involved in the autoimmune reaction where the body's immune system attacks its own healthy tissues [medquad_ans_0000015-2__dup2]. For Type 2 diabetes, risk factors include being overweight or obese, high blood pressure, abnormal cholesterol levels, being over 45 years of age, having a first-degree relative with diabetes, and belonging to certain ethnic groups such as African American, American Indian or Alaska Native, Asian American or Pacific Islander, or Hispanic American/Latino [medquad_ans_0000015-2__dup2]. Genes also play an important role in a person's risk for developing Type 2 diabetes [medquad_ans_0000015-2__dup2]. However, the

**Supported citations**: `medquad_ans_0000015-2__dup2`
**Unsupported citations**: _none_

### Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Scores**: context_relevance=0.636, groundedness=0.871, answer_relevance=0.782

**Top docs**:

1. `medquad_ans_0000016-21` (score=0.0492) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the lab...
2. `medquad_ans_0000016-6` (score=0.0484) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
3. `medquad_ans_0000016-8` (score=0.0476) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a com...
4. `medquad_ans_0000016-19` (score=0.0466) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although bo...
5. `medquad_ans_0000055-7` (score=0.0464) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need othe...

**Answer**:

Research being done for diabetic retinopathy includes studies conducted by the National Eye Institute (NEI) to find better ways to detect, treat, and prevent vision loss in people with diabetes. This research is carried out through laboratory studies and patient-based studies. For example, researchers are investigating drugs for the treatment of proliferative retinopathy that may reduce the need for laser surgery [medquad_ans_0000016-21]. Additionally, clinical trials are exploring medications such as injections that block a protein causing abnormal blood vessel growth and fluid leakage, which can help treat macular edema and restore retinal thickness [medquad_ans_0000055-7].

**Supported citations**: `medquad_ans_0000016-21`, `medquad_ans_0000055-7`
**Unsupported citations**: _none_

### Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Scores**: context_relevance=0.800, groundedness=0.963, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.0653) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart an...
2. `medquad_ans_0000007-4__dup1` (score=0.0648) - People at risk for ARDS have a condition or illness that can directly or indirectly injure their lungs. Direct Lung Injury Conditions that can directly injure the lungs include: Pneumonia. This is an infection in the ...
3. `medquad_ans_0000007-3__dup1` (score=0.0635) - Many conditions or factors can directly or indirectly injure the lungs and lead to ARDS. Some common ones are: Sepsis. This is a condition in which bacteria infect the bloodstream. Pneumonia. This is an infection in t...
4. `medquad_ans_0000137-1__dup1` (score=0.0618) - Acute respiratory distress syndrome (ARDS) is a life-threatening lung condition that prevents enough oxygen from getting to the lungs and into the blood. People who develop ARDS often are very ill with another disease...
5. `medquad_ans_0000007-1__dup1` (score=0.0606) - ARDS, or acute respiratory distress syndrome, is a lung condition that leads to low oxygen levels in the blood. ARDS can be life threatening because your body's organs need oxygen-rich blood to work well. People who d...

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles in the air from fires or toxic fumes. These substances can damage the eyes and respiratory system and may worsen chronic heart and lung diseases. Symptoms of acute inhalation injuries include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, it may worsen [medquad_ans_0000516-1__dup1].

**Supported citations**: `medquad_ans_0000516-1__dup1`
**Unsupported citations**: _none_

### Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Scores**: context_relevance=0.625, groundedness=0.961, answer_relevance=0.425

**Top docs**:

1. `medquad_ans_0000134-9` (score=0.0489) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing Zollinger-Ellison syndrome.
2. `medquad_ans_0000134-4__dup2` (score=0.0487) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has ...
3. `medquad_ans_0000134-3__dup2` (score=0.0476) - Experts do not know the exact cause of Zollinger-Ellison syndrome. About 25 to 30 percent of gastrinomas are caused by an inherited genetic disorder called multiple endocrine neoplasia type 1 (MEN1).1 MEN1 causes horm...
4. `medquad_ans_0000134-10` (score=0.0469) - - Zollinger-Ellison syndrome is a rare disorder that occurs when one or more tumors form in the pancreas and duodenum. - Experts do not know the exact cause of Zollinger-Ellison syndrome. - About 25 to 30 percent of g...
5. `medquad_ans_0000134-1__dup3` (score=0.0462) - Zollinger-Ellison syndrome is a rare disorder that occurs when one or more tumors form in the pancreas and duodenum. The tumors, called gastrinomas, release large amounts of gastrin that cause the stomach to produce l...

**Answer**:

Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people [medquad_ans_0000134-4__dup2].

**Supported citations**: `medquad_ans_0000134-4__dup2`
**Unsupported citations**: _none_

### Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Scores**: context_relevance=0.857, groundedness=0.736, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000974-1` (score=0.0817) - Tetrahydrobiopterin deficiency is a rare disorder characterized by a shortage (deficiency) of a molecule called tetrahydrobiopterin or BH4. This condition alters the levels of several substances in the body, including...
2. `medquad_ans_0006044-2` (score=0.0809) - What are the signs and symptoms of Tetrahydrobiopterin deficiency? Infants with tetrahydrobiopterin (BH4) deficiency typically appear normal and healthy at birth. Neurological signs and symptoms usually become apparen...
3. `medquad_ans_0000974-3` (score=0.0789) - Tetrahydrobiopterin deficiency can be caused by mutations in one of several genes, including GCH1, PCBD1, PTS, and QDPR. These genes provide instructions for making enzymes that help produce and recycle tetrahydrobiop...
4. `medquad_ans_0006044-1` (score=0.0779) - Tetrahydrobiopterin (BH4) deficiency is a neurological condition caused by an inborn error of metabolism. BH4 is a substance in the body that enhances the action of other enzymes. Deficiency of BH4 leads to abnormally...
5. `medquad_ans_0001860-1` (score=0.0762) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...

**Answer**:

The provided context does not mention any treatments for tetrahydrobiopterin deficiency. Therefore, based on the available information, it is not possible to answer the question about treatments. [medquad_ans_0000974-1] [medquad_ans_0006044-2]

**Supported citations**: `medquad_ans_0000974-1`, `medquad_ans_0006044-2`
**Unsupported citations**: _none_

### Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Scores**: context_relevance=0.875, groundedness=0.895, answer_relevance=0.375

**Top docs**:

1. `medquad_ans_0000736-2` (score=0.0492) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
2. `medquad_ans_0000736-1` (score=0.0481) - Nonsyndromic holoprosencephaly is an abnormality of brain development that also affects the head and face. Normally, the brain divides into two halves (hemispheres) during early development. Holoprosencephaly occurs w...
3. `medquad_ans_0000150-2__dup1` (score=0.0479) - There is no standard course of treatment for holoprosencephaly. Treatment is symptomatic and supportive.
4. `medquad_ans_0000150-4__dup1` (score=0.0469) - The NINDS supports and conducts a wide range of studies that focus on identifying and learning more about the factors involved in normal brain development. Recent research has identified specific genes that cause holo...
5. `medquad_ans_0000150-1__dup1` (score=0.0462) - Holoprosencephaly is a disorder caused by the failure of the prosencephalon (the embryonic forebrain) to sufficiently divide into the double lobes of the cerebral hemispheres. The result is a single-lobed brain struct...

**Answer**:

Nonsyndromic holoprosencephaly affects an estimated 1 in 10,000 newborns [medquad_ans_0000736-2]. This represents approximately 25 to 50 percent of all cases of holoprosencephaly [medquad_ans_0000736-2]. Therefore, the prevalence of nonsyndromic holoprosencephaly is about 1 in 10,000 newborns.

**Supported citations**: `medquad_ans_0000736-2`
**Unsupported citations**: _none_

### Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Scores**: context_relevance=0.636, groundedness=0.904, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000525-4` (score=0.0656) - Susceptibility to intrahepatic cholestasis of pregnancy is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to increase the risk of developing the disorde...
2. `medquad_ans_0000118-5` (score=0.0645) - These resources address the diagnosis or management of benign recurrent intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Benign recurrent intrahepatic cholestasis 1 -...
3. `medquad_ans_0000829-5` (score=0.0635) - These resources address the diagnosis or management of progressive familial intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Progressive familial intrahepatic cholest...
4. `medquad_ans_0000525-2` (score=0.0618) - Intrahepatic cholestasis of pregnancy is estimated to affect 1 percent of women of Northern European ancestry. The condition is more common in certain populations, such as women of Araucanian Indian ancestry in Chile ...
5. `medquad_ans_0005084-1` (score=0.0607) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching...

**Answer**:

The genetic changes related to progressive familial intrahepatic cholestasis type 2 (PFIC2) are mutations in the ABCB11 gene [medquad_ans_0005084-1]. PFIC2 is inherited in an autosomal recessive pattern [medquad_ans_0005084-1].

**Supported citations**: `medquad_ans_0005084-1`
**Unsupported citations**: _none_

### Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.990, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.0820) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below in...
2. `medquad_ans_0003845-2` (score=0.0782) - What are the signs and symptoms of McCune Albright syndrome? People with McCune Albright syndrome (MAS) may have symptoms related to bones, the endocrine system, and/or skin. The symptoms can range from mild to severe...
3. `medquad_ans_0003814-1` (score=0.0777) - Marshall-Smith syndrome is a malformation syndrome characterized by advanced bone age, failure to thrive, respiratory problems, dysmorphic facial features, and variable mental retardation. Less than 40 cases have been...
4. `medquad_ans_0002837-1` (score=0.0744) - Hereditary mucoepithelial dysplasia (HMD) is a condition that affects the skin, hair, mucosa (areas of the body that are lined with mucus), gums (gingiva), eyes, nose and lungs. Symptoms typically begin in infancy and...
5. `medquad_ans_0003814-2` (score=0.0740) - What are the signs and symptoms of Marshall-Smith syndrome? Marshall-Smith syndrome is characterized by accelerated skeletal maturation, relative failure to thrive, respiratory difficulties, mental retardation, and un...

**Answer**:

The symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

### Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Scores**: context_relevance=0.857, groundedness=0.837, answer_relevance=0.971

**Top docs**:

1. `medquad_ans_0000453-1` (score=0.0328) - Hepatic veno-occlusive disease with immunodeficiency (also called VODI) is a hereditary disorder of the liver and immune system. Its signs and symptoms appear after the first few months of life. Hepatic veno-occlusive...
2. `medquad_ans_0000453-5` (score=0.0323) - These resources address the diagnosis or management of VODI: - Gene Review: Gene Review: Hepatic Veno-Occlusive Disease with Immunodeficiency - Genetic Testing Registry: Hepatic venoocclusive disease with immunodefici...
3. `medquad_ans_0002804-1` (score=0.0317) - What are the signs and symptoms of Hepatic venoocclusive disease with immunodeficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Hepatic venoocclusive disease with immunodefic...
4. `medquad_ans_0000850-5` (score=0.0310) - These resources address the diagnosis or management of pulmonary veno-occlusive disease: - Genetic Testing Registry: Pulmonary veno-occlusive disease These resources from MedlinePlus offer information about the diagno...
5. `medquad_ans_0004534-1` (score=0.0308) - Omenn syndrome is an autosomal recessive form of severe combined immunodeficiency (SCID) characterized by erythroderma (skin redness), desquamation (peeling skin), alopecia (hair loss), chronic diarrhea, failure to th...

**Answer**:

Yes, hepatic veno-occlusive disease with immunodeficiency is inherited in an autosomal recessive pattern. This is supported by the presence of "autosomal recessive inheritance" in the list of signs and symptoms provided in the context [medquad_ans_0002804-1].

**Supported citations**: `medquad_ans_0002804-1`
**Unsupported citations**: _none_

### Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Scores**: context_relevance=1.000, groundedness=0.865, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000003_2-3` (score=0.0729) - Signs of gastrointestinal stromal tumors include blood in the stool or vomit. These and other signs and symptoms may be caused by a GIST or by other conditions. Check with your doctor if you have any of the following:...
2. `medquad_ans_0005902-1` (score=0.0656) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how o...
3. `medquad_ans_0001768-1` (score=0.0643) - What are the signs and symptoms of Dehydrated hereditary stomatocytosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Dehydrated hereditary stomatocytosis. If the information is a...
4. `medquad_ans_0000221-1__dup1` (score=0.0635) - Paraneoplastic syndromes are a group of rare disorders that are triggered by an abnormal immune system response to a cancerous tumor known as a "neoplasm." Paraneoplastic syndromes are thought to happen when cancer-fi...
5. `medquad_ans_0000573-1__dup1` (score=0.0628) - Lymphoma is a cancer of a part of the immune system called the lymph system. There are many types of lymphoma. One type is Hodgkin disease. The rest are called non-Hodgkin lymphomas. Non-Hodgkin lymphomas begin when a...

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. These signs and symptoms are based on the Human Phenotype Ontology, which compiles data from sources such as Orphanet. The frequency of these symptoms in patients with Stomatocytosis I is estimated, with some symptoms reported in a percentage of patients, though specific frequencies are not fully detailed in the provided context [medquad_ans_0005902-1].

**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

### Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Scores**: context_relevance=0.667, groundedness=0.246, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000159-8` (score=0.0492) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed...
2. `medquad_ans_0003688-1` (score=0.0479) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, h...
3. `medquad_ans_0000159-7` (score=0.0465) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may n...
4. `medquad_ans_0000159-1__dup3` (score=0.0465) - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). SLE is an autoimmune diseasea disorder in which the bodys immune system attacks the bodys own cells and organs. Up to 60 pe...
5. `medquad_ans_0000159-6` (score=0.0464) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...

**Answer**:

Treatment for lupus nephritis includes medications that suppress the immune system to prevent damage to the kidneys. Standard treatment typically involves a corticosteroid, such as prednisone, to reduce inflammation, and an immunosuppressive medication like cyclophosphamide or mycophenolate mofetil, used in combination with prednisone. People with lupus nephritis and high blood pressure may also need medications to lower blood pressure, which can slow the progression of kidney disease. In some cases, treatment may involve dialysis or a kidney transplant. Additionally, individuals with kidney disease may need to adjust their diet, particularly by reducing sodium intake, under the guidance of a healthcare provider [medquad_ans_0000159-8, medquad_ans_0003688-1, medquad_ans_0000159-7].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000159-8, medquad_ans_0003688-1, medquad_ans_0000159-7`

### Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.440, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0005317-1` (score=0.0812) - Restless legs syndrome is a neurological condition that causes an irresistible urge to move the legs. The movement is triggered by strange or uncomfortable feelings, which occur mostly while the affected person is sit...
2. `medquad_ans_0000114-6` (score=0.0798) - Restless legs syndrome (RLS) has no cure. If a condition or medicine triggers RLS, it may go away or get better if the trigger is relieved or stopped. RLS can be treated. The goals of treatment are to: Prevent or reli...
3. `medquad_ans_0000246-1__dup2` (score=0.0782) - Restless legs syndrome (RLS) is a neurological disorder characterized by unpleasant sensations in the legs and an uncontrollable, and sometimes overwhelming, urge to move them for relief. Individuals affected with the...
4. `medquad_ans_0000869-2` (score=0.0779) - Restless legs syndrome is one of the most common sleep and movement disorders. It affects an estimated 5 to 10 percent of adults and 2 to 4 percent of children in the United States. For unknown reasons, the disorder a...
5. `medquad_ans_0000776-1__dup2` (score=0.0772) - Restless legs syndrome (RLS) causes a powerful urge to move your legs. Your legs become uncomfortable when you are lying down or sitting. Some people describe it as a creeping, crawling, tingling, or burning sensation...

**Answer**:

Treatments for Restless Legs Syndrome (RLS) include lifestyle changes and, in more severe cases, daily medicines. For mild cases, lifestyle changes may be the only treatment needed. These include avoiding substances that can trigger RLS symptoms, such as alcohol and tobacco, and informing a doctor about all medications being taken to identify any that may cause or worsen RLS. Good sleep habits are also recommended, such as keeping the sleep environment cool, quiet, comfortable, and dark; removing distractions like TVs or phones; maintaining a consistent sleep schedule by going to bed and waking up at the same time every day; and avoiding long periods of lying awake at night. Some individuals may find it helpful to go to bed later and wake up later. In more severe cases, daily medicines are typically used. The goals of treatment are to prevent or relieve symptoms, increase sleep duration, improve sleep quality, and treat or correct any underlying conditions that may trigger or wors

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Scores**: context_relevance=0.500, groundedness=0.717, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000127-1__dup3` (score=0.0574) - A brain tumor is a growth of abnormal cells in the tissues of the brain. Brain tumors can be benign, with no cancer cells, or malignant, with cancer cells that grow quickly. Some are primary brain tumors, which start ...
2. `medquad_ans_0000643-1__dup2` (score=0.0514) - Neuroblastoma is a cancer that forms in your nerve tissue. It usually begins in the adrenal glands, which sit atop your kidneys. It may also begin in your neck, chest or spinal cord. The cancer often begins in early c...
3. `medquad_ans_0000008-12` (score=0.0501) - When cancer spreads from its original location in the breast to another part of the body such as the brain, it is called metastatic breast cancer, not brain cancer. Doctors sometimes call this "distant" disease.
4. `medquad_ans_0000063-16` (score=0.0494) - When the cancer spreads from its original tumor location in the skin to another part of the body such as the brain, it is called metastatic skin cancer. It is not the same as a cancer that started in the brain (brain ...
5. `medquad_ans_0002574-1` (score=0.0489) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...

**Answer**:

The provided context does not contain information about glioblastoma. Therefore, it is not possible to answer the question based on the given text. [medquad_ans_0002574-1] only mentions glioma as a type of brain tumor developing from glial cells but does not specify glioblastoma.

**Supported citations**: `medquad_ans_0002574-1`
**Unsupported citations**: _none_

### Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Scores**: context_relevance=0.333, groundedness=0.971, answer_relevance=0.333

**Top docs**:

1. `medquad_ans_0005496-1` (score=0.0814) - Schizencephaly is a rare congenital (present from birth) brain malformation in which abnormal slits or clefts form in the cerebral hemispheres of the brain. The signs and symptoms of this condition may include develop...
2. `medquad_ans_0000251-1__dup2` (score=0.0807) - Schizencephaly is an extremely rare developmental birth defect characterized by abnormal slits, or clefts, in the cerebral hemispheres of the brain. Babies with clefts in both hemispheres (called bilateral clefts) com...
3. `medquad_ans_0000251-3__dup1` (score=0.0799) - The prognosis for individuals with schizencephaly varies depending on the size of the clefts and the extent of neurological disabilities.
4. `medquad_ans_0000251-4__dup1` (score=0.0774) - The NINDS conducts and supports a wide range of studies that explore the mechanisms of normal brain development. The knowledge gained from these fundamental studies provides the foundation for understanding how to pre...
5. `medquad_ans_0005496-6` (score=0.0751) - How might schizencephaly be treated? The best treatment options for people with schizencephaly depend on many factors, including the severity of the condition and the signs and symptoms present. For example, people wi...

**Answer**:

The exact cause of schizencephaly is unknown, but it has been linked to a variety of genetic and non-genetic factors [medquad_ans_0005496-1].

**Supported citations**: `medquad_ans_0005496-1`
**Unsupported citations**: _none_

### Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Scores**: context_relevance=0.900, groundedness=0.831, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000396-1__dup2` (score=0.0565) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...
2. `medquad_ans_0000028-1__dup3` (score=0.0554) - Amyotrophic lateral sclerosis (ALS) is a nervous system disease that attacks nerve cells called neurons in your brain and spinal cord. These neurons transmit messages from your brain and spinal cord to your voluntary ...
3. `medquad_ans_0000309-5__dup1` (score=0.0484) - Is genetic testing available for amyotrophic lateral sclerosis? Yes. Clinical genetic testing is currently available for several genes in which mutations are known to cause ALS. Genetic testing on a research basis is ...
4. `medquad_ans_0000309-3__dup1` (score=0.0481) - What causes amyotrophic lateral sclerosis? In approximately 90-95% of cases the cause of amyotrophic lateral sclerosis (ALS) is unknown and is sporadic (occurring in individuals with no history of the condition in the...
5. `medquad_ans_0003331-3` (score=0.0479) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...

**Answer**:

The provided context does not specify the exact genetic changes related to amyotrophic lateral sclerosis (ALS). It mentions that genetic testing is available for several genes in which mutations are known to cause ALS and that research testing is available for select susceptibility genes associated with ALS [medquad_ans_0000309-5__dup1]. However, the specific genes or types of mutations are not detailed in the given text. Therefore, based on the available information, the exact genetic changes related to ALS cannot be determined.

**Supported citations**: `medquad_ans_0000309-5__dup1`
**Unsupported citations**: _none_

### Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.966, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000581-2` (score=0.0812) - Laron syndrome is a rare disorder. About 350 people have been diagnosed with the condition worldwide. The largest single group of affected individuals (about 100 people) lives in an area of southern Ecuador.
2. `medquad_ans_0003473-6` (score=0.0807) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injectio...
3. `medquad_ans_0003473-1` (score=0.0791) - Laron syndrome is a condition that occurs when the body is unable to utilize growth hormone. It is primarily characterized by short stature. Other signs and symptoms vary but may include reduced muscle strength and en...
4. `medquad_ans_0000581-5` (score=0.0779) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Te...
5. `medquad_ans_0003473-5` (score=0.0769) - How is Laron syndrome diagnosed? A diagnosis of Laron syndrome is often suspected based on the presence of characteristic signs and symptoms. Additional testing can then be ordered to confirm the diagnosis and rule ou...

**Answer**:

The only specific treatment available for Laron syndrome is subcutaneous injections of insulin-like growth factor 1 (IGF-1), a growth-promoting hormone. IGF-1 stimulates linear growth (height), improves brain growth, and addresses metabolic abnormalities caused by long-term IGF-1 deficiency. It has also been shown to raise blood glucose levels, reduce cholesterol, and increase muscle growth. IGF-1 and growth hormone (GH) levels should be closely monitored during treatment because overdosage can cause various health problems [medquad_ans_0003473-6]. Treatment is focused on improving growth and generally includes IGF-1 injections [medquad_ans_0003473-1].

**Supported citations**: `medquad_ans_0003473-6`, `medquad_ans_0003473-1`
**Unsupported citations**: _none_

### Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Scores**: context_relevance=1.000, groundedness=0.928, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000016-6` (score=0.0787) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
2. `medquad_ans_0000016-15` (score=0.0784) - During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels of blood s...
3. `medquad_ans_0000016-8` (score=0.0781) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a com...
4. `medquad_ans_0000016-4__dup2` (score=0.0775) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...
5. `medquad_ans_0000055-7` (score=0.0771) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need othe...

**Answer**:

Treatments for diabetic retinopathy include:

- No treatment is needed during the first three stages, unless macular edema is present [medquad_ans_0000016-15].
- For macular edema, prompt treatment with anti-VEGF drugs, with or without laser treatment, results in better vision than laser treatment alone [medquad_ans_0000016-4__dup2].
- Proliferative retinopathy is treated with scatter laser treatment, which involves placing 1,000 to 2,000 laser burns in the retina away from the macula to shrink abnormal blood vessels. Two or more sessions are usually required [medquad_ans_0000016-15].
- If bleeding is severe, a surgical procedure called vitrectomy may be needed to remove blood from the center of the eye [medquad*ans*

**Supported citations**: `medquad_ans_0000016-15`, `medquad_ans_0000016-4__dup2`
**Unsupported citations**: _none_

### Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Scores**: context_relevance=0.857, groundedness=0.550, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0001421-1` (score=0.0812) - Collagenous colitis is a type of inflammatory bowel disease that affects the colon. It is a form of microscopic colitis, which means that the inflammation is only visible when a biopsy is examined under a microscope; ...
2. `medquad_ans_0001421-3` (score=0.0809) - How might collagenous colitis be treated? Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases the condition may resolve on its own (spontaneous re...
3. `medquad_ans_0001421-2` (score=0.0784) - What are the signs and symptoms of collagenous colitis? All individuals with collagenous colitis experience chronic, watery, non-bloody diarrhea which is what typically prompts individuals to seek medical attention. O...
4. `medquad_ans_0000116-8` (score=0.0734) - To help reduce symptoms, a health care provider may recommend the following dietary changes: - avoid foods and drinks that contain caffeine or artificial sugars - drink plenty of liquids to prevent dehydration during ...
5. `medquad_ans_0000122-1__dup3` (score=0.0732) - Ulcerative colitis is a chronic, or long lasting, disease that causes inflammationirritation or swellingand sores called ulcers on the inner lining of the large intestine. Ulcerative colitis is a chronic inflammatory ...

**Answer**:

Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases, the condition may resolve on its own (spontaneous remission), although most people continue to have ongoing or occasional diarrhea. Dietary changes are usually tried first to alleviate symptoms, including a reduced-fat diet, eliminating foods that contain caffeine and lactose, and avoiding over-the-counter pain relievers such as ibuprofen or aspirin. If these changes alone are not enough, medications can be used to help control symptoms. Prescription anti-inflammatory medications such as mesalamine and sulfasalazine may help reduce swelling. Steroids including budesonide and prednisone can be used to reduce inflammation, but they are usually only used to control sudden attacks of diarrhea. Long-term use of steroids is typically avoided because of unwanted side effects. Anti-diarrheal medications such as bismuth subsalicylate, diphenoxylate with

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Scores**: context_relevance=0.667, groundedness=0.827, answer_relevance=0.844

**Top docs**:

1. `medquad_ans_0000418-6` (score=0.0489) - Safe and effective prescription drugs are available to treat both Trichinella infection and the symptoms that occur as a result of infection. Treatment should begin as soon as possible; a doctor will make the decision...
2. `medquad_ans_0000418-1__dup2` (score=0.0487) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...
3. `medquad_ans_0000418-2__dup1` (score=0.0476) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
4. `medquad_ans_0000418-5__dup1` (score=0.0466) - A diagnosis of trichinellosis is made in patients whose signs and symptoms are compatible with trichinellosis, have a positive laboratory test for Trichinella, and who can recall eating raw or undercooked pork or wild...
5. `medquad_ans_0000418-7` (score=0.0464) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause ...

**Answer**:

To prevent Trichinellosis (also known as trichinosis), it is essential to avoid consuming raw or undercooked meat, particularly from wild game animals such as bear or cougar, and from domestic pigs or wild boar. Proper cooking of meat to safe internal temperatures kills the Trichinella parasites. Additionally, commercial and home freezing of pork can reduce the risk of infection. Public awareness of the dangers of eating raw or undercooked meat products also plays a key role in prevention. The risk of trichinellosis from commercially raised and properly prepared pork is very low due to improved pig-raising practices and freezing methods [medquad_ans_0000418-2__dup1]. Eating undercooked wild game, especially bear meat, remains a significant risk factor [medquad_ans_0000418-2__dup1].

**Supported citations**: `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

### Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Scores**: context_relevance=0.750, groundedness=0.277, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000182-1__dup3` (score=0.0812) - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the normal components, or cells, of the liver and causes inflammation and liver damage. The immune system normally prot...
2. `medquad_ans_0000182-3__dup1` (score=0.0801) - A combination of autoimmunity, environmental triggers, and a genetic predisposition can lead to autoimmune hepatitis.
3. `medquad_ans_0000182-5__dup1` (score=0.0777) - Autoimmune hepatitis is classified into several types. Type 1 autoimmune hepatitis is the most common form in North America. Type 1 can occur at any age; however, it most often starts in adolescence or young adulthood...
4. `medquad_ans_0000182-10` (score=0.0774) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing autoimmune hepatitis.
5. `medquad_ans_0000182-11` (score=0.0767) - - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the liver and causes inflammation and damage. - Autoimmune hepatitis is a serious condition that may worsen over time...

**Answer**:

Autoimmune hepatitis is caused by a combination of autoimmunity, environmental triggers, and a genetic predisposition [doc_id: medquad_ans_0000182-3__dup1].

**Supported citations**: _none_
**Unsupported citations**: `doc_id: medquad_ans_0000182-3__dup1`

### Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Scores**: context_relevance=0.800, groundedness=0.923, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000186-1__dup2` (score=0.0556) - Brain tumors are abnormal growths inside the skull. They are among the most common types of childhood cancers. Some are benign tumors, which aren't cancer. They can still be serious. Malignant tumors are cancerous. Ch...
2. `medquad_ans_0000006_3-1` (score=0.0492) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
3. `medquad_ans_0000006_3-4` (score=0.0484) - The signs and symptoms of astrocytomas are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain or spinal cord. - The size of the tumor. - How fast the tumor gr...
4. `medquad_ans_0004883-4` (score=0.0469) - Are pilocytic astrocytomas inherited? Pilocytic astrocytomas are typically sporadic, occurring by chance in individuals with no history of the condition in the family. Sporadic abnormalities are not inherited from a p...
5. `medquad_ans_0004883-1` (score=0.0466) - Pilocytic astrocytoma is an often benign, slow-growing tumor of the brain or spinal cord. The tumor may be in the form of a cyst and usually does not spread to nearby tissues. Symptoms vary depending upon the size and...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They arise from star-shaped brain cells called astrocytes, which are a type of glial cell that supports and protects nerve cells. Astrocytomas are a type of glioma and are the most common type of glioma diagnosed in children. They can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1]. The signs and symptoms vary among children, and diagnosis typically involves imaging tests and surgery, with most cases diagnosed and removed surgically [medquad_ans_0000006_3-1]. The cause of most childhood brain tumors, including astrocytomas, is not known [medquad_ans_0000006_3-1].

**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

### Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Scores**: context_relevance=0.571, groundedness=0.000, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000010-1__dup4` (score=0.0594) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. Thes...
2. `medquad_ans_0000040-10` (score=0.0550) - Acute leukemia gets worse quickly. In chronic leukemia, symptoms develop gradually and are generally not as severe as in acute leukemia.
3. `medquad_ans_0000040-15` (score=0.0547) - There are no standard or over-the-counter tests for leukemia. Your doctor can request lab analyses for leukemia that include blood tests that check the levels and types of blood cells and look for changes in the shape...
4. `medquad_ans_0000040-9` (score=0.0540) - Leukemia is a cancer of the blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white ...
5. `medquad_ans_0000188-1__dup2` (score=0.0533) - Leukemia is cancer of the white blood cells. It is the most common type of childhood cancer. Your blood cells form in your bone marrow. White blood cells help your body fight infection. In leukemia, the bone marrow pr...

**Answer**:

The provided context does not contain any information about atypical chronic myelogenous leukemia. Therefore, it is not possible to answer the question based on the given text. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Scores**: context_relevance=0.125, groundedness=0.000, answer_relevance=0.625

**Top docs**:

1. `medquad_ans_0000770-4` (score=0.0656) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0001083-4` (score=0.0645) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000784-4` (score=0.0635) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000349-4` (score=0.0625) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000101-4` (score=0.0615) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain any information about genetic changes related to trisomy 13. Therefore, the question cannot be answered based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Scores**: context_relevance=0.714, groundedness=0.953, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004415-3` (score=0.0814) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respirato...
2. `medquad_ans_0000218-2__dup1` (score=0.0812) - There is currently no cure for Niemann-Pick disease. Treatment is supportive. Children usually die from infection or progressive neurological loss. There is currently no effective treatment for persons with type A. Bo...
3. `medquad_ans_0004415-1` (score=0.0794) - Niemann-Pick disease is an inherited condition involving lipid metabolism, which is the breakdown, transport, and use of fats and cholesterol in the body. In people with this condition, abnormal lipid metabolism cause...
4. `medquad_ans_0000731-1` (score=0.0774) - Niemann-Pick disease is a condition that affects many body systems. It has a wide range of symptoms that vary in severity. Niemann-Pick disease is divided into four main types: type A, type B, type C1, and type C2. Th...
5. `medquad_ans_0000218-1__dup1` (score=0.0769) - Niemann-Pick disease (NP) refers to a group of inherited metabolic disorders known as lipid storage diseases. Lipids (fatty materials such as waxes, fatty acids, oils, and cholesterol) and proteins are usually broken ...

**Answer**:

There is no specific treatment for Niemann-Pick disease type A. Treatment is supportive and may include care from specialists such as a pulmonologist for respiratory problems, a cardiologist for heart problems, liver and spleen specialists, nutritionists, physical therapists, and gastroenterologists, as well as learning specialists [medquad_ans_0004415-3]. There is currently no effective treatment for persons with type A, and no cure exists for the disease [medquad_ans_0000218-2__dup1]. Restricting one's diet does not prevent the buildup of lipids in cells and tissues [medquad_ans_0000218-2__dup1]. Enzyme replacement and gene therapies may be helpful for type B, but not for type A [medquad_ans_0000218-2__dup1].

**Supported citations**: `medquad_ans_0004415-3`, `medquad_ans_0000218-2__dup1`
**Unsupported citations**: _none_

### Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.530, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001714-1` (score=0.0820) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial poly...
2. `medquad_ans_0000270-5` (score=0.0791) - These resources address the diagnosis or management of Dandy-Walker malformation: - Genetic Testing Registry: Dandy-Walker syndrome - National Hydrocephalus Foundation: Treatment of Hydrocephalus These resources from ...
3. `medquad_ans_0001708-1` (score=0.0787) - What are the signs and symptoms of Dandy-Walker like malformation with atrioventricular septal defect? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker like malformation ...
4. `medquad_ans_0000270-4` (score=0.0781) - Most cases of Dandy-Walker malformation are sporadic, which means they occur in people with no history of the disorder in their family. A small percentage of cases seem to run in families; however, Dandy-Walker malfor...
5. `medquad_ans_0000270-2` (score=0.0761) - Dandy-Walker malformation is estimated to affect 1 in 10,000 to 30,000 newborns.

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include:

- Dandy-Walker malformation (90%)
- Postaxial hand polydactyly (90%)
- Agenesis of cerebellar vermis
- Aortic valve stenosis
- Autosomal recessive inheritance
- Chorioretinal atrophy
- Cranial nerve paralysis
- Depressed nasal bridge
- Dilated fourth ventricle
- Dolichocephaly
- Elevated imprint of the transverse sinuses
- Frontal bossing
- Hydrocephalus
- Low-set ears
- Macrocephaly
- Microretrognathia
- Nystagmus
- Partial absence of cerebellar vermis
- Patent ductus arteriosus
- Posterior embryotoxon
- Posterior fossa cyst at the fourth ventricle
- Small pal

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Scores**: context_relevance=0.800, groundedness=0.287, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000573-1__dup1` (score=0.0763) - Lymphoma is a cancer of a part of the immune system called the lymph system. There are many types of lymphoma. One type is Hodgkin disease. The rest are called non-Hodgkin lymphomas. Non-Hodgkin lymphomas begin when a...
2. `medquad_ans_0000040-12` (score=0.0760) - Myeloma and lymphoma are other types of blood cancers. Both are common among older adults and occur more often in men than women. Myeloma affects plasma cells, a type of white blood cells typically found in the bone m...
3. `medquad_ans_0000130-2__dup2` (score=0.0663) - Many factors can cause thrombocytopenia (a low platelet count). The condition can be inherited or acquired. "Inherited" means your parents pass the gene for the condition to you. "Acquired" means you aren't born with ...
4. `medquad_ans_0000483-1__dup1` (score=0.0658) - Hodgkin disease is a type of lymphoma. Lymphoma is a cancer of a part of the immune system called the lymph system. The first sign of Hodgkin disease is often an enlarged lymph node. The disease can spread to nearby l...
5. `medquad_ans_0003789-1` (score=0.0656) - Mantle cell lymphoma (MCL) belongs to a group of diseases known as non-Hodgkins lymphomas (NHL), which are cancers that affect the the lymphatic system (part of the immune system). MCL accounts for 6% of all non-Hodgk...

**Answer**:

The provided context does not mention Mantle cell lymphoma or its causes. Therefore, it is not possible to answer the question based on the given information.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.850, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000589-5` (score=0.0164) - These resources address the diagnosis or management of Legg-Calv-Perthes disease: - National Osteonecrosis Foundation - Seattle Children's Hospital These resources from MedlinePlus offer information about the diagnosi...
2. `medquad_ans_0001143-1` (score=0.0164) - CHILD syndrome, also known as congenital hemidysplasia with ichthyosiform erythroderma and limb defects, is a genetic condition that is typically characterized by large patches of skin that are red and inflamed (eryth...
3. `medquad_ans_0000589-3` (score=0.0161) - Legg-Calv-Perthes disease is usually not caused by genetic factors. The cause in these cases is unknown. In a small percentage of cases, mutations in the COL2A1 gene cause the bone abnormalities characteristic of Legg...
4. `medquad_ans_0000261-4__dup1` (score=0.0161) - How is Alport syndrome inherited? Alport syndrome can have different inheritance patterns. About 80 percent of cases are caused by mutations in the COL4A5 gene and are inherited in an X-linked recessive pattern. This ...
5. `medquad_ans_0000589-1` (score=0.0159) - Legg-Calv-Perthes disease is a bone disorder that affects the hips. Usually, only one hip is involved, but in about 10 percent of cases, both hips are affected. Legg-Calv-Perthes disease begins in childhood, typically...

**Answer**:

Legg-Calv-Perthes disease is usually not caused by genetic factors, and the cause in most cases is unknown. However, in a small percentage of cases, mutations in the COL2A1 gene may be involved, which suggests a possible genetic component in those specific instances [medquad_ans_0000589-3]. Therefore, while it is not typically inherited, there can be rare genetic cases linked to COL2A1 mutations. [medquad_ans_0000589-3]

**Supported citations**: `medquad_ans_0000589-3`
**Unsupported citations**: _none_

### Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.527, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000647-2` (score=0.0653) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigra...
2. `medquad_ans_0000647-1` (score=0.0638) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the sub...
3. `medquad_ans_0003566-1` (score=0.0628) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to def...
4. `medquad_ans_0003965-2` (score=0.0611) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the ta...
5. `medquad_ans_0000181-1__dup2` (score=0.0608) - Leukodystrophy refers to progressive degeneration of the white matter of the brain due to imperfect growth or development of the myelin sheath, the fatty covering that acts as an insulator around nerve fiber. Myelin, ...

**Answer**:

Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells, particularly in the nervous system cells that produce myelin, the substance that insulates and protects nerves. This accumulation leads to progressive destruction of white matter throughout the nervous system, including the brain, spinal cord, and peripheral nerves. The damage results in progressive deterioration of intellectual functions and motor skills, such as the ability to walk, loss of sensation in the extremities (peripheral neuropathy), incontinence, seizures, paralysis, inability to speak, blindness, and hearing loss. Eventually, individuals may lose awareness of their surroundings and become unresponsive. Effects of sulfatide accumulation have also been reported on other organs, most commonly the gallbladder. The most common form, affecting about 50 to 60 percent of individuals, is the late infantile form, which typically appears in the second year of life and leads to

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005135-1` (score=0.0797) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
2. `medquad_ans_0005127-1` (score=0.0744) - Pseudoachondroplasia is an inherited disorder of bone growth which is characterized by short stature. Other features include short arms and legs, a waddling walk, early-onset joint pain (osteoarthritis), and a limited...
3. `medquad_ans_0005140-1` (score=0.0554) - Pseudohypoparathyroidism type 1A is a type of pseudohypoparathyroidism. Pseudohypoparathyroidism is when your body is unable to respond to parathyroid hormone, which is a hormone that controls the levels of calcium, p...
4. `medquad_ans_0000284-1` (score=0.0525) - Diastrophic dysplasia is a disorder of cartilage and bone development. Affected individuals have short stature with very short arms and legs. Most also have early-onset joint pain (osteoarthritis) and joint deformitie...
5. `medquad_ans_0000948-1` (score=0.0490) - Spondyloperipheral dysplasia is a disorder that impairs bone growth. This condition is characterized by flattened bones of the spine (platyspondyly) and unusually short fingers and toes (brachydactyly), with the excep...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies (90%), elbow dislocation (90%), hypoplasia of the zygomatic bone (90%), scoliosis (90%), omphalocele (7.5%), hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, and tongue-like lumbar vertebral deformities [medquad_ans_0005135-1].

**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

### Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000251-2` (score=0.0650) - Cri-du-chat syndrome occurs in an estimated 1 in 20,000 to 50,000 newborns. This condition is found in people of all ethnic backgrounds.
2. `medquad_ans_0001664-1` (score=0.0640) - Cri du chat syndrome, also known as 5p- (5p minus) syndrome or cat cry syndrome, is a genetic condition that is caused by the deletion of genetic material on the small arm (the p arm) of chromosome 5. Infants with thi...
3. `medquad_ans_0001664-5` (score=0.0635) - How might cri du chat syndrome be treated? While there is no specific treatment available for cri du chat syndrome, early intervention is recommended in the areas of physical therapy (achieving physical and motor mile...
4. `medquad_ans_0001664-2` (score=0.0625) - What are the signs and symptoms of Cri du chat syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Cri du chat syndrome. If the information is available, the table below includ...
5. `medquad_ans_0000251-5` (score=0.0618) - These resources address the diagnosis or management of cri-du-chat syndrome: - Cri du Chat Syndrome Support Group (UK): Diagnosis - Cri du Chat Syndrome Support Group (UK): Therapies - Genetic Testing Registry: 5p par...

**Answer**:

Cri-du-chat syndrome, also known as 5p- (5p minus) syndrome or cat cry syndrome, is a genetic condition caused by the deletion of genetic material on the small arm (the p arm) of chromosome 5. It is characterized by a high-pitched cry that sounds like that of a cat, intellectual disability, delayed development, small head size (microcephaly), low birth weight, weak muscle tone in infancy (muscular hypotonia), and distinctive facial features such as epicanthus, low-set and posteriorly rotated ears, a wide nasal bridge, and a round face [medquad_ans_0001664-1]. The condition occurs in an estimated 1 in 20,000 to 50,000 newborns and affects individuals of all ethnic backgrounds. Most cases are not inherited [medquad_ans_0000251-2]med

**Supported citations**: `medquad_ans_0001664-1`, `medquad_ans_0000251-2`
**Unsupported citations**: _none_

### Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000718-1__dup1` (score=0.0814) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
2. `medquad_ans_0000719-2__dup1` (score=0.0797) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmen...
3. `medquad_ans_0000719-1__dup1` (score=0.0796) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides ...
4. `medquad_ans_0000807-5` (score=0.0763) - These resources address the diagnosis or management of polymicrogyria: - Gene Review: Gene Review: Polymicrogyria Overview - Genetic Testing Registry: Congenital bilateral perisylvian syndrome - Genetic Testing Regist...
5. `medquad_ans_0000720-1__dup1` (score=0.0763) - Bilateral generalized polymicrogyria is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). This is the most widespread form of polymicrogyria and typically affects the enti...

**Answer**:

The symptoms of Bilateral frontoparietal polymicrogyria may include: moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze (eyes that are not aligned), ataxia, strabismus, increased muscle tone, finger dysmetria (difficulty controlling speed, distance, and/or power of movements), ankle clonus, Babinski sign, broad-based gait, cerebellar hypoplasia, cerebral dysmyelination, esotropia, exotropia, hypoplasia of the brainstem, hyperreflexia, hypertonia, nystagmus, and a polymicrogyria pattern anterior to posterior gradient [medquad_ans_0000719-2__dup1]. The frequency of these symptoms is estimated based on data from the Human Phenotype Ontology and Orphanet, with some symptoms reported in a significant proportion of patients [medquad

**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

### Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Scores**: context_relevance=0.333, groundedness=0.013, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0005818-1` (score=0.0492) - Spondylocarpotarsal synostosis (SCT) syndrome is an inherited syndrome characterized by disproportionate short stature, abnormalities of the vertebrae in the spine, scoliosis and lordosis, carpal and tarsal fusion (fu...
2. `medquad_ans_0006375-1` (score=0.0484) - Vohwinkel syndrome is an inherited condition that affects the skin. People with the "classic form" generally have honeycomb-patterned calluses on the palms of the hands and the soles of the feet (palmoplantar keratose...
3. `medquad_ans_0001666-1` (score=0.0471) - Crigler Najjar syndrome, type 2 is caused by mutations in the UGT1A1 gene. The gene mutation causes the body to be unable to make adequate enzyme to convert bilirubin into a form that can easily be removed from the bo...
4. `medquad_ans_0000840-1__dup1` (score=0.0471) - Branchiootorenal syndrome is characterized by birth defects or anomalies of tissues in the neck, malformations of the external ear, hearing loss, and kidney malformations. Symptom and symptom severity can vary greatly...
5. `medquad_ans_0001143-1` (score=0.0457) - CHILD syndrome, also known as congenital hemidysplasia with ichthyosiform erythroderma and limb defects, is a genetic condition that is typically characterized by large patches of skin that are red and inflamed (eryth...

**Answer**:

The provided context does not mention "Konigsmark Knox Hussels syndrome" or any information related to its cause. Therefore, based on the given context, it is not possible to determine the cause of this syndrome.

Answer: Insufficient context to determine the cause of Konigsmark Knox Hussels syndrome. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.500, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004584-2` (score=0.0820) - What are the signs and symptoms of Orofaciodigital syndrome 1? Oral features of OFD1 may include a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormali...
2. `medquad_ans_0004590-1` (score=0.0806) - What are the signs and symptoms of Orofaciodigital syndrome 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 3. If the information is available, the table ...
3. `medquad_ans_0004593-1` (score=0.0794) - What are the signs and symptoms of Orofaciodigital syndrome 8? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 8. If the information is available, the table ...
4. `medquad_ans_0004589-2` (score=0.0776) - What are the signs and symptoms of Orofaciodigital syndrome 2? Although the signs and symptoms that occur in people with orofaciodigital syndrome type 2 may vary, the following findings may be present:Facial findings ...
5. `medquad_ans_0004586-1` (score=0.0774) - What are the signs and symptoms of Orofaciodigital syndrome 11? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 11. If the information is available, the tabl...

**Answer**:

The symptoms of Orofaciodigital syndrome 1 include oral features such as a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormalities. Facial features may include hypertelorism (increased width between the eyes), a small nose, micrognathia (small jaw), and other facial abnormalities. Finger and toe abnormalities may include shortness (brachydactyly), webbing or joining (syndactyly), abnormally curved fingers (clinodactyly), or other malformations. Brain abnormalities such as cysts and kidney problems like polycystic kidney disease may also occur. About half of individuals have some degree of learning disability, usually mild. Additionally, symptoms include bifid tongue (90%), broad alveolar ridges (90%), cleft upper lip (90%), frontal bossing (90%), hypert

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Scores**: context_relevance=0.667, groundedness=0.912, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000863-1__dup1` (score=0.0328) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For exa...
2. `medquad_ans_0000718-1__dup2` (score=0.0323) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts...
3. `medquad_ans_0000018-1__dup2` (score=0.0310) - After any operation, you'll have some side effects. There is usually some pain with surgery. There may also be swelling and soreness around the area that the surgeon cut. Your surgeon can tell you which side effects t...
4. `medquad_ans_0000969-1__dup2` (score=0.0308) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
5. `medquad_ans_0000010-23` (score=0.0308) - Several types of surgery are available for someone with colorectal cancer. If the cancer is found at a very early stage, the doctor may remove it without cutting through the abdominal wall. Instead, the doctor may put...

**Answer**:

Yes, there is information about surgery. Surgery can relieve or prevent pain, reduce symptoms, improve body function, or find a problem such as through a biopsy. Some surgeries, like heart surgery, can save a person's life. Laparoscopic surgery allows for smaller incisions using a thin tube with a camera and small tools. After surgery, there is usually some pain, swelling, and soreness, and possible complications such as infection, too much bleeding, reaction to anesthesia, or accidental injury. Surgeons can provide information about expected side effects, recovery, hospital stay, home care, return to work, and when it is safe to exercise. Weight loss surgery may be an option for people with extreme obesity who cannot lose weight through diet and exercise or have serious health problems caused by obesity [medquad_ans_0000863-1__dup1]medquad_ans_0000018-1\_\_dup2

**Supported citations**: `medquad_ans_0000863-1__dup1`
**Unsupported citations**: _none_

### Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Scores**: context_relevance=0.875, groundedness=0.098, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000040-11` (score=0.0559) - There are four common types of leukemia. They are chronic lymphocytic leukemia, chronic myeloid leukemia, acute myeloid leukemia, and acute lymphocytic leukemia. Chronic lymphocytic leukemia, chronic myeloid leukemia,...
2. `medquad_ans_0000040-1__dup3` (score=0.0545) - Leukemia is a cancer of the blood cells. It is the most common type of blood cancer and affects 10 times as many adults as children. Most people diagnosed with leukemia are over 50 years old. Leukemia Starts in Bone M...
3. `medquad_ans_0000554-1__dup2` (score=0.0513) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells c...
4. `medquad_ans_0002935-1` (score=0.0489) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affe...
5. `medquad_ans_0002935-4` (score=0.0487) - How is human T-cell leukemia virus, type 2 diagnosed? Human T-cell leukemia virus, type 2 (HTLV-2) is usually diagnosed based on blood tests that detect antibodies to the virus. However, HTLV-2 is often never suspecte...

**Answer**:

The provided context does not mention Human T-cell leukemia virus type 2 (HTLV-2). Therefore, it is not possible to answer the question based on the given information. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Scores**: context_relevance=0.833, groundedness=0.977, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000522-1__dup1` (score=0.0656) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...
2. `medquad_ans_0000038-1__dup2` (score=0.0645) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chroni...
3. `medquad_ans_0000038-5__dup1` (score=0.0628) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
4. `medquad_ans_0000038-6` (score=0.0613) - Childhood interstitial lung disease (chILD) has many signs and symptoms because the disease has many forms. Signs and symptoms may include: Fast breathing, which also is called tachypnea (tak-ip-NE-ah) Labored breathi...
5. `medquad_ans_0000038-7` (score=0.0605) - Doctors diagnose childhood interstitial lung disease (chILD) based on a child's medical and family histories and the results from tests and procedures. To diagnose chILD, doctors may first need to rule out other disea...

**Answer**:

Interstitial lung diseases (ILD) are a large group of diseases that inflame or scar the lungs, making it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Causes include breathing in dust or other particles, such as coal dust (black lung disease), farm dust (farmer's lung), asbestos fibers (asbestosis), iron from mines or welding fumes (siderosis), and silica dust (silicosis). Other causes include autoimmune diseases or occupational exposures to molds, gases, or fumes. Some types have no known cause. Treatment depends on the type of exposure and disease stage and may involve medicines, oxygen therapy, or lung transplant in severe cases [medquad_ans_0000522-1__dup1]. In children, childhood interstitial lung disease (chILD) is a broad term for rare lung diseases affecting babies, children, and teens, with similar symptoms like chronic cough, rapid

**Supported citations**: `medquad_ans_0000522-1__dup1`
**Unsupported citations**: _none_

### Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.020, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0003743-4` (score=0.0328) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition w...
2. `medquad_ans_0002735-4` (score=0.0318) - Is Hashimoto's encephalitis inherited? We are aware of only one instance when more than one person in the same family had Hashimoto's encephalitis (HE). To our knowledge, no other cases of familial HE have been report...
3. `medquad_ans_0005004-4` (score=0.0313) - Is postural orthostatic tachycardia syndrome inherited? Most cases of postural orthostatic tachycardia syndrome (POTS) are not thought to be inherited. Although the condition generally occurs sporadically, some people...
4. `medquad_ans_0000261-4` (score=0.0310) - In most cases of cyclic vomiting syndrome, affected people have no known history of the disorder in their family. However, many affected individuals have a family history of related conditions, such as migraines, irri...
5. `medquad_ans_0000139-2__dup2` (score=0.0308) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...

**Answer**:

The provided context does not contain any information about Melnick-Needles syndrome or its inheritance pattern. Therefore, it is not possible to determine based on the given text whether Melnick-Needles syndrome is inherited.

Answer: Insufficient context [doc_id].

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Scores**: context_relevance=0.750, groundedness=0.910, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0006138-5` (score=0.0650) - How is tracheobronchomalacia diagnosed? A diagnosis of tracheobronchomalacia (TBM) may be suspected based on the presence of characteristic signs and symptoms or abnormal pulmonary function tests. Additional testing s...
2. `medquad_ans_0006140-1` (score=0.0643) - Tracheobronchopathia osteoplastica (TO) is a rare condition of the large airways. It is characterized by the presence of multiple growths (nodules) made of bone and cartilage tissue, in the submucosa of the tracheobro...
3. `medquad_ans_0006138-6` (score=0.0626) - How might tracheobronchomalacia be treated? Treatment is only medically necessary in people who have signs and symptoms of tracheobronchomalacia (TBM). Management of symptomatic TBM first involves identifying underlyi...
4. `medquad_ans_0006140-6` (score=0.0613) - How might tracheobronchopathia osteoplastica be treated? There is no specific treatment for tracheobronchopathia osteoplastica (TO). Recurrent infections and collapse of the lung are treated conventionally. Inhaled co...
5. `medquad_ans_0006138-1` (score=0.0611) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of ...

**Answer**:

Tracheobronchomalacia (TBM) is a condition characterized by the weakening of the tracheal and bronchial walls, leading to narrowing of the airways. A diagnosis of TBM may be suspected based on characteristic signs and symptoms or abnormal pulmonary function tests. Confirmation and evaluation of severity can be achieved through additional testing such as CT scan and bronchoscopy. TBM is classified as mild if the trachea narrows to 50% of its initial size during exhalation, moderate if it narrows to 25%, and severe if the walls of the trachea touch [medquad_ans_0006138-5]. Treatment is only necessary in individuals with symptoms and involves addressing underlying conditions such as chronic inflammation, compression, or injury; treatment options may include silicone or long-term stenting, surgical correction, continuous positive airway pressure (CPAP), or tracheostomy (

**Supported citations**: `medquad_ans_0006138-5`
**Unsupported citations**: _none_

### Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Scores**: context_relevance=1.000, groundedness=0.993, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006030-1` (score=0.0656) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people m...
2. `medquad_ans_0000880-1__dup2` (score=0.0643) - The temporomandibular joint (TMJ) connects your jaw to the side of your head. When it works well, it enables you to talk, chew, and yawn. For people with TMJ dysfunction, problems with the joint and muscles around it ...
3. `medquad_ans_0000762-1` (score=0.0613) - Otopalatodigital syndrome type 2 is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, whi...
4. `medquad_ans_0003474-2` (score=0.0583) - What are the signs and symptoms of Larsen syndrome? The signs and symptoms of Larsen syndrome vary from person to person, but may include the following: Joint dislocation (especially of the hips, knees, and elbows) Hy...
5. `medquad_ans_0000389-1` (score=0.0578) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which al...

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene, and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

### Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.702, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000139-2__dup2` (score=0.0197) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...
2. `medquad_ans_0000542-1__dup2` (score=0.0193) - Klinefelter syndrome (KS) is a condition that occurs in men who have an extra X chromosome. The syndrome can affect different stages of physical, language, and social development. The most common symptom is infertilit...
3. `medquad_ans_0003845-4` (score=0.0164) - Is McCune Albright syndrome inherited? McCune Albright syndrome (MAS) is not inherited. It is caused by a random change (mutation) in the GNAS gene that occurs very early in development. As a result, some of the body'...
4. `medquad_ans_0000554-2` (score=0.0164) - Kallmann syndrome is estimated to affect 1 in 10,000 to 86,000 people and occurs more often in males than in females. Kallmann syndrome 1 is the most common form of the disorder.
5. `medquad_ans_0000631-4` (score=0.0161) - McCune-Albright syndrome is not inherited. Instead, it is caused by a random mutation in the GNAS gene that occurs very early in development. As a result, some of the body's cells have a normal version of the GNAS gen...

**Answer**:

The provided context does not contain information about whether Kallmann syndrome is inherited. Therefore, based on the given text, it is not possible to determine if Kallmann syndrome is inherited.

[medquad_ans_0000554-2]

**Supported citations**: `medquad_ans_0000554-2`
**Unsupported citations**: _none_

### Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Scores**: context_relevance=1.000, groundedness=0.915, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005967-2` (score=0.0814) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how...
2. `medquad_ans_0005967-1` (score=0.0807) - Syndactyly type 3 (SD3) is a limb abnormality present at birth that is characterized by complete fusion of the 4th and 5th fingers on both hands. In most cases only the soft tissue is fused, but in some cases the bone...
3. `medquad_ans_0005970-1` (score=0.0796) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how...
4. `medquad_ans_0000908-1__dup1` (score=0.0763) - What are the signs and symptoms of Camptodactyly syndrome Guadalajara type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Camptodactyly syndrome Guadalajara type 3. If the inform...
5. `medquad_ans_0005022-1` (score=0.0763) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the tabl...

**Answer**:

The symptoms of Syndactyly type 3 include complete fusion of the 4th and 5th fingers on both hands, which may involve only soft tissue or, in some cases, the bones of the fingers (distal phalanges). Other associated features include camptodactyly of the finger in 50% of patients, short toe in 7.5% of patients, absence of the middle phalanx of the 5th finger, and a short 5th finger. This condition is inherited in an autosomal dominant manner and is caused by mutations in the GJA1 gene [medquad_ans_0005967-1][medquad_ans_0005967-2].

**Supported citations**: `medquad_ans_0005967-1`, `medquad_ans_0005967-2`
**Unsupported citations**: _none_

### Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Scores**: context_relevance=0.800, groundedness=0.471, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000197-1__dup3` (score=0.0656) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...
2. `medquad_ans_0000133-1__dup2` (score=0.0635) - The bronchi are two tubes that branch off the trachea, or windpipe. The bronchi carry air to your lungs. The most common problem with the bronchi is bronchitis, an inflammation of the tubes. Bronchitis can be acute or...
3. `medquad_ans_0000853-1__dup1` (score=0.0623) - Bronchiolitis obliterans is an inflammatory obstruction of the lung's tiniest airways, the bronchioles. The bronchioles may become damaged and inflamed after inhalation of toxic fumes, as a result of respiratory infec...
4. `medquad_ans_0000008-1__dup3` (score=0.0614) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...
5. `medquad_ans_0000021-1__dup3` (score=0.0609) - Espaol Bronchitis (bron-KI-tis) is a condition in which the bronchial tubes become inflamed. These tubes carry air to your lungs. (For more information about the bronchial tubes and airways, go to the Diseases and Con...

**Answer**:

Chronic bronchitis is an inflammation of the bronchial tubes, the airways that carry air to the lungs. It causes a cough that often brings up mucus and can also cause shortness of breath, wheezing, a low fever, and chest tightness. It is one type of COPD (chronic obstructive pulmonary disease) and is characterized by the production of a large amount of mucus by the inflamed bronchial tubes, leading to persistent coughing and difficulty breathing. Cigarette smoking is the most common cause, along with long-term exposure to air pollution, fumes, or dust. It is a long-term condition that tends to recur or never goes away completely. Diagnosis involves evaluating symptoms, listening to breathing, and possibly other tests. Treatment includes medicines to open airways and clear mucus, oxygen therapy, and pulmonary rehabilitation to help manage symptoms in daily life [medquad_ans_0000

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Scores**: context_relevance=0.875, groundedness=0.980, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000935-2` (score=0.0817) - Spinal muscular atrophy affects 1 in 6,000 to 1 in 10,000 people.
2. `medquad_ans_0000840-1__dup2` (score=0.0804) - Spinal muscular atrophy (SMA) is a genetic disease that attacks nerve cells, called motor neurons, in the spinal cord. These cells communicate with your voluntary muscles - the ones you can control, like in your arms ...
3. `medquad_ans_0000934-5` (score=0.0794) - These resources address the diagnosis or management of spinal and bulbar muscular atrophy: - Gene Review: Gene Review: Spinal and Bulbar Muscular Atrophy - Genetic Testing Registry: Bulbo-spinal atrophy X-linked - Med...
4. `medquad_ans_0000935-1` (score=0.0770) - Spinal muscular atrophy is a genetic disorder that affects the control of muscle movement. It is caused by a loss of specialized nerve cells, called motor neurons, in the spinal cord and the part of the brain that is ...
5. `medquad_ans_0000936-5` (score=0.0758) - These resources address the diagnosis or management of spinal muscular atrophy with progressive myoclonic epilepsy: - Genetic Testing Registry: Jankovic Rivera syndrome - Muscular Dystrophy Association: Spinal Muscula...

**Answer**:

Treatments for Spinal Muscular Atrophy include machines to help with breathing, nutritional support, physical therapy, and medicines. There is no cure, but these treatments help with symptoms and prevent complications [medquad_ans_0000840-1__dup2].

**Supported citations**: `medquad_ans_0000840-1__dup2`
**Unsupported citations**: _none_

### Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Scores**: context_relevance=0.778, groundedness=0.775, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000040-19` (score=0.0786) - Standard treatments for leukemia include chemotherapy, biological therapy, radiation therapy, and immunotherapy. Some patients receive a combination of treatments. Learn more about treatments for acute myeloid leukemi...
2. `medquad_ans_0000009-1__dup3` (score=0.0766) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. Thes...
3. `medquad_ans_0000040-4__dup2` (score=0.0761) - There are many treatment options for people with leukemia. The choice of treatment depends on your age and general health, the type of leukemia you have, whether or not it has spread outside the bone marrow, and other...
4. `medquad_ans_0000040-5__dup1` (score=0.0714) - Unlike other types of cancer, leukemia isn't a tumor that your doctor can surgically remove. Leukemia cells are produced in the bone marrow and travel throughout the body. The Goal of Treatment The goal of treatment f...
5. `medquad_ans_0000040-18` (score=0.0710) - Treatment depends on a number of factors, including the type of leukemia, the patient's age and general health, where leukemia cells have collected in the body, and whether the leukemia has been treated before. Certai...

**Answer**:

The provided context does not specify treatments for Adult Acute Lymphoblastic Leukemia (ALL). While it mentions treatments for ALL in general, such as chemotherapy, radiation therapy, stem cell transplants, and targeted therapy, it does not differentiate or detail treatments specifically for adult patients with ALL. Therefore, based on the available information, the treatments for Adult Acute Lymphoblastic Leukemia cannot be accurately determined. [medquad_ans_0000009-1__dup3]

**Supported citations**: `medquad_ans_0000009-1__dup3`
**Unsupported citations**: _none_

### Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Scores**: context_relevance=0.750, groundedness=0.026, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000388-2` (score=0.0820) - Friedreich ataxia is estimated to affect 1 in 40,000 people. This condition is found in people with European, Middle Eastern, or North African ancestry. It is rarely identified in other ethnic groups.
2. `medquad_ans_0000382-1__dup2` (score=0.0799) - Friedreich's ataxia is an inherited disease that damages your nervous system. The damage affects your spinal cord and the nerves that control muscle movement in your arms and legs. Symptoms usually begin between the a...
3. `medquad_ans_0000388-1` (score=0.0789) - Friedreich ataxia is a genetic condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features of...
4. `medquad_ans_0000388-5` (score=0.0784) - These resources address the diagnosis or management of Friedreich ataxia: - Friedreich's Ataxia Research Alliance: Clinical Care Guidelines - Gene Review: Gene Review: Friedreich Ataxia - Genetic Testing Registry: Fri...
5. `medquad_ans_0000133-1__dup1` (score=0.0779) - Friedreich's ataxia is a rare inherited disease that causes progressive damage to the nervous system and movement problems. Neurological symptoms include awkward, unsteady movements, impaired sensory function, speech ...

**Answer**:

The provided context does not specify the cause of Friedreich ataxia. While it describes symptoms, inheritance, and associated conditions, it does not explain the underlying genetic or biological cause. Therefore, based on the given information, the cause cannot be determined. [doc_id] is not available.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Scores**: context_relevance=0.833, groundedness=0.785, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000013-7` (score=0.0615) - Atrial fibrillation (AF) is diagnosed based on your medical and family histories, a physical exam, and the results from tests and procedures. Sometimes AF doesn't cause signs or symptoms. Thus, it may be found during ...
2. `medquad_ans_0000013-4__dup2` (score=0.0608) - Atrial fibrillation (AF) occurs if the heart's electrical signals don't travel through the heart in a normal way. Instead, they become very rapid and disorganized. Damage to the heart's electrical system causes AF. Th...
3. `medquad_ans_0000013-8` (score=0.0579) - Treatment for atrial fibrillation (AF) depends on how often you have symptoms, how severe they are, and whether you already have heart disease. General treatment options include medicines, medical procedures, and life...
4. `medquad_ans_0000013-5__dup1` (score=0.0576) - Atrial fibrillation (AF) affects millions of people, and the number is rising. Men are more likely than women to have the condition. In the United States, AF is more common among Whites than African Americans or Hispa...
5. `medquad_ans_0000070-1__dup4` (score=0.0576) - An arrhythmia is a problem with the speed or rhythm of the heartbeat. Atrial fibrillation (AF) is the most common type of arrhythmia. The cause is a disorder in the heart's electrical system. Often, people who have AF...

**Answer**:

Familial atrial fibrillation is not defined or discussed in the provided context. Therefore, based on the given information, it cannot be determined what familial atrial fibrillation is. [medquad_ans_0000013-7]

**Supported citations**: `medquad_ans_0000013-7`
**Unsupported citations**: _none_

### Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.0656) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the assoc...
2. `medquad_ans_0004627-2` (score=0.0645) - What are the signs and symptoms of Osteopathia striata cranial sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopathia striata cranial sclerosis. If the information is...
3. `medquad_ans_0004625-1` (score=0.0608) - Osteomesopyknosis is a bone disorder characterized by abnormal hardening of bone (osteosclerosis). It is generally limited to the axial spine, pelvis, and proximal part of the long bones, which is what distinguishes t...
4. `medquad_ans_0000922-2` (score=0.0594) - SOST-related sclerosing bone dysplasia is a rare condition; its exact prevalence is unknown. Approximately 100 individuals with sclerosteosis have been reported in the scientific literature. Sclerosteosis is most comm...
5. `medquad_ans_0000238-1__dup1` (score=0.0591) - Primary lateral sclerosis (PLS) is a rare neuromuscular disease with slowly progressive weakness in voluntary muscle movement. PLS belongs to a group of disorders known as motor neuron diseases. PLS affects the upper ...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary significantly from person to person, even within a single family. Features of the condition are generally present at birth and may include skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected people may also have developmental delay, hearing loss, heart defects, and/or ophthalmoplegia (paralysis of the muscles surrounding the eyes). Osteopathia striata cranial sclerosis is caused by changes (mutations) in the WTX gene and is inherited in an X-linked dominant manner. Treatment is based on the signs and symptoms present in each person [medquad_ans_0004

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0004097-5` (score=0.0820) - How is mixed connective tissue disease diagnosed? Mixed connective tissue disease (MCTD) is often suspected after a physical examination reveals signs and symptoms associated with the condition. The diagnosis is suppo...
2. `medquad_ans_0004097-1` (score=0.0794) - Mixed connective tissue disease (MCTD) is a rare autoimmune disorder that is characterized by features commonly seen in three different connective tissue disorders: systemic lupus erythematosus, scleroderma, and polym...
3. `medquad_ans_0004097-2` (score=0.0792) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus eryt...
4. `medquad_ans_0000230-1__dup2` (score=0.0791) - Connective tissue is the material inside your body that supports many of its parts. It is the "cellular glue" that gives your tissues their shape and helps keep them strong. It also helps some of your tissues do their...
5. `medquad_ans_0004097-6` (score=0.0769) - How might mixed connective tissue disease be treated? There is currently no cure for mixed connective tissue disease (MCTD). However, treatments can help manage symptoms of the condition. For example, medications such...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. A condition called Raynaud's phenomenon sometimes occurs months or years before other symptoms of MCTD develop. Most people with MCTD have pain in multiple joints, and/or inflammation of joints (arthritis). Muscle weakness, fevers, and fatigue are also common. Other signs and symptoms may include: accumulation of fluid in the tissue of the hands that causes puffiness and swelling (edema); skin findings including lupus-like rashes (including reddish brown patches), reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia); dysfunction of the esophagus

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Scores**: context_relevance=1.000, groundedness=0.841, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000003-6__dup1` (score=0.0817) - Most people with Alzheimers disease are cared for at home by family members. Within families, caregiving is provided most often by wives and husbands, followed by daughters. As Alzheimers disease gets worse, the perso...
2. `medquad_ans_0000024-1__dup2` (score=0.0799) - Caring for someone who has Alzheimer's disease (AD) can be stressful and overwhelming. It's important to take care of yourself. Ask for and accept help. Talk to the doctor. Find out what treatments might help control ...
3. `medquad_ans_0000003-30` (score=0.0770) - As Alzheimers disease gets worse, you will need more help to care for the person. It's okay to seek help whenever you need it. Several kinds of help are available. - Home health care agencies send a home health aide o...
4. `medquad_ans_0000004-5__dup2` (score=0.0753) - Medications Can Treat Symptoms There is no known cure for Alzheimer's disease, but there are medicines that can treat symptoms of the disease. Most Alzheimers medicines work best for people in the mild or moderate sta...
5. `medquad_ans_0000098-2__dup1` (score=0.0734) - Drugs to specifically treat Alzheimers disease and some other progressive dementias are now available. Although these drugs do not halt the disease or reverse existing brain damage, they can improve symptoms and slow ...

**Answer**:

The provided context does not mention specific treatments for Alzheimer's caregiving. It discusses support systems, services such as home health care, and resources for finding help, but does not outline treatments for caregiving itself. Therefore, based on the given information, no treatments for Alzheimer's caregiving can be identified.

[medquad_ans_0000003-6__dup1]

**Supported citations**: `medquad_ans_0000003-6__dup1`
**Unsupported citations**: _none_

### Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.740

**Top docs**:

1. `medquad_ans_0000293-4` (score=0.0656) - Distal myopathy 2 is inherited in an autosomal dominant pattern, which means one copy of the altered MATR3 gene in each cell is sufficient to cause the disorder.
2. `medquad_ans_0003179-1` (score=0.0643) - Inclusion body myopathy 2, also known as hereditary inclusion body myopathy (HIBM), GNE-related myopathy, distal myopathy with rimmed vacuoles, and Nonaka myopathy, is an inherited condition that primarily affects the...
3. `medquad_ans_0003179-4` (score=0.0633) - How is inclusion body myopathy 2 inherited? Inclusion body myopathy 2 is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with ...
4. `medquad_ans_0006236-4` (score=0.0625) - Is tubular aggregate myopathy genetic? It is evident from family history studies that the condition can be passed through families in either an autosomal dominant or autosomal recessive fashion. Some cases appear to b...
5. `medquad_ans_0006500-1` (score=0.0618) - X-linked scapuloperoneal myopathy is an inherited muscular dystrophy characterized by weakness and wasting of the muscles in the lower legs and the area of the shoulder blades. In some individuals, facial muscles may ...

**Answer**:

The provided context does not contain any information about the genetic changes related to idiopathic inflammatory myopathy. Therefore, a definitive answer cannot be provided based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.845, answer_relevance=0.567

**Top docs**:

1. `medquad_ans_0000855-2` (score=0.0492) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
2. `medquad_ans_0000855-5` (score=0.0476) - These resources address the diagnosis or management of pyruvate dehydrogenase deficiency: - Genetic Testing Registry: Pyruvate dehydrogenase E1-beta deficiency - Genetic Testing Registry: Pyruvate dehydrogenase E2 def...
3. `medquad_ans_0000855-1` (score=0.0476) - Pyruvate dehydrogenase deficiency is characterized by the buildup of a chemical called lactic acid in the body and a variety of neurological problems. Signs and symptoms of this condition usually first appear shortly ...
4. `medquad_ans_0005223-2` (score=0.0471) - What are the signs and symptoms of Pyruvate dehydrogenase deficiency? Pyruvate dehydrogenase (PDH) deficiency can have a significant effect on fetal development, which may become apparent during late pregnancy with po...
5. `medquad_ans_0005223-5` (score=0.0466) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for...

**Answer**:

Diagnosis of pyruvate dehydrogenase deficiency can be approached through genetic testing, which identifies specific deficiencies in components of the pyruvate dehydrogenase complex, such as E1-beta, E2, E3-binding protein, or phosphatase deficiencies [medquad_ans_0000855-5]. Additionally, diagnostic resources such as those provided by MedlinePlus, including Diagnostic Tests, Genetic Counseling, and Drug Therapy, may support the evaluation and management of the condition [medquad_ans_0000855-5]. Signs and symptoms, including lactic acidosis, neurological issues, and developmental delays, may also prompt further diagnostic evaluation [medquad_ans_0000855-1]. However, specific diagnostic procedures or tests are not detailed in the provided context.

**Supported citations**: `medquad_ans_0000855-5`, `medquad_ans_0000855-1`
**Unsupported citations**: _none_

### Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.938, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006562-2` (score=0.0795) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb a...
2. `medquad_ans_0006562-3` (score=0.0774) - What causes 17q23.2q23.2 microdeletion syndrome? The syndrome is caused by an interstitial deletion (a deletion that does not involve the ends of a chromosome) encompassing bands 23.1 to 23.2 on the long (q) arm of ch...
3. `medquad_ans_0006554-1` (score=0.0768) - 15q11.2 microdeletion refers to a chromosome abnormality in which a tiny piece of genetic material on the long arm of chromosome 15 (at a location designated q11.2) is missing (deleted). The features of people with a ...
4. `medquad_ans_0006562-5` (score=0.0767) - How is 17q23.1q23.2 microdeletion syndrome diagnosed? The deletion can be identified by comparative genomic hybridization (CGH) microarray and fluorescence in situ hybridization (FISH).
5. `medquad_ans_0003415-3` (score=0.0764) - How is 17q21.31 microdeletion syndrome diagnosed? 17q21.31 microdeletion syndrome is diagnosed in individuals who have a deletion of 500,000 to 650,000 DNA building blocks (base pairs) at chromosome 17q21.31. The diag...

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

**Scores**: context_relevance=1.000, groundedness=0.923, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003860-1` (score=0.0820) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...
2. `medquad_ans_0003860-3` (score=0.0794) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
3. `medquad_ans_0000635-3` (score=0.0791) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
4. `medquad_ans_0000635-2` (score=0.0786) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of...
5. `medquad_ans_0000635-1` (score=0.0779) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infan...

**Answer**:

MECP2 duplication syndrome is caused by an extra copy (duplication) of the MECP2 gene in each cell, which is generally due to a duplication of genetic material located on the long (q) arm of the X chromosome [medquad_ans_0003860-1]. The duplication can vary in size, ranging from 100,000 to 900,000 base pairs (100 to 900 kilobases), and the MECP2 gene is always included in this duplication [medquad_ans_0000635-3]. This results in the production of excess MeCP2 protein, which disrupts the normal regulation of other genes in the brain, leading to irregular brain activity and the associated signs and symptoms [medquad_ans_0003860-3].

**Supported citations**: `medquad_ans_0003860-1`, `medquad_ans_0000635-3`, `medquad_ans_0003860-3`
**Unsupported citations**: _none_

### Example 187 (`0004633-2`)

**Question**: What are the symptoms of Osteopetrosis autosomal dominant type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004633-2` (score=0.0492) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 2. If the informatio...
2. `medquad_ans_0004632-2` (score=0.0474) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 1. If the informatio...
3. `medquad_ans_0000759-2` (score=0.0474) - Autosomal dominant osteopetrosis is the most common form of the disorder, affecting about 1 in 20,000 people. Autosomal recessive osteopetrosis is rarer, occurring in an estimated 1 in 250,000 people. Other forms of o...
4. `medquad_ans_0000759-1` (score=0.0469) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of ...
5. `medquad_ans_0004637-2` (score=0.0464) - What are the signs and symptoms of Osteopetrosis autosomal recessive 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 4. If the information is ava...

**Answer**:

The symptoms of Osteopetrosis autosomal dominant type 2 include abnormality of epiphysis morphology, abnormality of the metacarpal bones, abnormality of the metaphyses, aseptic necrosis, bone pain, facial palsy, frontal bossing, joint dislocation, macrocephaly, osteoarthritis, osteomyelitis, recurrent fractures, short distal phalanx of finger, anemia, genu valgum, optic atrophy, short stature, visual impairment, abnormality of leukocytes, carious teeth, hearing impairment, hydrocephalus, hypocalcemia, bone marrow hypocellularity, abnormality of pelvic girdle bone morphology, abnormality of the vertebral endplates, generalized osteosclerosis, hip osteoarthritis, juvenile onset, mandibular osteomyelitis, and fractures of the long bones [medquad_ans_0004633

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 188 (`0003459-4`)

**Question**: What are the treatments for Lambert Eaton myasthenic syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.542, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0003459-3` (score=0.0809) - What causes Lambert Eaton myasthenic syndrome? Lambert Eaton myasthenic syndrome is the result of an autoimmune process which causes a disruption of electrical impulses between nerve cells and muscle fibers. In cases ...
2. `medquad_ans_0003459-4` (score=0.0804) - How might Lambert-Eaton myasthenic syndrome be treated? Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, ...
3. `medquad_ans_0003459-1` (score=0.0794) - Lambert Eaton myasthenic syndrome (LEMS) is a disorder of the neuromuscular junction. The neuromuscular junction is the site where nerve cells meet muscle cells and help activate the muscles. This syndrome occurs when...
4. `medquad_ans_0003459-2` (score=0.0791) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficu...
5. `medquad_ans_0000175-1__dup1` (score=0.0742) - Lambert-Eaton myasthenic syndrome (LEMS) is a disorder of the neuromuscular junction-the site where nerve cells meet muscle cells and help activate the muscles. It is caused by a disruption of electrical impulses betw...

**Answer**:

Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, plasmapheresis (where blood plasma is removed and replaced with fluid, protein, or donated plasma) or IV immunoglobulins, steroids (e.g., prednisone), azathioprine or cyclosporine, and/or 3,4-diaminopyridine. 3,4-diaminopyridine is available in Europe and may be available in the U.S. on a compassionate use basis. While there has been some evidence that either 3,4-diaminopyridine or IV immunoglobulin can improve muscle strength and nerve to muscle cell communication, the degree of benefit (i.e., how much symptoms are improved) still needs to be determined [medquad_ans_00

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.956, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000445-1__dup1` (score=0.0814) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...
2. `medquad_ans_0000065-2` (score=0.0809) - Arginase deficiency is a very rare disorder; it has been estimated to occur once in every 300,000 to 1,000,000 individuals.
3. `medquad_ans_0000445-2__dup1` (score=0.0789) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes...
4. `medquad_ans_0000065-1` (score=0.0789) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down ...
5. `medquad_ans_0003472-2` (score=0.0745) - What are the signs and symptoms of L-arginine:glycine amidinotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for L-arginine:glycine amidinotransferase deficiency....

**Answer**:

The symptoms of Arginase deficiency include growth deficiency, spasticity (abnormal tensing of the muscles), developmental delay, loss of developmental milestones, intellectual disability, seizures, and microcephaly [medquad_ans_0000445-1__dup1]. Additional signs and symptoms reported in the Human Phenotype Ontology include aminoaciduria, behavioral abnormality, cognitive impairment, neurological speech impairment, EEG abnormality, hemiplegia/hemiparesis, hyperammonemia, seizures, anorexia, hyperactivity, irritability, oroticaciduria, postnatal growth retardation, progressive spastic quadriplegia, and vomiting [medquad_ans_0000445-2__dup1].

**Supported citations**: `medquad_ans_0000445-1__dup1`, `medquad_ans_0000445-2__dup1`
**Unsupported citations**: _none_

### Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Scores**: context_relevance=0.889, groundedness=0.908, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000961-1__dup1` (score=0.0809) - Von Hippel-Lindau disease (VHL) is a rare, genetic disease that causes tumors and cysts to grow in your body. The tumors can be either cancerous or benign. They can grow in your brain and spinal cord, kidneys, pancrea...
2. `medquad_ans_0006376-6` (score=0.0805) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but ar...
3. `medquad_ans_0006376-1` (score=0.0786) - Von Hippel-Lindau (VHL) disease is an inherited disorder characterized by the abnormal growth of both benign and cancerous tumors and cysts in many parts of the body. Tumors usually first appear in young adulthood. Th...
4. `medquad_ans_0006376-3` (score=0.0774) - What causes Von Hippel-Lindau disease? Von Hippel-Lindau (VHL) disease is caused by a mutation in the VHL gene. This gene is a tumor suppressor gene, which helps to control cell growth. Mutations in the VHL gene lead ...
5. `medquad_ans_0001020-2` (score=0.0768) - The incidence of von Hippel-Lindau syndrome is estimated to be 1 in 36,000 individuals.

**Answer**:

Treatments for Von Hippel-Lindau Disease (VHL) usually involve surgical removal of tumors, and in some cases, radiation therapy. The goal is to treat growths when they are small and cause symptoms, but before they lead to permanent damage. Early detection and treatment are important to prevent permanent damage [medquad_ans_0006376-6]. Surgical removal of tumors is a common approach, and radiation therapy may be used in specific cases [medquad_ans_0006376-1].

**Supported citations**: `medquad_ans_0006376-6`, `medquad_ans_0006376-1`
**Unsupported citations**: _none_

### Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Scores**: context_relevance=0.800, groundedness=0.052, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000738-2` (score=0.0492) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
2. `medquad_ans_0000738-1` (score=0.0481) - Noonan syndrome is a condition that affects many areas of the body. It is characterized by mildly unusual facial characteristics, short stature, heart defects, bleeding problems, skeletal malformations, and many other...
3. `medquad_ans_0004454-3` (score=0.0455) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004451-3` (score=0.0453) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004456-3` (score=0.0453) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not mention "Noonan syndrome 3" or any such classification. Therefore, based on the available information, it is not possible to answer the question. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005865-2` (score=0.0820) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may...
2. `medquad_ans_0005865-1` (score=0.0801) - Sprengel deformity is a congenital condition characterized by abnormal development and elevation of the shoulder blade (scapula). Severity can range considerably from being almost invisible when covered with clothes, ...
3. `medquad_ans_0005865-3` (score=0.0799) - How might Sprengel deformity be treated? Treatment of Sprengel deformity depends on the degree of functional impairment and/or cosmetic disfigurement. Many people with Sprengel deformity do not need surgery and may ha...
4. `medquad_ans_0003631-1` (score=0.0760) - Lipodermatosclerosis refers to changes in the skin of the lower legs. It is a form of panniculitis (inflammation of the layer of fat under the skin). Signs and symptoms include pain, hardening of skin, change in skin ...
5. `medquad_ans_0000799-1__dup1` (score=0.0686) - Scleroderma means hard skin. It is a group of diseases that cause abnormal growth of connective tissue. Connective tissue is the material inside your body that gives your tissues their shape and helps keep them strong...

**Answer**:

The symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additional features that may be present include abnormality of the shoulder (seen in approximately 90% of patients), cleft palate (in about 7.5% of patients), scoliosis, hemivertebrae, neck muscle hypoplasia, rib segmentation abnormalities, cervical segmentation defects, spina bifida occulta, and abnormalities of the collarbone or chest organs [medquad_ans_0005865-2]. The deformity more commonly occurs on the left side but can be bilateral [medquad_ans_0005865-2].

**Supported citations**: `medquad_ans_0005865-2`
**Unsupported citations**: _none_

### Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.921, answer_relevance=0.756

**Top docs**:

1. `medquad_ans_0000882-4` (score=0.0656) - As discussed above, Robinow syndrome can have either an autosomal recessive or an autosomal dominant pattern of inheritance. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. T...
2. `medquad_ans_0000882-2` (score=0.0643) - Both the autosomal recessive and autosomal dominant forms of Robinow syndrome are rare. Fewer than 200 people with autosomal recessive Robinow syndrome have been described in the medical literature. This form of the c...
3. `medquad_ans_0005415-1` (score=0.0630) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
4. `medquad_ans_0000882-5` (score=0.0623) - These resources address the diagnosis or management of Robinow syndrome: - Gene Review: Gene Review: Autosomal Dominant Robinow Syndrome - Gene Review: Gene Review: ROR2-Related Robinow Syndrome - Genetic Testing Regi...
5. `medquad_ans_0005415-4` (score=0.0618) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...

**Answer**:

Genetic changes related to Robinow syndrome include mutations in the ROR2 gene, which cause autosomal recessive Robinow syndrome, and mutations in the WNT5A or DVL1 gene, which may cause autosomal dominant Robinow syndrome. In some cases, the underlying cause is unknown [medquad_ans_0005415-1].

**Supported citations**: `medquad_ans_0005415-1`
**Unsupported citations**: _none_

### Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Scores**: context_relevance=0.909, groundedness=0.773, answer_relevance=0.836

**Top docs**:

1. `medquad_ans_0000180-1__dup3` (score=0.0648) - Kidney failure means your kidneys no longer work well enough to do their job. You need treatment to replace the work your damaged kidneys have stopped doing. The treatments for kidney failure are - hemodialysis - peri...
2. `medquad_ans_0000158-13` (score=0.0616) - For people who are on dialysis or approaching total kidney failure, adequate nutrition is important for maintaining energy, strength, healthy sleep patterns, bone health, heart health, and good mental health. A person...
3. `medquad_ans_0000179-13` (score=0.0616) - - Kidney failure means your kidneys no longer filter your blood well enough to keep you healthy. - The treatments for kidney failure are - hemodialysis - peritoneal dialysis - a kidney transplant - conservative manage...
4. `medquad_ans_0000179-12` (score=0.0615) - Eating healthy foods can help you keep up your energy and strength. All dialysis and transplant centers have a dietitian. The dietitian helps people with kidney failure learn about healthy food choices. You should tal...
5. `medquad_ans_0000179-4__dup2` (score=0.0609) - The treatments for kidney failure are - hemodialysis - peritoneal dialysis - a kidney transplant - conservative management

**Answer**:

The provided context does not contain information about "Kidney Failure: Eat Right to Feel Right on Hemodialysis." Therefore, it is not possible to answer the question based on the given text. [medquad_ans_0000180-1__dup3] discusses general dietary recommendations for kidney failure patients, such as following a special diet to prevent waste buildup in the blood, but it does not specifically address the topic "Eat Right to Feel Right on Hemodialysis."

**Supported citations**: `medquad_ans_0000180-1__dup3`
**Unsupported citations**: _none_

### Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Scores**: context_relevance=0.778, groundedness=0.945, answer_relevance=0.511

**Top docs**:

1. `medquad_ans_0000677-2` (score=0.0492) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
2. `medquad_ans_0000677-1` (score=0.0479) - Mucolipidosis type IV is an inherited disorder characterized by delayed development and vision impairment that worsens over time. The severe form of the disorder is called typical mucolipidosis type IV, and the mild f...
3. `medquad_ans_0000675-2` (score=0.0479) - Mucolipidosis III alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
4. `medquad_ans_0000674-2` (score=0.0469) - Mucolipidosis II alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
5. `medquad_ans_0004153-1` (score=0.0462) - Mucolipidosis type 4 is a metabolic condition that affects the body's ability to process certain carbohydrates and fats. As a result, these materials accumulate in cells leading to the various signs and symptoms of th...

**Answer**:

Mucolipidosis type IV is estimated to occur in 1 in 40,000 people [medquad_ans_0000677-2].

**Supported citations**: `medquad_ans_0000677-2`
**Unsupported citations**: _none_

### Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Scores**: context_relevance=0.750, groundedness=0.994, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.0656) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chi...
2. `medquad_ans_0000193-1__dup4` (score=0.0621) - Cirrhosis\* is scarring of the liver. Scar tissue forms because of injury or long-term disease. Scar tissue replaces healthy liver tissue and blocks the normal flow of blood through the liver. A healthy liver - makes p...
3. `medquad_ans_0005201-1` (score=0.0570) - Punctate palmoplantar keratoderma type I, also known as keratosis palmoplantaris papulosa (KPPP) or Brauer-Buschke-Fisher Syndrome is is a rare condition that affects the skin. It is a type of punctate palmoplantar ke...
4. `medquad_ans_0000120-11` (score=0.0569) - People can ask their health care providers about surgical techniques that minimize scar tissue. Scientists have not yet found a way to prevent short bowel syndrome that is present at birth, as its cause is unknown.
5. `medquad_ans_0000977-1__dup1` (score=0.0538) - An injury is damage to your body. It is a general term that refers to harm caused by accidents, falls, hits, weapons, and more. In the U.S., millions of people injure themselves every year. These injuries range from m...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams. [medquad_ans_0000795-1__dup1]

**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

### Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.786, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0001045-1` (score=0.0656) - X-linked adrenoleukodystrophy is a genetic disorder that occurs primarily in males. It mainly affects the nervous system and the adrenal glands, which are small glands located on top of each kidney. In this disorder, ...
2. `medquad_ans_0006488-4` (score=0.0640) - How is X-linked adrenal hypoplasia congenita inherited? X-linked adrenal hypoplasia congenita is inherited in an X-linked recessive pattern. A condition is considered X-linked if the mutated gene that causes the disor...
3. `medquad_ans_0006488-1` (score=0.0640) - X-linked adrenal hypoplasia congenita is an inherited disorder that mainly affects males. It involves many hormone-producing (endocrine) tissues in the body, particularly a pair of small glands on top of each kidney c...
4. `medquad_ans_0006502-4` (score=0.0616) - How is X-linked hypophosphatemia inherited? X-linked hypophosphatemia (XLH) is caused by mutations in the PHEX gene, and is inherited in an X-linked dominant manner. This means that the gene responsible for the condit...
5. `medquad_ans_0006553-1` (score=0.0611) - Congenital adrenal hyperplasia (CAH) due to 11-beta-hydroxylase deficiency is one of a group of disorders (collectively called congenital adrenal hyperplasia) that affect the adrenal glands. In this condition, the adr...

**Answer**:

The provided context does not specify the exact genetic changes related to X-linked adrenoleukodystrophy. It describes the disorder's effects on the nervous system and adrenal glands, as well as its types and symptoms, but does not detail the underlying genetic mutations or molecular mechanisms. [medquad_ans_0001045-1]

**Supported citations**: `medquad_ans_0001045-1`
**Unsupported citations**: _none_

### Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Scores**: context_relevance=0.625, groundedness=0.719, answer_relevance=0.625

**Top docs**:

1. `medquad_ans_0000136-1__dup4` (score=0.0479) - Amyloidosis is a rare disease that occurs when amyloid proteins are deposited in tissues and organs. Amyloid proteins are abnormal proteins that the body cannot break down and recycle, as it does with normal proteins....
2. `medquad_ans_0000027-1__dup3` (score=0.0460) - Amyloidosis occurs when abnormal proteins called amyloids build up and form deposits. The deposits can collect in organs such as the kidney and heart. This can cause the organs to become stiff and unable to work the w...
3. `medquad_ans_0000136-11` (score=0.0439) - - Amyloidosis is a rare disease that occurs when amyloid proteins are deposited in tissues and organs. - Primary amyloidosis and dialysis-related amyloidosis are the types of amyloidosis that can affect the kidneys. -...
4. `medquad_ans_0000136-4__dup3` (score=0.0438) - The most common sign of primary amyloidosis of the kidneys is nephrotic syndromea collection of signs that indicate kidney damage. The signs of nephrotic syndrome include - albuminuriaan increased amount of albumin, a...
5. `medquad_ans_0000136-5__dup2` (score=0.0421) - The symptoms of dialysis-related amyloidosis may include - pain, stiffness, and fluid in the joints. - abnormal, fluid-containing sacs, called cysts, in some bones. - carpal tunnel syndrome, caused by unusual buildup ...

**Answer**:

The provided context does not mention the number of people affected by transthyretin amyloidosis. [medquad_ans_0000136-1\_\_dup4][medquad_ans_0000027-1__dup3][medquad_ans_0000136-11] discuss general information about amyloidosis, including organ involvement and types such as primary, secondary, and familial, but none specify the prevalence or number of individuals affected by transthyretin amyloidosis. Therefore, the answer cannot be determined from the given context.

**Supported citations**: `medquad_ans_0000136-1__dup4`, `medquad_ans_0000027-1__dup3`, `medquad_ans_0000136-11`
**Unsupported citations**: _none_

### Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Scores**: context_relevance=1.000, groundedness=0.904, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.0820) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the informatio...
2. `medquad_ans_0001787-1` (score=0.0792) - What are the signs and symptoms of Dentin dysplasia, type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Dentin dysplasia, type 1. If the information is available, the table belo...
3. `medquad_ans_0000754-1` (score=0.0754) - Oral-facial-digital syndrome is actually a group of related conditions that affect the development of the oral cavity (the mouth and teeth), facial features, and digits (fingers and toes). Researchers have identified ...
4. `medquad_ans_0000024_3-3` (score=0.0736) - Signs of lip and oral cavity cancer include a sore or lump on the lips or in the mouth. These and other signs and symptoms may be caused by lip and oral cavity cancer or by other conditions. Check with your doctor if ...
5. `medquad_ans_0001133-2` (score=0.0704) - What are the signs and symptoms of Cherubism? Cherubism is characterized by abnormal bone tissue in the lower part of the face. Beginning in early childhood, both the lower jaw (the mandible) and the upper jaw (the ma...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include microdontia, taurodontia, dentinogenesis imperfecta limited to primary teeth, obliteration of the pulp chamber, and periapical radiolucency. These symptoms are reported in the Human Phenotype Ontology (HPO) database, which compiles information from sources such as Orphanet. The frequency of these symptoms is typically listed as a rough estimate of the percentage of patients who have the feature, or as a fraction indicating the number of patients with the symptom out of the total number examined in a study [medquad_ans_0001780-1]. However, specific symptom frequencies are not detailed in the provided context.

**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

### Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Scores**: context_relevance=0.875, groundedness=0.931, answer_relevance=0.375

**Top docs**:

1. `medquad_ans_0000782-2` (score=0.0492) - Periventricular heterotopia is a rare condition. Its incidence is unknown.
2. `medquad_ans_0004818-2` (score=0.0484) - What are the signs and symptoms of periventricular nodular heterotopia? The condition is first noticed when seizures appear, often during the teenage years. The nodules around the ventricles are then typically discove...
3. `medquad_ans_0000782-1` (score=0.0469) - Periventricular heterotopia is a condition in which nerve cells (neurons) do not migrate properly during the early development of the fetal brain, from about the 6th week to the 24th week of pregnancy. Heterotopia mea...
4. `medquad_ans_0004818-4` (score=0.0464) - How might periventricular nodular heterotopia be treated? Treatment of epilepsy generally follows principles for a seizure disorder caused by a known structural brain abnormality; carbamezipine is most often used, bec...
5. `medquad_ans_0004818-3` (score=0.0462) - What are the recommended evaluations for patients diagnosed with periventricular nodular heterotopia? The following evaluations are recommended:[1823] Imaging exams of the brain to establish the diagnosis Evaluation b...

**Answer**:

The incidence of periventricular heterotopia is unknown [medquad_ans_0000782-2].

**Supported citations**: `medquad_ans_0000782-2`
**Unsupported citations**: _none_

### Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Scores**: context_relevance=0.727, groundedness=0.709, answer_relevance=0.655

**Top docs**:

1. `medquad_ans_0003462-3` (score=0.0492) - How is Landau-Kleffner syndrome (LKS) diagnosed? LKS is diagnosed based on clinical features and the results of an electroencephalogram (EEG), a recording of the electric activity of the brain. All LKS children have a...
2. `medquad_ans_0003462-2` (score=0.0484) - What are the signs and symptoms of Landau-Kleffner syndrome? Landau-Kleffner syndrome is characterized by the sudden or gradual development of aphasia (the inability to understand or express language) in previously no...
3. `medquad_ans_0003462-1` (score=0.0476) - Landau-Kleffner syndrome (LKS) is a rare, childhood neurological disorder characterized by the sudden or gradual development of aphasia (the inability to understand or express language) and an abnormal electro-encepha...
4. `medquad_ans_0000004-1__dup2` (score=0.0469) - Landau-Kleffner syndrome (LKS) is a rare, childhood neurological disorder characterized by the sudden or gradual development of aphasia (the inability to understand or express language) and an abnormal electro-encepha...
5. `medquad_ans_0000190-4__dup1` (score=0.0459) - The NINDS supports research on neurological disorders such as Melkersson-Rosenthal syndrome. Much of this research is aimed at increasing knowledge of these disorders and finding ways to treat, prevent, and ultimately...

**Answer**:

The provided context does not mention any ongoing research or clinical trials for Landau-Kleffner Syndrome. [medquad_ans_0003462-3][medquad_ans_0003462-2]

**Supported citations**: `medquad_ans_0003462-3`, `medquad_ans_0003462-2`
**Unsupported citations**: _none_

### Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.323, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000631-1__dup1` (score=0.0651) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...
2. `medquad_ans_0001935-1` (score=0.0624) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystroph...
3. `medquad_ans_0003601-1` (score=0.0612) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
4. `medquad_ans_0001537-1` (score=0.0594) - Congenital muscular dystrophy (CMD) refers to a group of inherited conditions that affect the muscles and are present at birth or in early infancy. The severity of the condition, the associated signs and symptoms and ...
5. `medquad_ans_0001935-4` (score=0.0594) - How is Duchenne muscular dystrophy (DMD) diagnosed? Duchenne muscular dystrophy (DMD) is suspected and diagnosed when the following clinical findings are found: a positive family history of DMD, more men affected that...

**Answer**:

Muscular dystrophy (MD) is a group of more than 30 inherited diseases that cause muscle weakness and muscle loss. The different types of MD can vary in whom they affect, which muscles they affect, and the symptoms they present. Some forms appear in infancy or childhood, while others may not appear until middle age or later. All forms of MD worsen as the muscles weaken, and most people eventually lose the ability to walk. There is no cure for muscular dystrophy; however, treatments such as physical and speech therapy, orthopedic devices, surgery, and medications can help manage symptoms and prevent complications. The severity of the condition ranges from mild cases that worsen slowly to disabling and severe cases [medquad_ans_0000631-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000631-1`

### Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.342, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.0656) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that t...
2. `medquad_ans_0005019-1` (score=0.0623) - What are the signs and symptoms of Preaxial polydactyly type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 1. If the information is available, the tabl...
3. `medquad_ans_0000754-5` (score=0.0617) - These resources address the diagnosis or management of oral-facial-digital syndrome: - Gene Review: Gene Review: Oral-Facial-Digital Syndrome Type I - Genetic Testing Registry: Mohr syndrome - Genetic Testing Registry...
4. `medquad_ans_0004956-1` (score=0.0607) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. Specifically, the surface of the brain develops too many folds which are unusually small. The signs and symptoms associate...
5. `medquad_ans_0000024-1` (score=0.0592) - Adult polyglucosan body disease is a condition that affects the nervous system. People with this condition have problems walking due to reduced sensation in their legs (peripheral neuropathy) and progressive muscle we...

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding with no other physical anomalies or intellectual impairment, or it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits, with surgical management depending greatly on the complexity of the deformity [1] [2].

**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

### Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Scores**: context_relevance=0.833, groundedness=0.052, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0004534-2` (score=0.0484) - What are the signs and symptoms of Omenn syndrome? Infants with Omenn syndrome typically present shortly after birth, usually by 3 months of age. This is similar to other types of severe combined immunodeficiency (SCI...
2. `medquad_ans_0006244-2` (score=0.0464) - What are the signs and symptoms of Tumor necrosis factor receptor-associated periodic syndrome? The characteristic feature of TRAPS is recurrent episodes of fever. Episodes may begin at any age, but most often begin i...
3. `medquad_ans_0002992-2` (score=0.0460) - What are the signs and symptoms of Hyper-IgD syndrome? Hyper IgD syndrome is characterized by periodic high fevers accompanied by lymphadenopathy, abdominal pain, diarrhea, headache, joint pain, hepatomegaly and/or sp...
4. `medquad_ans_0002628-2` (score=0.0460) - What are the signs and symptoms of GM1 gangliosidosis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for GM1 gangliosidosis type 2. If the information is available, the table be...
5. `medquad_ans_0000145-3__dup3` (score=0.0429) - The symptoms of Goodpasture syndrome may initially include fatigue, nausea, vomiting, and weakness. The lungs are usually affected before or at the same time as the kidneys, and symptoms can include shortness of breat...

**Answer**:

The provided context does not contain any information about the symptoms of Geniospasm. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=0.900, groundedness=0.761, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000049-3__dup2` (score=0.0628) - A defect in the CFTR gene causes cystic fibrosis (CF). This gene makes a protein that controls the movement of salt and water in and out of your body's cells. In people who have CF, the gene makes a protein that doesn...
2. `medquad_ans_0000089-2__dup2` (score=0.0609) - Marfan syndrome is a genetic disorder. A mutation, or change, in the gene that controls how the body makes fibrillin causes Marfan syndrome. Fibrillin is a protein that plays a major role in your body's connective tis...
3. `medquad_ans_0000078-1__dup2` (score=0.0591) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...
4. `medquad_ans_0000758-1__dup2` (score=0.0587) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
5. `medquad_ans_0000253-1__dup2` (score=0.0574) - Cystic fibrosis (CF) is an inherited disease of the mucus and sweat glands. It affects mostly your lungs, pancreas, liver, intestines, sinuses, and sex organs. CF causes your mucus to be thick and sticky. The mucus cl...

**Answer**:

The provided context does not contain information about genetic changes related to idiopathic pulmonary fibrosis (IPF). While it describes IPF as a condition where fibrosis occurs without a known cause and notes that doctors often cannot find a cause, it does not specify any genetic mutations or changes associated with IPF [medquad_ans_0000078-1__dup2]. Therefore, based on the given context, the genetic changes related to idiopathic pulmonary fibrosis cannot be determined.

**Supported citations**: `medquad_ans_0000078-1__dup2`
**Unsupported citations**: _none_

### Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.644, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000813-2` (score=0.0328) - Potocki-Shaffer syndrome is a rare condition, although its prevalence is unknown. Fewer than 100 cases have been reported in the scientific literature.
2. `medquad_ans_0005008-1` (score=0.0320) - Potocki-Shaffer syndrome is a contiguous gene deletion syndrome associated with deletions in a specific region of chromosome 11 (11p11.2). The characteristic features of Potocki-Shaffer syndrome include openings in th...
3. `medquad_ans_0000813-5` (score=0.0320) - These resources address the diagnosis or management of Potocki-Shaffer syndrome: - Genetic Testing Registry: Potocki-Shaffer syndrome These resources from MedlinePlus offer information about the diagnosis and manageme...
4. `medquad_ans_0000813-1` (score=0.0310) - Potocki-Shaffer syndrome is a disorder that affects development of the bones, nerve cells in the brain, and other tissues. Most people with this condition have multiple noncancerous (benign) bone tumors called osteoch...
5. `medquad_ans_0000813-3` (score=0.0305) - Potocki-Shaffer syndrome (also known as proximal 11p deletion syndrome) is caused by a deletion of genetic material from the short (p) arm of chromosome 11 at a position designated 11p11.2. The size of the deletion va...

**Answer**:

The provided context does not address whether Potocki-Shaffer syndrome is inherited. Therefore, based on the available information, it cannot be determined if the syndrome is inherited. [medquad_ans_0005008-1] discusses the genetic basis of the syndrome, including deletions in chromosome 11, but does not specify inheritance patterns.

**Supported citations**: `medquad_ans_0005008-1`
**Unsupported citations**: _none_

### Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Scores**: context_relevance=0.857, groundedness=0.953, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000177-1__dup2` (score=0.0328) - Summary : Children's healthy development depends on safe and positive experiences when they are very young. If you work or go to school, you want to know that your child is in good hands while you are away. You may ch...
2. `medquad_ans_0000189-1__dup2` (score=0.0315) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...
3. `medquad_ans_0000227-5` (score=0.0313) - These resources address the diagnosis or management of CHILD syndrome: - Gene Review: Gene Review: NSDHL-Related Disorders - Genetic Testing Registry: Child syndrome These resources from MedlinePlus offer information ...
4. `medquad_ans_0000038-7` (score=0.0304) - Doctors diagnose childhood interstitial lung disease (chILD) based on a child's medical and family histories and the results from tests and procedures. To diagnose chILD, doctors may first need to rule out other disea...
5. `medquad_ans_0000744-1__dup2` (score=0.0300) - Summary : Prenatal care is the health care you get while you are pregnant. It includes your checkups and prenatal testing. Prenatal care can help keep you and your baby healthy. It lets your health care provider spot ...

**Answer**:

Yes, there is information about child care. Child care options include in-home care, where the caregiver comes to your home, or the child going to the caregiver's home, and child care centers. It is important to get to know your child's caregivers, as they will be a big part of your child's life. The caregiver's training should include knowledge of how young children learn and grow, positive and consistent discipline, signs that a child is sick, cleanliness and safety practices to prevent illness or injury, and basic first aid [medquad_ans_0000177-1__dup2].

**Supported citations**: `medquad_ans_0000177-1__dup2`
**Unsupported citations**: _none_

### Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Scores**: context_relevance=0.800, groundedness=0.954, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000046-2` (score=0.0492) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
2. `medquad_ans_0000016-3__dup1` (score=0.0481) - Children with the benign form of alternating hemiplegia have a good prognosis. Those who experience the more severe form have a poor prognosis because intellectual and mental capacities do not respond to drug therapy,...
3. `medquad_ans_0000046-5` (score=0.0479) - These resources address the diagnosis or management of alternating hemiplegia of childhood: - The Great Ormond Street Hospital - University of Utah School of Medicine These resources from MedlinePlus offer information...
4. `medquad_ans_0000046-1` (score=0.0469) - Alternating hemiplegia of childhood is a neurological condition characterized by recurrent episodes of temporary paralysis, often affecting one side of the body (hemiplegia). During some episodes, the paralysis altern...
5. `medquad_ans_0000016-1__dup1` (score=0.0462) - Alternating hemiplegia is a rare neurological disorder that develops in childhood, most often before the child is 18 months old. The disorder is characterized by recurrent episodes of paralysis that involve one or bot...

**Answer**:

Approximately 1 in 1 million people are affected by alternating hemiplegia of childhood [medquad_ans_0000046-2].

**Supported citations**: `medquad_ans_0000046-2`
**Unsupported citations**: _none_

### Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000186-1__dup1` (score=0.0804) - Lupus (also called systemic lupus erythematosus) is a disorder of the immune system. Normally, the immune system protects the body against invading infections and cancers. In lupus, the immune system is over-active an...
2. `medquad_ans_0000159-3__dup2` (score=0.0799) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time...
3. `medquad_ans_0000569-1__dup2` (score=0.0789) - If you have lupus, your immune system attacks healthy cells and tissues by mistake. This can damage your joints, skin, blood vessels and organs. There are many kinds of lupus. The most common type, systemic lupus eryt...
4. `medquad_ans_0003688-1` (score=0.0779) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, h...
5. `medquad_ans_0003687-2` (score=0.0640) - What are the signs and symptoms of Lupus? You can read about the signs and symptoms of lupus from MedlinePlus and the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS). The Human Phenotype ...

**Answer**:

The symptoms of lupus include joint pain or swelling, muscle pain, fever with no known cause, fatigue, red rashes often on the face (also called the "butterfly rash"), vascular headaches, epilepsy, psychoses, mild cognitive dysfunction, organic brain syndrome, peripheral neuropathies, sensory neuropathy, psychological problems (including personality changes, paranoia, mania, and schizophrenia), seizures, transverse myelitis, paralysis, and stroke. Additionally, in lupus nephritis, symptoms may include high blood pressure, foamy urine, and edema (swelling), usually in the legs, feet, or ankles and less often in the hands or face [medquad_ans_0000186-1\_\_dup1][medquad_ans_0000159-3__dup2][medquad_ans_0000569-1__dup2].

**Supported citations**: `medquad_ans_0000186-1__dup1`, `medquad_ans_0000159-3__dup2`, `medquad_ans_0000569-1__dup2`
**Unsupported citations**: _none_

### Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.544, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000194-2` (score=0.0489) - Cockayne syndrome occurs in about 2 per million newborns in the United States and Europe.
2. `medquad_ans_0000919-1__dup1` (score=0.0486) - Tuberous sclerosis is a rare genetic disease that causes benign tumors to grow in the brain and other organs. Symptoms vary, depending on where the tumors grow. They could include - Skin problems, such as light patche...
3. `medquad_ans_0000194-1` (score=0.0481) - Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features of this disorder include a failure to gain weight and grow at the expected rate (failure to thrive), a...
4. `medquad_ans_0000194-5` (score=0.0474) - These resources address the diagnosis or management of Cockayne syndrome: - Gene Review: Gene Review: Cockayne Syndrome - Genetic Testing Registry: Cockayne syndrome - Genetic Testing Registry: Cockayne syndrome type ...
5. `medquad_ans_0001397-1` (score=0.0469) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...

**Answer**:

Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features include failure to gain weight and grow at the expected rate (failure to thrive), abnormally small head size (microcephaly), and impaired development of the nervous system. Affected individuals have extreme sensitivity to sunlight (photosensitivity), and even small amounts of sun exposure can cause sunburn. Other possible signs and symptoms include hearing loss, eye abnormalities, severe tooth decay, bone abnormalities, and changes in the brain that can be seen on brain scans. Cockayne syndrome can be divided into subtypes: classical (type I), which has an onset in early childhood (usually after age 1 year; sometimes called type I), type II, which has more severe symptoms apparent at birth (congenital) and is sometimes called cerebro-oculo-facio-skeletal (COFS) syndrome or Pena-Shokeir syndrome type II, and type III

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Scores**: context_relevance=0.833, groundedness=0.917, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000029-6__dup1` (score=0.0650) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead...
2. `medquad_ans_0000029-8__dup1` (score=0.0645) - Gingivitis is inflammation of the gums. In gingivitis, the gums become red, swollen and can bleed easily. Gingivitis is a mild form of gum disease. It can usually be reversed with daily brushing and flossing, and regu...
3. `medquad_ans_0000029-12` (score=0.0635) - In some studies, researchers have observed that people with periodontal disease (when compared to people without periodontal disease) were more likely to develop heart disease or have difficulty controlling their bloo...
4. `medquad_ans_0000413-1__dup1` (score=0.0625) - If you have gum disease, you're not alone. Many U.S. adults currently have some form of the disease. It ranges from simple gum inflammation, called gingivitis, to serious damage to the tissue and bone supporting the t...
5. `medquad_ans_0000029-1__dup5` (score=0.0620) - An Infection of the Gums and Surrounding Tissues Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. The two forms of gum disease are gingivitis, a mild form that is...

**Answer**:

Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. It ranges from a mild form called gingivitis, which involves red, swollen gums that can bleed easily and is reversible with daily brushing and flossing and regular dental cleanings, to more serious forms such as periodontitis, where the gums pull away from the teeth and form infected pockets, leading to destruction of the bones, gums, and connective tissue that support the teeth [medquad_ans_0000029-6\_\_dup1][medquad_ans_0000413-1__dup1][medquad_ans_0000029-8__dup1].

**Supported citations**: `medquad_ans_0000029-6__dup1`, `medquad_ans_0000413-1__dup1`, `medquad_ans_0000029-8__dup1`
**Unsupported citations**: _none_

### Example 212 (`0000494-2`)

**Question**: How many people are affected by hyperkalemic periodic paralysis ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.0492) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0000494-5` (score=0.0484) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis ...
3. `medquad_ans_0000494-1` (score=0.0476) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to...
4. `medquad_ans_0003000-2` (score=0.0469) - What are the signs and symptoms of Hyperkalemic periodic paralysis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hyperkalemic periodic paralysis. If the information is available, ...
5. `medquad_ans_0003000-1` (score=0.0459) - Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move...

**Answer**:

An estimated 1 in 200,000 people are affected by hyperkalemic periodic paralysis [medquad_ans_0000494-2].

**Supported citations**: `medquad_ans_0000494-2`
**Unsupported citations**: _none_

### Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Scores**: context_relevance=1.000, groundedness=0.991, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0005212-1` (score=0.0656) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end o...
2. `medquad_ans_0004813-1` (score=0.0640) - Periodic fever, aphthous stomatitis, pharyngitis, cervical adenitis (PFAPA) is a periodic disease, which is a heterogeneous group of disorders characterized by short episodes of illness that regularly recur for severa...
3. `medquad_ans_0000054-7` (score=0.0542) - Peripheral arterial disease (P.A.D.) is a disease in which plaque (plak) builds up in the arteries that carry blood to your head, organs, and limbs. Plaque is made up of fat, cholesterol, calcium, fibrous tissue, and ...
4. `medquad_ans_0004844-1` (score=0.0540) - Phacomatosis pigmentovascularis (PPV) is a skin and blood vessel disorder that is present from birth. Common signs and symptoms include port wine stain and pigmentary lesions, such as melanocytic nevi or epidermal nev...
5. `medquad_ans_0000424-2__dup1` (score=0.0519) - What are the signs and symptoms of Aplasia cutis congenita? The Human Phenotype Ontology provides the following list of signs and symptoms for Aplasia cutis congenita. If the information is available, the table below ...

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (relative constriction of the diaphysis and flaring of the metaphysis), widening of the ribs and clavicles, platyspondyly (flattening of the spine bones), and cortical thinning. Cranial involvement is minimal, with some individuals showing mild hyperostosis of the skull base and thickening of the frontal and occipital bones. Pyle disease is inherited in an autosomal recessive manner and only about 30 cases have been reported in the literature [medquad_ans_0005212-1].

**Supported citations**: `medquad_ans_0005212-1`
**Unsupported citations**: _none_

### Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Scores**: context_relevance=0.600, groundedness=0.920, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000051-5__dup1` (score=0.0797) - The signs and symptoms of deep vein thrombosis (DVT) might be related to DVT itself or pulmonary embolism (PE). See your doctor right away if you have signs or symptoms of either condition. Both DVT and PE can cause s...
2. `medquad_ans_0000051-4__dup2` (score=0.0796) - The risk factors for deep vein thrombosis (DVT) include: A history of DVT. Conditions or factors that make your blood thicker or more likely to clot than normal. Some inherited blood disorders (such as factor V Leiden...
3. `medquad_ans_0000256-1__dup3` (score=0.0790) - Deep vein thrombosis, or DVT, is a blood clot that forms in a vein deep in the body. Most deep vein clots occur in the lower leg or thigh. If the vein swells, the condition is called thrombophlebitis. A deep vein thro...
4. `medquad_ans_0000104-1__dup4` (score=0.0787) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent b...
5. `medquad_ans_0000051-1__dup2` (score=0.0773) - Espaol Deep vein thrombosis (throm-BO-sis), or DVT, is a blood clot that forms in a vein deep in the body. Blood clots occur when blood thickens and clumps together. Most deep vein blood clots occur in the lower leg o...

**Answer**:

Deep vein thrombosis (DVT) is caused by a combination of factors that make the blood more likely to clot or slow blood flow in deep veins. These include: conditions or factors that make blood thicker or more likely to clot (such as inherited blood disorders like factor V Leiden), hormone therapy or birth control pills, injury to a deep vein from surgery, broken bones, or trauma, slow blood flow due to lack of movement (e.g., after surgery, being ill and bedridden, or long travel), pregnancy and the first 6 weeks postpartum, recent or ongoing cancer treatment, presence of a central venous catheter, older age (over 60), overweight or obesity, and smoking [medquad_ans_0000051-4__dup2]. Additionally, sitting still for long periods can increase the likelihood of DVT [medquad_ans_000256-1\_\_dup

**Supported citations**: `medquad_ans_0000051-4__dup2`
**Unsupported citations**: _none_

### Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.108, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004079-1` (score=0.0328) - Complex II deficiency is a mitochondrial disease. Mitochondria are specialized compartments in cells that create more than 90% of the energy needed by the body. In mitochondrial diseases, the mitochondria don't work c...
2. `medquad_ans_0004079-4` (score=0.0323) - How might mitochondrial complex II deficiency be treated? Treatment options for complex II deficiency may be similar to those for other mitochondrial disorders in general.[8677] The United Mitochondrial Disease Founda...
3. `medquad_ans_0000661-1` (score=0.0317) - Mitochondrial complex III deficiency is a genetic condition that can affect several parts of the body, including the brain, kidneys, liver, heart, and the muscles used for movement (skeletal muscles). Signs and sympto...
4. `medquad_ans_0004081-2` (score=0.0312) - What are the signs and symptoms of Mitochondrial complex IV deficiency? There are currently 4 known forms of COX deficiency. The range and severity of signs and symptoms can vary widely from case to case. In one form,...
5. `medquad_ans_0004079-2` (score=0.0308) - What are the signs and symptoms of Mitochondrial complex II deficiency? The signs and symptoms of mitochondrial complex II deficiency can vary greatly from severe life-threatening symptoms in infancy to muscle disease...

**Answer**:

The provided context does not contain any information about mitochondrial complex IV deficiency. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Scores**: context_relevance=1.000, groundedness=0.497, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000078-2` (score=0.0653) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.
2. `medquad_ans_0000078-5` (score=0.0643) - These resources address the diagnosis or management of ataxia with oculomotor apraxia: - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type 1 - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type ...
3. `medquad_ans_0000078-1` (score=0.0638) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...
4. `medquad_ans_0000496-1__dup1` (score=0.0627) - What are the signs and symptoms of Ataxia with Oculomotor Apraxia Type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Ataxia with Oculomotor Apraxia Type 2. If the information is...
5. `medquad_ans_0000495-2__dup1` (score=0.0615) - What are the signs and symptoms of Ataxia telangiectasia? Ataxia-telangiectasia affects the nervous system, immune system, and other body systems. This disorder is characterized by progressive difficulty with coordina...

**Answer**:

Ataxia with oculomotor apraxia is a rare condition characterized by progressive problems with movement, including difficulty coordinating movements (ataxia), which is often the first symptom. The hallmark feature is oculomotor apraxia, making it difficult to move the eyes side-to-side, requiring individuals to turn their head to see peripheral vision. There are multiple types, with types 1 and 2 being the most common. Type 1 typically begins around age 4, while type 2 begins around age 15. Both types share features such as involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, chorea and myoclonus tend to disappear gradually, whereas in type 2, these movement problems persist throughout life. Type 1 individuals often develop wasting (atrophy) in hands and feet, and nearly all

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Scores**: context_relevance=1.000, groundedness=0.923, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000221-1` (score=0.0656) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that fo...
2. `medquad_ans_0001744-1` (score=0.0645) - What are the signs and symptoms of Deafness with labyrinthine aplasia microtia and microdontia (LAMM)? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness with labyrinthine apla...
3. `medquad_ans_0000221-5` (score=0.0635) - These resources address the diagnosis or management of LAMM syndrome: - Gene Review: Gene Review: Congenital Deafness with Labyrinthine Aplasia, Microtia, and Microdontia - Genetic Testing Registry: Deafness with laby...
4. `medquad_ans_0000623-1` (score=0.0625) - Mandibulofacial dysostosis with microcephaly (MFDM) is a disorder that causes abnormalities of the head and face. People with this disorder often have an unusually small head at birth, and the head does not grow at th...
5. `medquad_ans_0000248-1` (score=0.0597) - Craniofacial microsomia is a term used to describe a spectrum of abnormalities that primarily affect the development of the skull (cranium) and face before birth. Microsomia means abnormal smallness of body structures...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects the development of the ears and teeth. It is characterized by the complete absence of inner ear structures (labyrinthine aplasia), leading to congenital sensorineural hearing loss. Affected individuals often have abnormally small outer ears (microtia) with narrow ear canals and unusually small, widely spaced teeth (microdontia). The condition may also cause delayed motor skill development due to balance issues from inner ear abnormalities. Additional signs include abnormalities of cranial nerves (seen in approximately 90% of patients), microdontia (90%), and other features such as pointed chin, wide nasal bridge, and preauricular skin tags [medquad_ans_0000221-1][medquad_ans_0001744-1]. The

**Supported citations**: `medquad_ans_0000221-1`, `medquad_ans_0001744-1`
**Unsupported citations**: _none_

### Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000064-4` (score=0.0328) - Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases of Apert syndrome result from new mutations in ...
2. `medquad_ans_0000064-2` (score=0.0323) - Apert syndrome affects an estimated 1 in 65,000 to 88,000 newborns.
3. `medquad_ans_0000064-1` (score=0.0313) - Apert syndrome is a genetic disorder characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the head and fa...
4. `medquad_ans_0000420-1__dup1` (score=0.0312) - Apert syndrome is a disorder mainly characterized by craniosynostosis (premature fusion of skull bones, causing abnormalities in the shape of the head and face) and syndactyly (fusion or webbing or fingers and/or toes...
5. `medquad_ans_0000064-5` (score=0.0308) - These resources address the diagnosis or management of Apert syndrome: - Gene Review: Gene Review: FGFR-Related Craniosynostosis Syndromes - Genetic Testing Registry: Acrocephalosyndactyly type I - MedlinePlus Encyclo...

**Answer**:

Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. However, almost all cases result from new mutations and occur in people with no family history of the disorder. Individuals with Apert syndrome can pass the condition to their offspring [medquad_ans_0000064-4].

**Supported citations**: `medquad_ans_0000064-4`
**Unsupported citations**: _none_

### Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Scores**: context_relevance=0.750, groundedness=0.550, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.0656) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organi...
2. `medquad_ans_0000341-3__dup2` (score=0.0640) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact ...
3. `medquad_ans_0000341-4__dup2` (score=0.0633) - In the United States, Q fever outbreaks have resulted mainly from occupational exposure involving veterinarians, meat processing plant workers, sheep and dairy workers, livestock farmers, and researchers at facilities...
4. `medquad_ans_0000341-1__dup3` (score=0.0632) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella bu...
5. `medquad_ans_0000092-4__dup3` (score=0.0608) - CHHF virus has been successfully isolated from both blood and serum during the acute febrile phase of illness. Although not undertaken at the time of the initial cluster, virus can certainly be isolated from tissue ob...

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air that contains airborne barnyard dust contaminated by dried placental material, birth fluids, and excreta of infected animals. Other modes of transmission, including tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission, are rare. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, people with acute Q fever develop a chronic form of the disease. Treatment of the acute form is with antibiotics,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000038-1__dup2` (score=0.0328) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chroni...
2. `medquad_ans_0000038-5__dup1` (score=0.0315) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
3. `medquad_ans_0001152-2` (score=0.0315) - How might chILD be treated? There is no single treatment for interstitial lung diseases in children. Different forms of chILD require different treatments and support depending on the condition. The goals of treatment...
4. `medquad_ans_0000038-7` (score=0.0313) - Doctors diagnose childhood interstitial lung disease (chILD) based on a child's medical and family histories and the results from tests and procedures. To diagnose chILD, doctors may first need to rule out other disea...
5. `medquad_ans_0000038-9` (score=0.0310) - At this time, most types of childhood interstitial lung disease (chILD) can't be prevented. People who have a family history of inherited (genetic) interstitial lung disease may want to consider genetic counseling. A ...

**Answer**:

Babies, children, and teens are at risk for Childhood Interstitial Lung Disease (chILD) [medquad_ans_0000038-1__dup2].

**Supported citations**: `medquad_ans_0000038-1__dup2`
**Unsupported citations**: _none_

### Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Scores**: context_relevance=1.000, groundedness=0.975, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.0656) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in lat...
2. `medquad_ans_0000547-1` (score=0.0645) - Juvenile idiopathic arthritis refers to a group of conditions involving joint inflammation (arthritis) that first appears before the age of 16. This condition is an autoimmune disorder, which means that the immune sys...
3. `medquad_ans_0000087-1__dup1` (score=0.0632) - Vasculitis is an inflammation of blood vessels, which includes the veins, arteries, and capillaries. Inflammation occurs with infection or is thought to be due to a faulty immune system response. It also can be caused...
4. `medquad_ans_0000404-1__dup1` (score=0.0620) - Giant cell arteritis is a disorder that causes inflammation of your arteries, usually in the scalp, neck, and arms. It narrows the arteries, which keeps blood from flowing well. Giant cell arteritis often occurs with ...
5. `medquad_ans_0003322-1` (score=0.0604) - What are the signs and symptoms of Juvenile idiopathic arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Juvenile idiopathic arthritis. If the information is available, the ...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels. It is generally diagnosed in late childhood or early adulthood and specifically affects the temporal arteries, which are located at the lower sides of the skull, directly underneath the temple. Affected individuals often have no signs or symptoms aside from a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically with no family history. It is generally treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

### Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.499, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0006577-1` (score=0.0656) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particul...
2. `medquad_ans_0001082-2` (score=0.0643) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyry...
3. `medquad_ans_0001082-1` (score=0.0637) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as...
4. `medquad_ans_0001090-2` (score=0.0620) - The exact incidence of 3-hydroxyacyl-CoA dehydrogenase deficiency is unknown; it has been reported in only a small number of people worldwide.
5. `medquad_ans_0001082-5` (score=0.0620) - These resources address the diagnosis or management of 2-methylbutyryl-CoA dehydrogenase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of 2-methylbutyryl-CoA dehydrogenase These resources from...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a rare metabolic disorder in which individuals lack adequate levels of the enzyme 2-methylbutyryl-CoA dehydrogenase, which is necessary for the proper processing of the amino acid isoleucine. This deficiency leads to the buildup of isoleucine and organic acids in the body, causing a range of health problems that vary from severe and life-threatening to mild or absent. Initial symptoms often include poor feeding, lack of energy (lethargy), vomiting, and irritability, and may progress to serious medical issues such as difficulty breathing, seizures, coma, poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills. Symptoms can be triggered by fasting, infections, or high-protein diets. The disorder is more common among individuals of Hmong ancestry, occurring in 1 in 250 to 1 in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005840-1` (score=0.0799) - Spondyloepiphyseal dysplasia congenita is an inherited disorder of bone growth that affects the bones of the spine and ends of the long bones in the arms and legs. Features of this condition include short stature (dwa...
2. `medquad_ans_0005841-2` (score=0.0752) - What are the signs and symptoms of Spondyloepiphyseal dysplasia Maroteaux type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia Maroteaux type. If the in...
3. `medquad_ans_0000947-1` (score=0.0747) - Spondyloepiphyseal dysplasia congenita is an inherited bone growth disorder that results in short stature (dwarfism), skeletal abnormalities, and problems with vision and hearing. This condition affects the bones of t...
4. `medquad_ans_0000948-1` (score=0.0739) - Spondyloperipheral dysplasia is a disorder that impairs bone growth. This condition is characterized by flattened bones of the spine (platyspondyly) and unusually short fingers and toes (brachydactyly), with the excep...
5. `medquad_ans_0005840-2` (score=0.0736) - What are the signs and symptoms of Spondyloepiphyseal dysplasia congenita? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia congenita. If the information ...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include short stature (dwarfism), a very short trunk and neck, abnormal curvature of the spine, barrel-shaped chest, shortened limbs, an abnormality of the hip joint, and problems with vision and hearing. Arthritis and decreased joint mobility often develop early in life [medquad_ans_0005840-1].

**Supported citations**: `medquad_ans_0005840-1`
**Unsupported citations**: _none_

### Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Scores**: context_relevance=0.750, groundedness=0.935, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000853-2` (score=0.0820) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
2. `medquad_ans_0005219-1` (score=0.0806) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepti...
3. `medquad_ans_0000853-1` (score=0.0794) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepti...
4. `medquad_ans_0005219-2` (score=0.0776) - What are the signs and symptoms of Pyridoxine-dependent epilepsy? Those affected by pyridoxine-dependent epilepsy typically experience prolonged seizures lasting several minutes (status epilepticus). These seizures in...
5. `medquad_ans_0005219-4` (score=0.0769) - How might pyridoxine-dependent epilepsy be treated? Anticonvulsant drugs, which are usually given to control seizures, are ineffective in people with pyridoxine-dependent epilepsy. Instead, people with this type of se...

**Answer**:

Pyridoxine-dependent epilepsy is caused by mutations in the ALDH7A1 gene, which is inherited in an autosomal recessive fashion [medquad_ans_0005219-1].

**Supported citations**: `medquad_ans_0005219-1`
**Unsupported citations**: _none_

### Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Scores**: context_relevance=0.875, groundedness=0.906, answer_relevance=0.975

**Top docs**:

1. `medquad_ans_0000004_6-2` (score=0.0328) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
2. `medquad_ans_0000004_7-2` (score=0.0320) - Signs of childhood non-Hodgkin lymphoma include breathing problems and swollen lymph nodes. These and other signs may be caused by childhood non-Hodgkin lymphoma or by other conditions. Check with a doctor if your chi...
3. `medquad_ans_0000004_1-2` (score=0.0318) - Age, gender, and Epstein-Barr infection can affect the risk of adult Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will g...
4. `medquad_ans_0000004_6-1` (score=0.0315) - Key Points - Childhood Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two types of childhood Hodgkin lymphoma. - Epstein-Barr virus infection increases the risk o...
5. `medquad_ans_0000004_6-5` (score=0.0308) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whet...

**Answer**:

Individuals at risk for childhood Hodgkin lymphoma include those who are infected with the Epstein-Barr virus, infected with the human immunodeficiency virus (HIV), have certain diseases of the immune system, have a personal history of mononucleosis ("mono"), or have a parent or sibling with a personal history of Hodgkin lymphoma [medquad_ans_0000004_6-2]. Being exposed to common infections in early childhood may decrease the risk, suggesting that early immune exposure is protective [medquad_ans_0000004_6-2].

**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

### Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Scores**: context_relevance=1.000, groundedness=0.935, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0001795-1` (score=0.0656) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow...
2. `medquad_ans_0001795-3` (score=0.0640) - How might dermatitis herpetiformis be treated? The antibiotic dapsone is extremely effective in treating this condition. Symptomatic improvement may occur in as little as several hours after the first dose. However, d...
3. `medquad_ans_0001795-2` (score=0.0635) - What are the signs and symptoms of Dermatitis herpetiformis ? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatitis herpetiformis . If the information is available, the table be...
4. `medquad_ans_0000095-2__dup3` (score=0.0630) - Dermatitis herpetiformis is caused by the deposit of immunoglobulin A (IgA) in the skin, which triggers further immunologic reactions resulting in lesion formation. DH is an external manifestation of an abnormal immun...
5. `medquad_ans_0000095-1__dup4` (score=0.0615) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation ...

**Answer**:

Dermatitis herpetiformis is a rare, chronic skin disorder characterized by groups of severely itchy blisters and raised skin lesions, more commonly found on the knees, elbows, buttocks, and shoulder blades. It typically begins in adulthood but can also affect children. Other symptoms include fluid-filled sores, red lesions resembling hives, and sensations of itchiness, redness, and burning. The exact cause is unknown, but it is frequently associated with the inability to digest gluten. The condition is often treated with the drug dapsone, which can provide symptomatic relief within several hours of the first dose, though it may cause serious side effects and requires regular monitoring. A strict gluten-free diet is also recommended and may eliminate the need for medications and prevent complications [medquad_ans_0001795-1][medquad_ans_0001795-3].

**Supported citations**: `medquad_ans_0001795-1`, `medquad_ans_0001795-3`
**Unsupported citations**: _none_

### Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Scores**: context_relevance=0.857, groundedness=0.294, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000709-5` (score=0.0807) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Regi...
2. `medquad_ans_0000210-3__dup1` (score=0.0804) - Myotonia is a chronic disorder. Symptoms may improve later in life.
3. `medquad_ans_0000040-4__dup1` (score=0.0772) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to myotonia congenita and also supports additional research through grants to major research institutions across the countr...
4. `medquad_ans_0000210-2__dup2` (score=0.0772) - Treatment for myotonia may include mexiletine, quinine, phenytoin, and other anticonvulsant drugs. Physical therapy and other rehabilitative measures may help muscle function.
5. `medquad_ans_0000040-2__dup1` (score=0.0761) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors ...

**Answer**:

Treatments for myotonia congenita may include mexiletine, quinine, phenytoin, and other anticonvulsant drugs. Physical therapy and other rehabilitative measures may help improve muscle function. Most people with myotonia congenita do not require special treatments, as stiff muscles usually resolve with exercise or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors have had some success with medications such as quinine or anticonvulsant drugs like phenytoin, and physical therapy or other rehabilitative therapies may be used [medquad_ans_0000210-2__dup2, medquad_ans_0000040-2__dup1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000210-2__dup2, medquad_ans_0000040-2__dup1`

### Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Scores**: context_relevance=0.667, groundedness=0.884, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0006449-5` (score=0.0489) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lif...
2. `medquad_ans_0001037-2` (score=0.0484) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
3. `medquad_ans_0001037-5` (score=0.0476) - These resources address the diagnosis or management of Wilson disease: - Gene Review: Gene Review: Wilson Disease - Genetic Testing Registry: Wilson's disease - MedlinePlus Encyclopedia: Wilson's disease - National Hu...
4. `medquad_ans_0006449-1` (score=0.0466) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver,...
5. `medquad_ans_0000133-12` (score=0.0462) - A person cannot prevent Wilson disease; however, people with a family history of Wilson disease, especially those with an affected sibling or parent, should talk with a health care provider about testing. A health car...

**Answer**:

Treatment for Wilson disease involves lifelong management aimed at reducing or controlling copper accumulation in the body. This may include specific medications and dietary modifications. In cases where treatment is ineffective or liver failure develops, a liver transplant may be necessary [medquad_ans_0006449-5]. Early diagnosis and treatment can reduce or even prevent organ damage, especially in individuals with a family history of the disease [medquad_ans_0000133-12]. People with a family history should consult a healthcare provider about testing and may benefit from genetic testing and referral to a geneticist [medquad_ans_0000133-12]. Additional resources for diagnosis and management include GeneReviews, the Genetic Testing Registry, MedlinePlus Encyclopedia, and the National Human Genome Research Institute [medquad_ans_0001037-5].

**Supported citations**: `medquad_ans_0006449-5`, `medquad_ans_0000133-12`, `medquad_ans_0001037-5`
**Unsupported citations**: _none_

### Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.654, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000592-3` (score=0.0656) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are re...
2. `medquad_ans_0000592-2` (score=0.0643) - Lennox-Gastaut syndrome affects an estimated 1 in 50,000 to 1 in 100,000 children. This condition accounts for about 4 percent of all cases of childhood epilepsy. For unknown reasons, it appears to be more common in m...
3. `medquad_ans_0000592-4` (score=0.0633) - Most cases of Lennox-Gastaut syndrome are sporadic, which means they occur in people with no history of the disorder in their family. However, 3 to 30 percent of people with this condition have a family history of som...
4. `medquad_ans_0000179-1__dup1` (score=0.0623) - Lennox-Gastaut syndrome is a severe form of epilepsy. Seizures usually begin before 4 years of age. Seizure types, which vary among patients, include tonic (stiffening of the body, upward deviation of the eyes, dilati...
5. `medquad_ans_0000179-3__dup1` (score=0.0623) - The prognosis for individuals with Lennox-Gastaut syndrome varies. There is no cure for the disorder. Complete recovery, including freedom from seizures and normal development, is very unusual.

**Answer**:

Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component [medquad_ans_0000592-3]. In about one-third of cases, the cause is unknown (cryptogenic), and in those cases, individuals have no history of epilepsy, neurological problems, or delayed development prior to onset [medquad_ans_0000592-1]. While some cases are associated with genetic disorders such as tuberous sclerosis complex, no specific genes have been identified as directly causing Lennox-Gastaut syndrome [medquad_ans_0000592-3]. Additionally, 3 to 30 percent of individuals with the condition have a family history of some type of epilepsy, suggesting a possible genetic predisposition, but no definitive genetic changes are established [medquad_ans_0000592-4].

**Supported citations**: `medquad_ans_0000592-3`, `medquad_ans_0000592-4`
**Unsupported citations**: `medquad_ans_0000592-1`

### Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.890, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000066-2` (score=0.0492) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
2. `medquad_ans_0000066-5` (score=0.0484) - These resources address the diagnosis or management of arginine:glycine amidinotransferase deficiency: - Gene Review: Gene Review: Creatine Deficiency Syndromes - Genetic Testing Registry: Arginine:glycine amidinotran...
3. `medquad_ans_0003472-1` (score=0.0476) - L-arginine:glycine amidinotransferase (AGAT) deficiency is a rare condition that primarily affects the brain. People with AGAT deficiency generally have mild to moderate intellectual disability. Other signs and sympto...
4. `medquad_ans_0003472-2` (score=0.0469) - What are the signs and symptoms of L-arginine:glycine amidinotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for L-arginine:glycine amidinotransferase deficiency....
5. `medquad_ans_0000445-1__dup1` (score=0.0462) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...

**Answer**:

The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families [medquad_ans_0000066-2]. Therefore, the exact number of people affected is not known.

**Supported citations**: `medquad_ans_0000066-2`
**Unsupported citations**: _none_

### Example 231 (`0002399-2`)

**Question**: What are the symptoms of Focal dermal hypoplasia ?

**Scores**: context_relevance=1.000, groundedness=0.543, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000379-2` (score=0.0809) - Focal dermal hypoplasia appears to be a rare condition, although its exact prevalence is unknown.
2. `medquad_ans_0002399-2` (score=0.0807) - What are the signs and symptoms of Focal dermal hypoplasia? Focal dermal hypoplasia is usually evident from birth and primarily affects the skin, skeleton, eyes, and face. The signs and symptoms of vary widely, althou...
3. `medquad_ans_0002399-1` (score=0.0801) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. The skin abnormalities are present from birth and can include streaks of very thin skin (dermal hypoplasia), cut...
4. `medquad_ans_0000379-1` (score=0.0776) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. About 90 percent of affected individuals are female. Males usually have milder signs and symptoms than females. ...
5. `medquad_ans_0003100-2` (score=0.0687) - What are the signs and symptoms of Hypotrichosis-lymphedema-telangiectasia syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypotrichosis-lymphedema-telangiectasia syndrome....

**Answer**:

The symptoms of Focal dermal hypoplasia include skin abnormalities such as streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), telangiectases, and streaks of slightly darker or lighter skin. These can cause pain, itching, irritation, or lead to skin infections. With age, most develop wart-like growths called papillomas around the nostrils, lips, anus, and female genitalia, and possibly in the throat (esophagus or larynx), which can cause problems with swallowing, breathing, or sleeping. Other features include small, ridged fingernails and toenails, sparse or brittle scalp hair, hand and foot abnormalities such as missing fingers or toes (oligodactyly), webbed or fused fingers or toes (syndactyly),

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Scores**: context_relevance=0.818, groundedness=0.890, answer_relevance=0.818

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=0.0328) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
2. `medquad_ans_0000418-1__dup2` (score=0.0323) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...
3. `medquad_ans_0000418-6` (score=0.0317) - Safe and effective prescription drugs are available to treat both Trichinella infection and the symptoms that occur as a result of infection. Treatment should begin as soon as possible; a doctor will make the decision...
4. `medquad_ans_0000418-5__dup1` (score=0.0312) - A diagnosis of trichinellosis is made in patients whose signs and symptoms are compatible with trichinellosis, have a positive laboratory test for Trichinella, and who can recall eating raw or undercooked pork or wild...
5. `medquad_ans_0000418-7` (score=0.0305) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause ...

**Answer**:

People at risk for trichinellosis (also known as trichinosis) include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating undercooked wild game, especially bear meat, puts individuals at risk. Additionally, individuals who consume raw or undercooked meat from animals such as domestic pigs or wild boar, which can harbor Trichinella larvae, are also at risk [medquad_ans_0000418-1\_\_dup2][medquad_ans_0000418-2__dup1].

**Supported citations**: `medquad_ans_0000418-1__dup2`, `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

### Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.050, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001784-2` (score=0.0683) - What are the signs and symptoms of Dentatorubral-pallidoluysian atrophy? The signs and symptoms of DRPLA differ somewhat between affected children and adults. When DRPLA appears before age 20, it most often involves e...
2. `medquad_ans_0000276-1` (score=0.0654) - Dentatorubral-pallidoluysian atrophy, commonly known as DRPLA, is a progressive brain disorder that causes involuntary movements, mental and emotional problems, and a decline in thinking ability. The average age of on...
3. `medquad_ans_0001924-1` (score=0.0650) - Dravet syndrome is a severe form of epilepsy. The condition appears during the first year of life as frequent fever-related (febrile) seizures. As the condition progresses, other types of seizures typically occur, inc...
4. `medquad_ans_0000108-1__dup2` (score=0.0638) - Dravet syndrome, also called severe myoclonic epilepsy of infancy (SMEI), is a severe form of epilepsy. It appears during the first year of life with frequent febrile seizures fever-related seizures that, by definitio...
5. `medquad_ans_0000278-2` (score=0.0633) - The prevalence of Denys-Drash syndrome is unknown; at least 150 affected individuals have been reported in the scientific literature.

**Answer**:

The provided context does not contain any information about the symptoms of Dravet syndrome. Therefore, a response cannot be generated based on the given context. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Scores**: context_relevance=0.889, groundedness=0.963, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0000826-1` (score=0.0650) - Prion disease represents a group of conditions that affect the nervous system in humans and animals. In people, these conditions impair brain function, causing changes in memory, personality, and behavior; a decline i...
2. `medquad_ans_0000826-2` (score=0.0650) - These disorders are very rare. Although the exact prevalence of prion disease is unknown, studies suggest that this group of conditions affects about one person per million worldwide each year. Approximately 350 new c...
3. `medquad_ans_0000826-3` (score=0.0635) - Between 10 and 15 percent of all cases of prion disease are caused by mutations in the PRNP gene. Because they can run in families, these forms of prion disease are classified as familial. Familial prion diseases, whi...
4. `medquad_ans_0000011-13` (score=0.0620) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...
5. `medquad_ans_0000826-5` (score=0.0620) - These resources address the diagnosis or management of prion disease: - Creutzfeldt-Jakob Disease Foundation: Suggestions for Patient Care - Gene Review: Gene Review: Genetic Prion Diseases - Genetic Testing Registry:...

**Answer**:

Between 10 and 15 percent of all cases of prion disease are caused by mutations in the PRNP gene. These mutations result in the production of an abnormally shaped protein called PrPSc from one copy of the gene. This abnormal protein can bind to the normal prion protein (PrPC) and promote its transformation into PrPSc, leading to the accumulation of abnormal protein in the brain and neuronal damage [medquad_ans_0000826-3].

**Supported citations**: `medquad_ans_0000826-3`
**Unsupported citations**: _none_

### Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.984, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0006430-1` (score=0.0656) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information ...
2. `medquad_ans_0003474-2` (score=0.0566) - What are the signs and symptoms of Larsen syndrome? The signs and symptoms of Larsen syndrome vary from person to person, but may include the following: Joint dislocation (especially of the hips, knees, and elbows) Hy...
3. `medquad_ans_0000884-1` (score=0.0529) - Rothmund-Thomson syndrome is a rare condition that affects many parts of the body, especially the skin. People with this condition typically develop redness on the cheeks between ages 3 months and 6 months. Over time ...
4. `medquad_ans_0000420-1__dup1` (score=0.0512) - Apert syndrome is a disorder mainly characterized by craniosynostosis (premature fusion of skull bones, causing abnormalities in the shape of the head and face) and syndactyly (fusion or webbing or fingers and/or toes...
5. `medquad_ans_0001490-2` (score=0.0503) - What are the signs and symptoms of Congenital contractural arachnodactyly? Congenital contractural arachnodactyly represents a broad spectrum of characteristics. The features are quite variable, both within and betwee...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, cleft palate, cleft upper lip, high palate, hydronephrosis, hypoplasia of the radius, hypotelorism, long face, mesomelia, narrow face, oligodactyly (hands), proximal placement of thumb, and proximal radial head dislocation [medquad_ans_0006430-1].

**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

### Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Scores**: context_relevance=1.000, groundedness=0.938, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.0820) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the ta...
2. `medquad_ans_0002061-2` (score=0.0801) - What are the signs and symptoms of Ehlers-Danlos syndrome, progeroid type? Ehlers-Danlos syndrome refers to a group of connective tissue disorders characterized by stretchy or kneadable skin, double jointedness, and d...
3. `medquad_ans_0000903-1` (score=0.0749) - Short stature, hyperextensibility, hernia, ocular depression, Rieger anomaly, and teething delay, commonly known by the acronym SHORT syndrome, is a rare disorder that affects many parts of the body. Most people with ...
4. `medquad_ans_0002058-2` (score=0.0651) - What are the signs and symptoms of Ehlers-Danlos syndrome, kyphoscoliosis type? The signs and symptoms of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type vary but may include: Hyperextensible skin that is fragile an...
5. `medquad_ans_0005030-1` (score=0.0649) - What are the signs and symptoms of Premature ovarian failure, familial? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature ovarian failure, familial. If the information is ava...

**Answer**:

The signs and symptoms of Premature aging Okamoto type include abnormality of blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face. The frequency of these symptoms is estimated based on data from the Human Phenotype Ontology, which often draws from Orphanet, a European rare disease database. Frequencies are typically listed as percentages or fractions, with the fraction indicating the number of patients with the symptom out of the total number examined in a specific study [medquad_ans_0005029-1].

**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

### Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.741, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000979-1` (score=0.0492) - Thrombocytopenia-absent radius (TAR) syndrome is characterized by the absence of a bone called the radius in each forearm. Affected individuals also have a shortage (deficiency) of blood cells involved in clotting (pl...
2. `medquad_ans_0001065-2` (score=0.0484) - The estimated incidence of X-linked thrombocytopenia is between 1 and 10 per million males worldwide; this condition is rarer among females.
3. `medquad_ans_0000329-2` (score=0.0474) - Essential thrombocythemia affects an estimated 1 to 24 per 1 million people worldwide.
4. `medquad_ans_0001065-1` (score=0.0471) - X-linked thrombocytopenia is a bleeding disorder that primarily affects males. This condition is characterized by a blood cell abnormality called thrombocytopenia, which is a shortage in the number of cells involved i...
5. `medquad_ans_0006526-1` (score=0.0462) - What are the signs and symptoms of X-linked thrombocytopenia? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked thrombocytopenia. If the information is available, the table be...

**Answer**:

The provided context does not include the number of people affected by thrombocytopenia-absent radius (TAR) syndrome. Therefore, it is not possible to determine the incidence or prevalence of TAR syndrome from the given information. [medquad_ans_0000979-1] describes the characteristics of the syndrome but does not provide population statistics.

**Supported citations**: `medquad_ans_0000979-1`
**Unsupported citations**: _none_

### Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Scores**: context_relevance=0.455, groundedness=0.690, answer_relevance=0.673

**Top docs**:

1. `medquad_ans_0000972-5` (score=0.0492) - These resources address the diagnosis or management of Tay-Sachs disease: - Gene Review: Gene Review: Hexosaminidase A Deficiency - Genetic Testing Registry: Tay-Sachs disease - MedlinePlus Encyclopedia: Tay-Sachs Dis...
2. `medquad_ans_0000871-1__dup2` (score=0.0484) - Tay-Sachs disease is a rare, inherited disorder. It causes too much of a fatty substance to build up in the brain. This buildup destroys nerve cells, causing mental and physical problems. Infants with Tay-Sachs diseas...
3. `medquad_ans_0006017-5` (score=0.0476) - How might children with Tay-Sachs disease be treated? Although several attempts have been made at purified enzyme replacement therapy for children with Tay-Sachs disease, none has been successful. Cellular infusions a...
4. `medquad_ans_0000265-2__dup1` (score=0.0469) - Presently there is no specific treatment for Tay-Sachs disease. Anticonvulsant medicine may initially control seizures. Other supportive treatment includes proper nutrition and hydration and techniques to keep the air...
5. `medquad_ans_0000265-1__dup1` (score=0.0462) - Tay-Sachs disease is a inherited metabolic disease caused by the harmful buildup of lipids (fatty materials such as oils and acids) in various cells and tissues in the body. It is part of a group of genetic disorders ...

**Answer**:

The provided context does not mention any current research or clinical trials being conducted for Tay-Sachs disease. [medquad_ans_0000972-5], [medquad_ans_0000871-1__dup2], and [medquad_ans_0006017-5] discuss diagnosis, management, and supportive treatments but do not reference ongoing research or clinical trials. Therefore, based on the given information, no such research or clinical trials are described.

**Supported citations**: `medquad_ans_0000972-5`, `medquad_ans_0000871-1__dup2`, `medquad_ans_0006017-5`
**Unsupported citations**: _none_

### Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000008-5` (score=0.0656) - These resources address the diagnosis or management of achromatopsia: - Gene Review: Gene Review: Achromatopsia - Genetic Testing Registry: Achromatopsia - MedlinePlus Encyclopedia: Color Vision Test These resources f...
2. `medquad_ans_0000008-1` (score=0.0630) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achro...
3. `medquad_ans_0000008-2` (score=0.0630) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on o...
4. `medquad_ans_0000050-1__dup1` (score=0.0625) - What are the signs and symptoms of Achromatopsia 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Achromatopsia 3. If the information is available, the table below includes how oft...
5. `medquad_ans_0000994-5` (score=0.0621) - These resources address the diagnosis or management of triple A syndrome: - Genetic Testing Registry: Glucocorticoid deficiency with achalasia - MedlinePlus Encyclopedia: Achalasia - MedlinePlus Encyclopedia: Anisocor...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. The condition also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals may have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia is different from common forms of color vision deficiency (color blindness), in which people can perceive color but have difficulty distinguishing between certain colors, such as red and green [medquad_ans_00000

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Scores**: context_relevance=0.857, groundedness=0.769, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000544-1__dup1` (score=0.0809) - Knee replacement is surgery for people with severe knee damage. Knee replacement can relieve pain and allow you to be more active. Your doctor may recommend it if you have knee pain and medicine and other treatments a...
2. `medquad_ans_0000039-11` (score=0.0799) - Treatments your doctor will likely recommend before knee replacement include - exercises to strengthen the muscles around the knee and improve flexibility - weight loss, if needed, to reduce the load the knee must bea...
3. `medquad_ans_0000039-26` (score=0.0794) - You can learn more about knee replacement from the following resources. National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) Information Clearinghouse National Institutes of Health 1 AMS Circl...
4. `medquad_ans_0000039-9` (score=0.0784) - The main reason to have knee replacement surgery is to ease pain and disability caused by arthritis or other joint problems, while preserving movement. Less commonly, it is used to correct some kinds of knee deformity.
5. `medquad_ans_0000061-13` (score=0.0765) - Medication, exercise, and, in some cases, surgery are common treatments for this disease. Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation...

**Answer**:

The provided context does not list treatments for knee replacement surgery itself. Instead, it describes treatments recommended _before_ knee replacement, such as exercises to strengthen muscles and improve flexibility, weight loss, walking aids, shoe inserts, and medicines to relieve pain [medquad_ans_0000039-11].

Knee replacement is a surgical procedure, not a condition treated with the same interventions. Therefore, there are no treatments for knee replacement surgery as such—rather, it is a treatment option for severe knee damage when other methods fail [medquad_ans_0000544-1__dup1].

Answer: The context does not provide treatments for knee replacement surgery; it outlines non-surgical treatments recommended before knee replacement. [medquad_ans_0000039-11]

**Supported citations**: `medquad_ans_0000039-11`, `medquad_ans_0000544-1__dup1`
**Unsupported citations**: _none_

### Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.774, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000608-1__dup1` (score=0.0807) - Metabolic syndrome is a group of conditions that put you at risk for heart disease and diabetes. These conditions are - High blood pressure - High blood glucose, or blood sugar, levels - High levels of triglycerides, ...
2. `medquad_ans_0000044-11` (score=0.0804) - Metabolic syndrome, also called insulin resistance syndrome, is a group of traits and medical conditions linked to overweight and obesity that puts people at risk for both CVD and type 2 diabetes. Metabolic syndrome i...
3. `medquad_ans_0000090-5__dup1` (score=0.0793) - Metabolic syndrome is a group of risk factors that raises your risk for heart disease and other health problems, such as diabetes and stroke. These risk factors can increase your risk for health problems even if they'...
4. `medquad_ans_0000032-6__dup1` (score=0.0774) - Metabolic syndrome is a grouping of traits and medical conditions that puts people at risk for both heart disease and type 2 diabetes. It is defined by the National Cholesterol Education Program as having any three of...
5. `medquad_ans_0000090-3__dup2` (score=0.0767) - Metabolic syndrome has several causes that act together. You can control some of the causes, such as overweight and obesity, an inactive lifestyle, and insulin resistance. You can't control other factors that may play...

**Answer**:

The provided context does not describe symptoms of metabolic syndrome. Instead, it lists the conditions and risk factors associated with metabolic syndrome, such as high blood pressure, high blood glucose, high triglycerides, low HDL cholesterol, excess fat around the waist, and obesity. However, these are not presented as symptoms but rather as diagnostic criteria or risk factors. Therefore, based on the given information, there are no explicit symptoms listed.

Answer: The context does not provide symptoms of metabolic syndrome. [medquad_ans_0000608-1__dup1] [medquad_ans_0000044-11]

**Supported citations**: `medquad_ans_0000608-1__dup1`, `medquad_ans_0000044-11`
**Unsupported citations**: _none_

### Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Scores**: context_relevance=1.000, groundedness=0.957, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005211-1` (score=0.0656) - What are the signs and symptoms of Pyknoachondrogenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyknoachondrogenesis. If the information is available, the table below includ...
2. `medquad_ans_0004625-2` (score=0.0631) - What are the signs and symptoms of Osteomesopyknosis? Osteomesopyknosis may cause chronic, low-grade back pain in the thoracic (middle) and lumbar (lower) regions. It is considered a mild form of osteosclerosis and is...
3. `medquad_ans_0004229-1` (score=0.0615) - What are the signs and symptoms of Mutiple parosteal osteochondromatous proliferations? The Human Phenotype Ontology provides the following list of signs and symptoms for Mutiple parosteal osteochondromatous prolifera...
4. `medquad_ans_0000986-2__dup1` (score=0.0603) - What are the signs and symptoms of Catamenial pneumothorax? The Human Phenotype Ontology provides the following list of signs and symptoms for Catamenial pneumothorax. If the information is available, the table below ...
5. `medquad_ans_0005210-2` (score=0.0526) - What are the signs and symptoms of Pycnodysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pycnodysostosis. If the information is available, the table below includes how oft...

**Answer**:

The symptoms of Pyknoachondrogenesis include abnormal vertebral ossification, abnormality of pelvic girdle bone morphology, abnormality of the mouth, abnormality of the sacrum, depressed nasal ridge, enlarged thorax, increased bone mineral density, low-set, posteriorly rotated ears, macrocephaly, micromelia, palpebral edema, premature birth, short stature, short thorax, thickened nuchal skin fold, and stillbirth. These symptoms are reported in approximately 90% of patients with this condition [medquad_ans_0005211-1].

**Supported citations**: `medquad_ans_0005211-1`
**Unsupported citations**: _none_

### Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.964, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000647-1` (score=0.0799) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the sub...
2. `medquad_ans_0000647-2` (score=0.0794) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigra...
3. `medquad_ans_0003566-1` (score=0.0794) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to def...
4. `medquad_ans_0003965-2` (score=0.0788) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the ta...
5. `medquad_ans_0000555-1__dup1` (score=0.0760) - The leukodystrophies are rare diseases that affect the cells of the brain. Specifically, the diseases affect the myelin sheath, the material that surrounds and protects nerve cells. Damage to this sheath slows down or...

**Answer**:

The symptoms of metachromatic leukodystrophy include progressive deterioration of intellectual functions and motor skills, such as the ability to walk, loss of sensation in the extremities (peripheral neuropathy), incontinence, seizures, paralysis, an inability to speak, blindness, and hearing loss. Eventually, individuals lose awareness of their surroundings and become unresponsive. In the late infantile form, children lose speech, become weak, and develop gait disturbance; muscle tone first decreases and then increases to rigidity. In the juvenile form, initial signs may include behavioral problems and difficulty with schoolwork, with slower progression and survival of about 20 years after diagnosis. In the adult form, symptoms often begin during teenage years or later and may include behavioral problems such as alcoholism, drug abuse, or difficulties at school or work, along with psychiatric symptoms like delusions or hallucinations [medquad_ans_0000647-1].

**Supported citations**: `medquad_ans_0000647-1`
**Unsupported citations**: _none_

### Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Scores**: context_relevance=0.600, groundedness=0.000, answer_relevance=0.720

**Top docs**:

1. `medquad_ans_0000673-4` (score=0.0325) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
2. `medquad_ans_0000362-4` (score=0.0320) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000410-1__dup1` (score=0.0315) - Granulomatosis with polyangiitis (GPA), previously known as Wegener's granulomatosis, is a rare disease. It is a type of vasculitis, or inflammation of the blood vessels. The inflammation limits the flow of blood to i...
4. `medquad_ans_0000095-4` (score=0.0315) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
5. `medquad_ans_0000585-4` (score=0.0310) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...

**Answer**:

The provided context does not mention chronic granulomatous disease or its inheritance pattern. Therefore, it is not possible to determine based on the given information whether chronic granulomatous disease is inherited.

Answer: Insufficient context [medquad_ans_0000673-4, medquad_ans_0000362-4, medquad_ans_0000410-1__dup1, medquad_ans_0000095-4, medquad_ans_0000585-4].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000673-4, medquad_ans_0000362-4, medquad_ans_0000410-1__dup1, medquad_ans_0000095-4, medquad_ans_0000585-4`

### Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.272, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000061-4__dup1` (score=0.0656) - Our understanding of central cord syndrome has increased greatly in recent decades as a result of research funded conducted by the National Institute of Neurological Disorders and Stroke (NINDS). Much of this research...
2. `medquad_ans_0000061-3__dup1` (score=0.0645) - The prognosis for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict he likeliho...
3. `medquad_ans_0000061-2__dup1` (score=0.0630) - There is no cure for central cord syndrome although some people recover near-normal function. There is no standard course of treatment, although drug therapy, surgery, and rest are often part of the program. Magnetic ...
4. `medquad_ans_0000061-1__dup1` (score=0.0630) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to a...
5. `medquad_ans_0000164-2` (score=0.0604) - Central core disease is probably an uncommon condition, although its incidence is unknown.

**Answer**:

The outlook for central cord syndrome is generally favorable, with most people experiencing some recovery of neurological function, particularly after trauma. Many individuals recover substantial function following their initial injury, and the ability to walk is regained in most cases, although some impairment may remain. Recovery typically begins in the legs, followed by the bladder, and hand function recovers last, if at all. Improvement is generally better in younger patients compared to those over the age of 50. Early medical intervention is associated with good outcomes, and evaluation of abnormal signals on MRI images can help predict the likelihood of natural neurological recovery. While there is no cure, early surgery may improve recovery chances, and surgery is beneficial in cases of persistent spinal cord compression and ongoing neurological deterioration [medquad_ans_0000061-3__dup1, medquad_ans_0000061-2__dup1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000061-3__dup1, medquad_ans_0000061-2__dup1`

### Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Scores**: context_relevance=0.600, groundedness=0.939, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000094-13` (score=0.0814) - - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no ...
2. `medquad_ans_0000094-1__dup6` (score=0.0812) - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no ap...
3. `medquad_ans_0000094-4__dup3` (score=0.0794) - Cyclic vomiting syndrome is more common in children than adults, although reports of the syndrome in adults have increased in recent years.1 Usually, children are about 5 years old when diagnosed with cyclic vomiting ...
4. `medquad_ans_0000094-6__dup1` (score=0.0769) - The main symptoms of cyclic vomiting syndrome are severe nausea and sudden vomiting lasting hours to days. A person may also experience one or more of the following symptoms: - retching, or making an attempt to vomit ...
5. `medquad_ans_0000094-3__dup3` (score=0.0767) - The cause of cyclic vomiting syndrome is unknown. However, some experts believe that some possible problems with bodily functions may contribute to the cause, such as the following: - gastrointestinal motilitythe way ...

**Answer**:

The cause of cyclic vomiting syndrome is unknown [medquad_ans_0000094-13].

**Supported citations**: `medquad_ans_0000094-13`
**Unsupported citations**: _none_

### Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Scores**: context_relevance=0.625, groundedness=0.917, answer_relevance=0.625

**Top docs**:

1. `medquad_ans_0000180-1__dup2` (score=0.0328) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not alwa...
2. `medquad_ans_0000603-1__dup1` (score=0.0323) - Summary : Mental health includes our emotional, psychological, and social well-being. It affects how we think, feel and act as we cope with life. It also helps determine how we handle stress, relate to others, and mak...
3. `medquad_ans_0000776-5` (score=0.0315) - These resources address the diagnosis or management of Partington syndrome: - American Academy of Child and Adolescent Psychiatry: Services in School for Children with Special Needs - American Academy of Pediatrics: W...
4. `medquad_ans_0000189-1__dup2` (score=0.0304) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...
5. `medquad_ans_0000602-1__dup1` (score=0.0303) - Mental disorders include a wide range of problems, including - Anxiety disorders, including panic disorder, obsessive-compulsive disorder, post-traumatic stress disorder, and phobias - Bipolar disorder - Depression - ...

**Answer**:

Yes, there is information about child mental health. Early recognition and treatment of mental illnesses in children are important because once mental illness develops, it becomes a regular part of the child's behavior and is more difficult to treat [medquad_ans_0000180-1__dup2]. Warning signs that may indicate a more serious problem include problems in multiple settings (at school, at home, with peers), changes in appetite or sleep, social withdrawal or fear of things previously not feared, returning to younger behaviors like bedwetting, signs of being upset such as sadness or tearfulness, self-destructive behaviors like head-banging or frequent injuries, and repeated thoughts of death [medquad_ans_0000180-1__dup2]. Diagnosis involves evaluating the child's signs and symptoms, medical history, and family history [medquad_ans_0000180-1__dup2]. Treatments

**Supported citations**: `medquad_ans_0000180-1__dup2`
**Unsupported citations**: _none_

### Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.152, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000095-1__dup4` (score=0.0693) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation ...
2. `medquad_ans_0000011-4__dup3` (score=0.0640) - Most people were treated with pituitary hGH because their pituitary glands did not make enough of their own GH. Some of these people also had problems making other pituitary hormones. One of these hormones tells the a...
3. `medquad_ans_0000003-2__dup3` (score=0.0489) - Signs and symptoms of Cushing's syndrome vary, but most people with the disorder have upper body obesity, a rounded face, increased fat around the neck, and relatively slender arms and legs. Children tend to be obese ...
4. `medquad_ans_0000673-2__dup1` (score=0.0487) - What are the signs and symptoms of Behcet's disease? Symptoms of Behcet's disease include recurrent ulcers in the mouth (resembling canker sores) and on the genitals, and eye inflammation (uveitis). The disorder may a...
5. `medquad_ans_0000122-5__dup1` (score=0.0474) - The most common signs and symptoms of ulcerative colitis are diarrhea with blood or pus and abdominal discomfort. Other signs and symptoms include - an urgent need to have a bowel movement - feeling tired - nausea or ...

**Answer**:

The provided context does not contain any information about the symptoms of Cowden syndrome. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.809, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000395-5` (score=0.0817) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
2. `medquad_ans_0000395-2` (score=0.0809) - Fukuyama congenital muscular dystrophy is seen almost exclusively in Japan, where it is the second most common form of childhood muscular dystrophy (after Duchenne muscular dystrophy). Fukuyama congenital muscular dys...
3. `medquad_ans_0000395-1` (score=0.0794) - Fukuyama congenital muscular dystrophy is an inherited condition that predominantly affects the muscles, brain, and eyes. Congenital muscular dystrophies are a group of genetic conditions that cause muscle weakness an...
4. `medquad_ans_0004225-1` (score=0.0761) - The congenital muscle dystrophies are currently classified according to the genetic defects. Historically, congenital muscular dystrophies were classified in two broad groups: Classic CMD (which included the Merosin-d...
5. `medquad_ans_0000576-5` (score=0.0752) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...

**Answer**:

The provided context does not specify treatments for Fukuyama congenital muscular dystrophy. While it mentions resources related to diagnosis and management, such as genetic counseling, drug therapy, surgery and rehabilitation, palliative care, and diagnostic tests, it does not detail specific treatments for this condition [medquad_ans_0000395-5]. Therefore, based on the given information, the exact treatments are not available.

**Supported citations**: `medquad_ans_0000395-5`
**Unsupported citations**: _none_

### Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000004_7-2` (score=0.0492) - Signs of childhood non-Hodgkin lymphoma include breathing problems and swollen lymph nodes. These and other signs may be caused by childhood non-Hodgkin lymphoma or by other conditions. Check with a doctor if your chi...
2. `medquad_ans_0000004_6-1` (score=0.0484) - Key Points - Childhood Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two types of childhood Hodgkin lymphoma. - Epstein-Barr virus infection increases the risk o...
3. `medquad_ans_0000483-1__dup1` (score=0.0471) - Hodgkin disease is a type of lymphoma. Lymphoma is a cancer of a part of the immune system called the lymph system. The first sign of Hodgkin disease is often an enlarged lymph node. The disease can spread to nearby l...
4. `medquad_ans_0000004_6-3` (score=0.0471) - Signs of childhood Hodgkin lymphoma include swollen lymph nodes, fever, night sweats, and weight loss. These and other signs and symptoms may be caused by childhood Hodgkin lymphoma or by other conditions. Check with ...
5. `medquad_ans_0000004_7-1` (score=0.0464) - Key Points - Childhood non-Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - The main types of lymphoma are Hodgkin lymphoma and non-Hodgkin lymphoma. - There are three major ...

**Answer**:

Tests that examine the lymph system are used to detect (find) and diagnose childhood Hodgkin lymphoma [medquad_ans_0000004_6-1].

**Supported citations**: `medquad_ans_0000004_6-1`
**Unsupported citations**: _none_

### Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.098, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001083-4` (score=0.0255) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000045-4` (score=0.0251) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000370-4` (score=0.0241) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000858-4` (score=0.0230) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000122-4` (score=0.0228) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention KBG syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether KBG syndrome is inherited.

Answer: Insufficient context to determine if KBG syndrome is inherited.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Scores**: context_relevance=1.000, groundedness=0.962, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000029-2` (score=0.0492) - Aicardi-Goutieres syndrome is a rare disorder. Its exact prevalence is unknown.
2. `medquad_ans_0000197-1__dup1` (score=0.0481) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
3. `medquad_ans_0000029-5` (score=0.0474) - These resources address the diagnosis or management of Aicardi-Goutieres syndrome: - Gene Review: Gene Review: Aicardi-Goutieres Syndrome - Genetic Testing Registry: Aicardi Goutieres syndrome - Genetic Testing Regist...
4. `medquad_ans_0000196-1__dup1` (score=0.0474) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
5. `medquad_ans_0000194-2__dup1` (score=0.0459) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...

**Answer**:

Aicardi-Goutieres syndrome is a rare disorder that primarily affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy), which usually results in severe intellectual and physical disability. Additional symptoms may include epilepsy, painful and itchy skin lesions (chilblains), vision problems, and joint stiffness. Symptoms typically progress over several months before the disease course stabilizes. There are six different types of Aicardi-Goutieres syndrome, distinguished by the gene responsible: TREX1, RNASEH2A, RNASEH2B, RNASEH2C, SAMHD1, and ADAR genes. Most cases are inherited in an autosomal recessive pattern, although rare autosomal dominant cases have been reported. Treatment is symptomatic and supportive [medquad_ans_0000197-1__dup1]. The exact prevalence is unknown [medquad

**Supported citations**: `medquad_ans_0000197-1__dup1`
**Unsupported citations**: _none_

### Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Scores**: context_relevance=1.000, groundedness=0.933, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000026_1-7` (score=0.0492) - Key Points - There are different types of treatment for patients with renal cell cancer. - Five types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Biologic therapy - Targeted therapy ...
2. `medquad_ans_0000026_2-7` (score=0.0484) - Key Points - There are different types of treatment for patients with transitional cell cancer of the renal pelvis and ureter. - One type of standard treatment is used: - Surgery - New types of treatment are being tes...
3. `medquad_ans_0000026_3-7` (score=0.0474) - Key Points - There are different types of treatment for patients with Wilms tumor and other childhood kidney tumors. - Children with Wilms tumor or other childhood kidney tumors should have their treatment planned by ...
4. `medquad_ans_0001354-2` (score=0.0469) - What treatments for metastatic clear cell renal cell carcinoma are available in North America? There are several treatments for metastatic clear cell renal cell carcinoma available in North America. IL-2 and sunitinib...
5. `medquad_ans_0000026_1-1` (score=0.0464) - Key Points - Renal cell cancer is a disease in which malignant (cancer) cells form in tubules of the kidney. - Smoking and misuse of certain pain medicines can affect the risk of renal cell cancer. - Signs of renal ce...

**Answer**:

Research involving clinical trials is being conducted for renal cell cancer to improve current treatments or obtain information on new treatments. These clinical trials are designed to test new therapies and may be open to patients before, during, or after starting their cancer treatment. When clinical trials demonstrate that a new treatment is more effective than the standard treatment, it may become the new standard of care [medquad_ans_0000026_1-7].

**Supported citations**: `medquad_ans_0000026_1-7`
**Unsupported citations**: _none_

### Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Scores**: context_relevance=0.857, groundedness=0.469, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000190-6` (score=0.0492) - Treatment for primary biliary cirrhosis depends on how early a health care provider diagnoses the disease and whether complications are present. In the early stages of primary biliary cirrhosis, treatment can slow the...
2. `medquad_ans_0000190-9` (score=0.0481) - - Primary biliary cirrhosis is a chronic disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. - When chronic inflammation damages the bile ducts, bile and toxi...
3. `medquad_ans_0000183-10` (score=0.0474) - - Biliary atresia is a life-threatening condition in infants in which the bile ducts inside or outside the liver do not have normal openings. - The first symptom of biliary atresia is jaundicewhen the skin and whites ...
4. `medquad_ans_0000183-1__dup4` (score=0.0458) - Biliary atresia is a life-threatening condition in infants in which the bile ducts inside or outside the liver do not have normal openings. Bile ducts in the liver, also called hepatic ducts, are tubes that carry bile...
5. `medquad_ans_0000190-2__dup2` (score=0.0453) - The causes of primary biliary cirrhosis are unknown. Most research suggests it is an autoimmune disease. The immune system protects people from infection by identifying and destroying bacteria, viruses, and other pote...

**Answer**:

Treatment for primary biliary cirrhosis depends on the stage of the disease and the presence of complications. In the early stages, treatment can slow the progression of liver damage to cirrhosis. The primary medication prescribed is ursodiol (Actigall, Urso), a nontoxic bile acid that replaces more toxic bile acids produced by the liver, reducing levels of bilirubin and liver enzymes in the blood. Early treatment with ursodiol reduces the likelihood of needing a liver transplant and improves survival, although it does not cure the disease. Treatment benefits are greatest when started early, but ursodiol can still slow disease progression even when used later.

Patients should avoid alcohol and illegal substances, as these can worsen liver damage. They should also avoid complementary and alternative medications, such as herbs, and should consult a healthcare provider before taking any new prescription, over-the-counter medications, or vitamins, as many can affect liver

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Scores**: context_relevance=0.750, groundedness=0.975, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=0.0656) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People ...
2. `medquad_ans_0000378-2` (score=0.0632) - Floating-Harbor syndrome is a rare disorder; only about 50 cases have been reported in the medical literature.
3. `medquad_ans_0000967-1__dup1` (score=0.0628) - Summary : Playing in the water - whether swimming, boating or diving - can be fun. It can also be dangerous, especially for children. Being safe can help prevent injuries and drowning. To stay safe in the water - Avoi...
4. `medquad_ans_0000378-4` (score=0.0610) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of Floating-Harbor syndrome result from new mutations...
5. `medquad_ans_0000378-5` (score=0.0554) - These resources address the diagnosis or management of Floating-Harbor syndrome: - Gene Review: Gene Review: Floating-Harbor Syndrome - Genetic Testing Registry: Floating-Harbor syndrome - KidsHealth: Bone Age Study T...

**Answer**:

Drowning is when people get too much water in their lungs, which can happen quickly and silently, even with as little as an inch or two of water. Babies can drown in a sink or bathtub, and preschoolers are most likely to drown in a swimming pool. People with seizure disorders are also at risk in the water [medquad_ans_0000299-1__dup2].

**Supported citations**: `medquad_ans_0000299-1__dup2`
**Unsupported citations**: _none_

### Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Scores**: context_relevance=1.000, groundedness=0.529, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002058-1` (score=0.0656) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragi...
2. `medquad_ans_0002058-5` (score=0.0645) - How is Ehlers-Danlos syndrome, kyphoscoliosis type diagnosed? A diagnosis of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is typically based on the presence of characteristic signs and symptoms. The following tes...
3. `medquad_ans_0002058-3` (score=0.0628) - What causes Ehlers-Danlos syndrome, kyphoscoliosis type? Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is caused by changes (mutations) in the PLOD1 gene. This gene encodes an enzyme that helps process molecules w...
4. `medquad_ans_0000313-2` (score=0.0618) - Although it is difficult to estimate the overall frequency of Ehlers-Danlos syndrome, the combined prevalence of all types of this condition may be about 1 in 5,000 individuals worldwide. The hypermobility and classic...
5. `medquad_ans_0000313-1` (score=0.0611) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and sympt...

**Answer**:

Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder caused by defects in collagen due to mutations in the PLOD1 gene. It is inherited in an autosomal recessive manner. Common signs and symptoms include hyperextensible skin that is fragile and bruises easily, joint hypermobility, severe hypotonia at birth, progressive kyphoscoliosis (kyphosis and scoliosis), and fragility of the sclera. The diagnosis is typically based on characteristic signs and symptoms, and may be confirmed with urine tests, skin biopsy to detect deficiencies in enzymes important for collagen formation, or genetic testing for mutations in the PLOD1 gene. This form of EDS is very rare, with about 60 cases reported worldwide [medquad_ans_0002058-1, medquad_ans_0002058-5,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Scores**: context_relevance=0.875, groundedness=0.976, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000288-2` (score=0.0804) - Severe dihydropyrimidine dehydrogenase deficiency, with its early-onset neurological symptoms, is a rare disorder. Its prevalence is unknown. However, between 2 and 8 percent of the general population may be vulnerabl...
2. `medquad_ans_0000287-2` (score=0.0799) - Dihydropyrimidinase deficiency is thought to be a rare disorder. Only a few dozen affected individuals have been described in the medical literature.
3. `medquad_ans_0001862-1` (score=0.0789) - Dihydropyrimidine dehydrogenase (DPD) deficiency is a condition in which the body cannot break down the nucleotides thymine and uracil. DPD deficiency can have a wide range of severity; some individuals may have vario...
4. `medquad_ans_0001860-1` (score=0.0772) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...
5. `medquad_ans_0001098-2` (score=0.0763) - 5-alpha reductase deficiency is a rare condition; the exact incidence is unknown. Large families with affected members have been found in several countries, including the Dominican Republic, Papua New Guinea, Turkey, ...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include psychomotor delay, low muscle tone (hypotonia), seizures, abnormal movements, too much salivation, and swallowing difficulties. [medquad_ans_0001860-1]

**Supported citations**: `medquad_ans_0001860-1`
**Unsupported citations**: _none_

### Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Scores**: context_relevance=0.286, groundedness=0.921, answer_relevance=0.171

**Top docs**:

1. `medquad_ans_0000669-2` (score=0.0492) - The prevalence of monilethrix is unknown.
2. `medquad_ans_0000770-2` (score=0.0476) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.
3. `medquad_ans_0000484-2` (score=0.0465) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
4. `medquad_ans_0000409-2` (score=0.0459) - Gitelman syndrome affects an estimated 1 in 40,000 people worldwide.
5. `medquad_ans_0000359-2` (score=0.0444) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.

**Answer**:

The prevalence of monilethrix is unknown [medquad_ans_0000669-2].

**Supported citations**: `medquad_ans_0000669-2`
**Unsupported citations**: _none_

### Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Scores**: context_relevance=1.000, groundedness=0.511, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.0656) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...
2. `medquad_ans_0002733-2` (score=0.0645) - What are the signs and symptoms of Hartnup disease? The signs and symptoms of Hartnup disease may vary and include the following: Skin findings: sensitivity to sunlight Neurologic symptoms: ataxia, spasticity, headach...
3. `medquad_ans_0000870-2` (score=0.0605) - RAMSVPS is a rare disorder. Only a small number of affected individuals and families, all from Saudi Arabia, have been described in the medical literature.
4. `medquad_ans_0000372-2` (score=0.0605) - Feingold syndrome appears to be a rare condition, although its exact prevalence is unknown.
5. `medquad_ans_0001081-1__dup1` (score=0.0561) - Chanarin-Dorfman syndrome is an inherited condition in which fats are stored abnormally in the body. Affected individuals cannot break down certain fats called triglycerides. These fats accumulate in organs and tissue...

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system, making it a type of aminoaciduria. It is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. The condition may be diagnosed through newborn screening tests. Most individuals are asymptomatic, but those who do show symptoms typically experience onset between ages 3 and 9, although it can occasionally present in adulthood. Signs and symptoms include skin photosensitivity, neurologic findings (such as ataxia, spasticity, headaches, and hypotonia), psychiatric symptoms (like anxiety, emotional instability, and mood changes), and ocular findings (such as double vision, nystagmus, strabismus, and photophobia). Symptoms may be triggered by sunlight exposure, fever, drugs, or emotional or physical stress and often last 1–4 weeks before spontaneous

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Scores**: context_relevance=0.583, groundedness=0.739, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000427-5` (score=0.0653) - These resources address the diagnosis or management of glycogen storage disease type VI: - Gene Review: Gene Review: Glycogen Storage Disease Type VI - Genetic Testing Registry: Glycogen storage disease, type VI These...
2. `medquad_ans_0000423-5` (score=0.0643) - These resources address the diagnosis or management of glycogen storage disease type III: - Gene Review: Gene Review: Glycogen Storage Disease Type III - Genetic Testing Registry: Glycogen storage disease type III The...
3. `medquad_ans_0002622-1` (score=0.0632) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepa...
4. `medquad_ans_0000428-5` (score=0.0620) - These resources address the diagnosis or management of glycogen storage disease type VII: - Genetic Testing Registry: Glycogen storage disease, type VII - The Swedish Information Centre for Rare Diseases These resourc...
5. `medquad_ans_0000425-5` (score=0.0604) - These resources address the diagnosis or management of glycogen storage disease type IX: - Gene Review: Gene Review: Phosphorylase Kinase Deficiency - Genetic Testing Registry: Glycogen storage disease IXb - Genetic T...

**Answer**:

The provided context does not specify the genetic changes related to glycogen storage disease type VII. [medquad_ans_0000428-5] mentions resources for diagnosis and management of glycogen storage disease type VII, including the Genetic Testing Registry, but does not detail the underlying genetic mutations or changes. Therefore, based on the given information, the genetic changes are not available.

**Supported citations**: `medquad_ans_0000428-5`
**Unsupported citations**: _none_

### Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.923, answer_relevance=0.711

**Top docs**:

1. `medquad_ans_0000977-2` (score=0.0820) - Thiamine-responsive megaloblastic anemia syndrome has been reported in approximately 30 families worldwide. Its prevalence is unknown.
2. `medquad_ans_0006061-1` (score=0.0806) - Thiamine-responsive megaloblastic anemia syndrome is a very rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Affected individuals begin to show symptoms of this...
3. `medquad_ans_0000977-5` (score=0.0794) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megal...
4. `medquad_ans_0000977-1` (score=0.0781) - Thiamine-responsive megaloblastic anemia syndrome is a rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Megaloblastic anemia occurs when a person has a low numb...
5. `medquad_ans_0006061-2` (score=0.0769) - What are the signs and symptoms of Thiamine responsive megaloblastic anemia syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thiamine responsive megaloblastic anemia syndrom...

**Answer**:

The treatment for thiamine-responsive megaloblastic anemia syndrome is high doses of vitamin B1 (thiamine), which can treat the anemia associated with the condition [medquad_ans_0006061-1]. However, it remains unclear whether thiamine treatment can improve hearing loss or prevent its development [medquad_ans_0000977-1]. Individuals with the syndrome also usually require insulin to manage their diabetes [medquad_ans_0000977-1].

**Supported citations**: `medquad_ans_0006061-1`, `medquad_ans_0000977-1`
**Unsupported citations**: _none_

### Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.483, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000804-5` (score=0.0489) - These resources address the diagnosis or management of polycystic kidney disease: - Gene Review: Gene Review: Polycystic Kidney Disease, Autosomal Dominant - Gene Review: Gene Review: Polycystic Kidney Disease, Autoso...
2. `medquad_ans_0000587-4__dup1` (score=0.0487) - Is there a cure or treatment for autosomal recessive polycystic kidney disease? Although a cure or treatment for the underlying genetic cause of autosomal recessive polycystic kidney disease does not exist, advancemen...
3. `medquad_ans_0000804-1` (score=0.0471) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the...
4. `medquad_ans_0004939-1` (score=0.0469) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the ...
5. `medquad_ans_0004939-2` (score=0.0462) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their abilit...

**Answer**:

Management of polycystic kidney disease involves a range of supportive and symptomatic interventions. Medical management includes treatment of high blood pressure with medication, dialysis when kidney failure occurs, and kidney transplantation as an option for end-stage renal disease. For infants with severe kidney enlargement, dialysis may be required during the first days of life due to oliguria or anuria. Low sodium levels (hyponatremia) are treated with diuresis and/or sodium supplementation. Mechanical ventilation may be used to address lung underdevelopment and breathing issues caused by enlarged kidneys. When kidneys are severely enlarged, nephrectomy (removal of one or both kidneys) may be performed. Urinary tract infections are managed with antibiotics. Growth failure is addressed with gastrostomy tubes and growth hormone therapy. In cases of liver involvement, a shunt may be required to manage progressive hypertension, and liver transplantation may be considered. Genetic counseling, palliative care, and surgical interventions

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Scores**: context_relevance=0.818, groundedness=0.938, answer_relevance=0.745

**Top docs**:

1. `medquad_ans_0000061-5__dup2` (score=0.0492) - Scientists are making rapid progress in understanding the complexities of rheumatoid arthritis. They are learning more about how and why it develops and why some people have more severe symptoms than others. Research ...
2. `medquad_ans_0000090-4__dup1` (score=0.0471) - Much of the on-going research on RMDs is aimed at prevention and rehabilitation. The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) funds research on RMDs.
3. `medquad_ans_0000004-32` (score=0.0460) - People with Alzheimer's disease, those with mild cognitive impairment, those with a family history of Alzheimers, and healthy people with no memory problems who want to help scientists test new treatments may be able ...
4. `medquad_ans_0000004-31` (score=0.0458) - Clinical research is medical research involving people. It includes clinical studies, which observe and gather information about large groups of people. It also includes clinical trials, which test a medicine, therapy...
5. `medquad_ans_0000061-4__dup3` (score=0.0458) - Most Symptoms Are Treatable Doctors use a variety of approaches to treat rheumatoid arthritis. The goals of treatment are to help relieve pain, reduce swelling, slow down or help prevent joint damage, increase the abi...

**Answer**:

Research on rheumatoid arthritis is focused on understanding how and why the disease develops, particularly identifying possible triggers such as infectious agents like viruses and bacteria that may contribute to the disease in individuals with a genetic predisposition. Scientists are investigating the complex relationships between the hormonal, nervous, and immune systems, including how changes in steroid hormones like estrogen and testosterone during a person's lifetime may affect the development, improvement, or flares of the disease. Additionally, researchers are exploring why rheumatoid arthritis often improves during pregnancy. Efforts are also underway to develop drugs that can reduce inflammation and slow or stop the disease with few side effects. Clinical trials on rheumatoid arthritis are available through Clinical Trials.gov, where individuals can search for current studies by typing "rheumatoid arthritis" into the search box [medquad_ans_0000061-5__dup2]. The National Institute of Arthritis and Musculoskeletal and Skin Diseases (

**Supported citations**: `medquad_ans_0000061-5__dup2`
**Unsupported citations**: _none_

### Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.976, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0002561-4` (score=0.0812) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (an...
2. `medquad_ans_0002561-2` (score=0.0812) - What are the signs and symptoms of Glass-Chapman-Hockley syndrome? Glass-Chapman-Hockley syndrome has only been described in one family with five affected family members in three generations. The signs and symptoms se...
3. `medquad_ans_0002561-1` (score=0.0796) - The Glass-Chapman-Hockley syndrome is a very rare disease. To date, the syndrome has only been reported in one family with five members affected in three generations. The first patients were two brothers that had an a...
4. `medquad_ans_0000883-5__dup1` (score=0.0733) - How might Buschke Ollendorff syndrome be treated? There is currently no cure for BOS. Surgical removal of lesions on or under the skin may be done for cosmetic purposes. In some patients, surgical treatment of deafnes...
5. `medquad_ans_0000860-2__dup1` (score=0.0582) - How might Brown-Sequard syndrome be treated?

**Answer**:

The treatment for Glass-Chapman-Hockley syndrome is typically surgery, which is based on the person's specific signs and symptoms. The goal of the surgery is to increase the space in the front (anterior) part of the skull. The operation is usually performed when the person is between 9 to 12 months of age. If other sutures, other than the coronal suture, are involved, other surgeries may be performed [medquad_ans_0002561-4].

**Supported citations**: `medquad_ans_0002561-4`
**Unsupported citations**: _none_

### Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.769, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000868-2` (score=0.0820) - Renpenning syndrome is a rare disorder; its prevalence is unknown. More than 60 affected individuals in at least 15 families have been identified.
2. `medquad_ans_0000868-1` (score=0.0796) - Renpenning syndrome is a disorder that almost exclusively affects males, causing developmental delay, moderate to severe intellectual disability, and distinctive physical features. Individuals with Renpenning syndrome...
3. `medquad_ans_0003615-3` (score=0.0621) - How might CREST syndrome be treated? Unfortunately, CREST syndrome has no known cure. The condition carries both physical and psychological consequences, so a holistic approach to management should be taken. Treatment...
4. `medquad_ans_0003593-6` (score=0.0594) - How might Liddle syndrome be treated? Treatment for Liddle syndrome includes following a low sodium diet as well as taking potassium-sparing diuretics, which reduce blood pressure and correct hypokalemia and metabolic...
5. `medquad_ans_0000247-2__dup2` (score=0.0590) - There is no cure for Rett syndrome. Treatment for the disorder is symptomatic, focusing on the management of symptoms, and supportive. Medication may be needed for breathing irregularities and motor difficulties, and ...

**Answer**:

The treatments for Renpenning syndrome are not specified in the provided context. [medquad_ans_0000868-1] describes the features and characteristics of the disorder but does not mention any treatments. Therefore, based on the available information, no treatments for Renpenning syndrome can be identified.

**Supported citations**: `medquad_ans_0000868-1`
**Unsupported citations**: _none_

### Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Scores**: context_relevance=1.000, groundedness=0.735, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001100-1__dup1` (score=0.0799) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is a...
2. `medquad_ans_0001104-2` (score=0.0777) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2F? The subtypes of Charcot-Marie-Tooth type 2, including type 2F, have similar signs and symptoms. Affected individuals usually become symptomatic b...
3. `medquad_ans_0001105-1` (score=0.0768) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is ava...
4. `medquad_ans_0001111-1` (score=0.0763) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2N? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2N. If the information is ava...
5. `medquad_ans_0001112-1` (score=0.0754) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2O? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2O. If the information is ava...

**Answer**:

The provided context does not contain information about the symptoms of Charcot-Marie-Tooth disease type 2G. [medquad_ans_0001100-1__dup1] discusses symptoms of Charcot-Marie-Tooth disease type 2B2, and [medquad_ans_0001104-2] discusses symptoms of Charcot-Marie-Tooth disease type 2F, but neither mentions type 2G. Therefore, based on the available context, the symptoms of Charcot-Marie-Tooth disease type 2G cannot be determined.

**Supported citations**: `medquad_ans_0001100-1__dup1`, `medquad_ans_0001104-2`
**Unsupported citations**: _none_

### Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Scores**: context_relevance=1.000, groundedness=0.974, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0001393-1` (score=0.0489) - Coccygodynia is a rare condition in that causes pain in and around the coccyx (tailbone). Although various causes have been described for the condition, the more common causes are direct falls and injury.
2. `medquad_ans_0001393-2` (score=0.0487) - What signs and symptoms are associated with coccygodynia? The classic symptom is pain when pressure is applied to the tailbone, such as when sitting on a hard chair. Symptoms usually improve with relief of pressure wh...
3. `medquad_ans_0000189-6` (score=0.0469) - Some people with porphyria-causing gene mutations have latent porphyria, meaning they have no symptoms of the disorder. Symptoms of cutaneous porphyrias include - oversensitivity to sunlight - blisters on exposed area...
4. `medquad_ans_0001393-3` (score=0.0454) - What causes coccygodynia? A number of different causes have been associated with coccygodynia. However, the most common cause is a direct fall and injury to the area of the sacrum and coccyx. These types of injuries c...
5. `medquad_ans_0000134-5__dup1` (score=0.0431) - Zollinger-Ellison syndrome signs and symptoms are similar to those of peptic ulcers. A dull or burning pain felt anywhere between the navel and midchest is the most common symptom of a peptic ulcer. This discomfort us...

**Answer**:

The symptoms of coccygodynia include pain when pressure is applied to the tailbone, such as when sitting on a hard chair, which usually improves with relief of pressure when standing or walking. Other symptoms are immediate and severe pain when moving from sitting to standing, pain during bowel movements, pain during sex, and a deep ache in the region of the tailbone [medquad_ans_0001393-2].

**Supported citations**: `medquad_ans_0001393-2`
**Unsupported citations**: _none_

### Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Scores**: context_relevance=1.000, groundedness=0.901, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0004025-1` (score=0.0820) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgen...
2. `medquad_ans_0003783-2` (score=0.0796) - What are the signs and symptoms of Mandibulofacial dysostosis with microcephaly? Mandibulofacial dysostosis with microcephaly (MFDM) may affect multiple parts of the body but primarily affects the head and face. Peopl...
3. `medquad_ans_0003270-2` (score=0.0770) - What are the signs and symptoms of Isolated corpus callosum agenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Isolated corpus callosum agenesis. If the information is availab...
4. `medquad_ans_0000623-1` (score=0.0696) - Mandibulofacial dysostosis with microcephaly (MFDM) is a disorder that causes abnormalities of the head and face. People with this disorder often have an unusually small head at birth, and the head does not grow at th...
5. `medquad_ans_0002040-1` (score=0.0681) - What are the signs and symptoms of Ectrodactyly and ectodermal dysplasia without cleft lip/palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Ectrodactyly and ectodermal dysplas...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, and microcephaly. These symptoms are listed in the Human Phenotype Ontology as part of the signs and symptoms associated with this condition [medquad_ans_0004025-1]. Additionally, other symptoms such as preaxial hand polydactyly, proptosis, and seizures may also be present, though their frequency varies and is not always specified [medquad_ans_0004025-1].

**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

### Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.798, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000562-5` (score=0.0487) - These resources address the diagnosis or management of Klippel-Feil syndrome: - Genetic Testing Registry: Klippel Feil syndrome - Genetic Testing Registry: Klippel-Feil syndrome 1, autosomal dominant - Genetic Testing...
2. `medquad_ans_0000170-3__dup1` (score=0.0484) - The prognosis for most individuals with Klippel-Feil Syndrome is good if the disorder is treated early and appropriately. Activities that can injure the neck should be avoided.
3. `medquad_ans_0000170-2__dup1` (score=0.0481) - Treatment for Klippel-Feil Syndrome is symptomatic and may include surgery to relieve cervical or craniocervical instability and constriction of the spinal cord, and to correct scoliosis. Physical therapy may also be ...
4. `medquad_ans_0003402-2` (score=0.0469) - What are the signs and symptoms of Klippel Feil syndrome? Klippel Feil syndrome is characterized by the fusion of 2 or more spinal bones in the neck (cervical vertebrae). The condition is present from birth (congenita...
5. `medquad_ans_0000170-1__dup1` (score=0.0457) - Klippel-Feil Syndrome is a rare disorder characterized by the congenital fusion of two or more cervical (neck) vertebrae. It is caused by a failure in the normal segmentation or division of the cervical vertebrae duri...

**Answer**:

The provided context does not contain specific information on how to diagnose Klippel-Feil syndrome. While it mentions resources such as the Genetic Testing Registry and MedlinePlus that offer information on diagnosis and management, no detailed diagnostic methods or criteria are described in the given text. Therefore, based on the available context, the diagnosis of Klippel-Feil syndrome cannot be answered definitively.

[medquad_ans_0000562-5]

**Supported citations**: `medquad_ans_0000562-5`
**Unsupported citations**: _none_

### Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.833, groundedness=0.496, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000222-2` (score=0.0656) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
2. `medquad_ans_0000222-1` (score=0.0643) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
3. `medquad_ans_0000222-5` (score=0.0630) - These resources address the diagnosis or management of congenital diaphragmatic hernia: - Boston Children's Hospital - Children's Hospital of Philadelphia - Columbia University Medical Center: DHREAMS - Columbia Unive...
4. `medquad_ans_0000222-4` (score=0.0628) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or c...
5. `medquad_ans_0001497-1` (score=0.0620) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severit...

**Answer**:

Congenital diaphragmatic hernia is a defect in the diaphragm caused by abnormal development before birth. The diaphragm, composed of muscle and fibrous tissue, separates the abdominal organs from those in the chest. A defect allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs. This can lead to underdevelopment of the lungs (pulmonary hypoplasia) and life-threatening breathing difficulties at birth. The condition may present at birth or later in life with symptoms such as breathing problems or abdominal pain due to intestinal protrusion into the chest. In about 1 percent of cases, it is asymptomatic and may be detected incidentally during imaging for other reasons. It is classified by location: Bochdalek hernias (80–90% of cases, involving the side or back of the diaphragm), Morgagni hernias (about 2% of cases,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Scores**: context_relevance=1.000, groundedness=0.959, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000173-2` (score=0.0656) - Charcot-Marie-Tooth disease is the most common inherited disorder that involves the peripheral nerves, affecting an estimated 150,000 people in the United States. It occurs in populations worldwide with a prevalence o...
2. `medquad_ans_0001088-1__dup1` (score=0.0635) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
3. `medquad_ans_0000173-1` (score=0.0625) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
4. `medquad_ans_0001093-1__dup1` (score=0.0623) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is ava...
5. `medquad_ans_0000173-5` (score=0.0604) - These resources address the diagnosis or management of Charcot-Marie-Tooth disease: - Gene Review: Gene Review: Charcot-Marie-Tooth Hereditary Neuropathy Overview - Gene Review: Gene Review: Charcot-Marie-Tooth Neurop...

**Answer**:

Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves, which connect the brain and spinal cord to muscles and sensory cells that detect sensations such as touch, pain, heat, and sound. Damage to these nerves can result in loss of sensation and wasting (atrophy) of muscles in the feet, legs, and hands [medquad_ans_0000173-1]. It is the most common inherited disorder involving the peripheral nerves, affecting an estimated 150,000 people in the United States with a global prevalence of about 1 in 2,500 individuals [medquad_ans_0000173-2]. The disease can present with symptoms such as foot drop, foot deformity, numbness, slapping gait, and weakness in the hips, legs, or feet, and typically becomes apparent in adolescence or early adulthood, though onset can

**Supported citations**: `medquad_ans_0000173-1`, `medquad_ans_0000173-2`
**Unsupported citations**: _none_

### Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000784-4` (score=0.0256) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000998-4` (score=0.0249) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000898-4` (score=0.0228) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000676-4` (score=0.0225) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0001083-4` (score=0.0220) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether monilethrix is inherited. While multiple documents describe autosomal recessive inheritance patterns, none explicitly mention monilethrix or its mode of inheritance. Therefore, based on the given information, it cannot be determined if monilethrix is inherited.

[doc_id] is not applicable due to insufficient context.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Scores**: context_relevance=0.875, groundedness=0.896, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000215-15` (score=0.0648) - - Urinary tract infections (UTIs) usually occur when the body fails to remove bacteria rapidly from the urinary tract. - UTIs affect about 3 percent of children in the United States every year. - Most UTIs are not ser...
2. `medquad_ans_0000223-12` (score=0.0645) - - A urinary tract infection (UTI) is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscope. - Most UTIs are caused by bacteria that live in the bowel, ...
3. `medquad_ans_0000215-4__dup2` (score=0.0633) - Urinary tract infections affect about 3 percent of children in the United States every year. UTIs account for more than 1 million visits to pediatricians offices every year.1
4. `medquad_ans_0000214-1__dup4` (score=0.0618) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Norma...
5. `medquad_ans_0000215-1__dup3` (score=0.0613) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Norma...

**Answer**:

A urinary tract infection (UTI) in children is an infection in the urinary tract caused by microorganisms too small to be seen without a microscope, most commonly bacteria [medquad_ans_0000214-1__dup4]. These infections occur when bacteria that live in the bowel enter the urinary tract and overcome the body's natural defenses, which normally remove bacteria rapidly [medquad_ans_0000214-1__dup4]. Bacteria may travel up the ureters to infect the kidneys, leading to a kidney infection known as pyelonephritis [medquad_ans_0000214-1__dup4]. UTIs affect about 3 percent of children in the United States every year and account for over 1 million pediatric office visits annually [medquad_ans_0000215-4__dup2]. While most UTIs are not serious, chronic

**Supported citations**: `medquad_ans_0000214-1__dup4`, `medquad_ans_0000215-4__dup2`
**Unsupported citations**: _none_

### Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.965, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0004939-2` (score=0.0656) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their abilit...
2. `medquad_ans_0000804-1` (score=0.0645) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the...
3. `medquad_ans_0000804-2` (score=0.0635) - Polycystic kidney disease is a fairly common genetic disorder. It affects about 500,000 people in the United States. The autosomal dominant form of the disease is much more common than the autosomal recessive form. Au...
4. `medquad_ans_0004939-1` (score=0.0625) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the ...
5. `medquad_ans_0000167-1__dup4` (score=0.0611) - Polycystic kidney disease is a genetic disorder that causes numerous cysts to grow in the kidneys. A kidney cyst is an abnormal sac filled with fluid. PKD cysts can greatly enlarge the kidneys while replacing much of ...

**Answer**:

Complications of Polycystic Kidney Disease include dangerously high blood pressure (hypertension), severe pain in the back or sides, blood in the urine (hematuria), recurrent urinary tract infections, kidney stones (nephrolithiasis), heart valve abnormalities, and an increased risk of aortic aneurysm in the brain (cerebral aneurysm) [medquad_ans_0004939-2]. Additionally, cysts may develop in other organs, particularly the liver, leading to cystic liver disease [medquad_ans_0004939-2]. Other complications include renal insufficiency, proteinuria, and abnormality of prenatal development or birth [medquad_ans_0004939-2]. The frequency of these symptoms varies, with hypertension, hematuria, nephrolithiasis, proteinuria, and renal insufficiency each

**Supported citations**: `medquad_ans_0004939-2`
**Unsupported citations**: _none_

### Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Scores**: context_relevance=0.800, groundedness=0.935, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000649-1__dup1` (score=0.0656) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contami...
2. `medquad_ans_0000949-1__dup1` (score=0.0643) - Viruses are capsules with genetic material inside. They are very tiny, much smaller than bacteria. Viruses cause familiar infectious diseases such as the common cold, flu and warts. They also cause severe illnesses su...
3. `medquad_ans_0000123-4__dup2` (score=0.0630) - Four types of viruses cause most cases of viral gastroenteritis. Rotavirus Rotavirus is the leading cause of gastroenteritis among infants and young children. Rotavirus infections are most common in infants 3 to 15 mo...
4. `medquad_ans_0000228-4__dup2` (score=0.0620) - Laboratory tests that are used to diagnose Hendra virus (HV) and Nipah virus (NV) include detection of antibody by ELISA (IgG and IgM), real time polymerase chain reaction (RT-PCR), and virus isolation attempts. In mo...
5. `medquad_ans_0000149-4__dup1` (score=0.0598) - The NINDS supports research on shingles and shingles-related conditions. Current studies focus on the relationship between the persistence of neurotropic viruses and development of neurological diseases including herp...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by noroviruses, which are a group of related viruses. Infection leads to an inflammation of the stomach and intestines, with common symptoms including diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms are fever, headache, and body aches. The illness can spread from person to person, through contaminated food or water, or by touching a contaminated surface. Norovirus is the most common calicivirus and the most common cause of viral gastroenteritis in adults, often responsible for epidemics, especially from October to April. There is no specific medicine to treat norovirus infections; treatment involves bed rest and plenty of fluids to prevent dehydration. Prevention can be achieved through proper hand washing and safe food preparation [medquad_ans_0000649-1__dup1]medquad_ans_0000123-4\_\_

**Supported citations**: `medquad_ans_0000649-1__dup1`
**Unsupported citations**: _none_

### Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.700, groundedness=0.725, answer_relevance=0.780

**Top docs**:

1. `medquad_ans_0000464-1__dup1` (score=0.0618) - A hernia happens when part of an internal organ or tissue bulges through a weak area of muscle. Most hernias are in the abdomen. There are several types of hernias, including - Inguinal, in the groin. This is the the ...
2. `medquad_ans_0000110-1__dup3` (score=0.0573) - An inguinal hernia happens when contents of the abdomenusually fat or part of the small intestinebulge through a weak area in the lower abdominal wall. The abdomen is the area between the chest and the hips. The area ...
3. `medquad_ans_0000167-2__dup3` (score=0.0341) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up...
4. `medquad_ans_0000110-3__dup2` (score=0.0339) - The cause of inguinal hernias depends on the type of inguinal hernia. Indirect inguinal hernias. A defect in the abdominal wall that is present at birth causes an indirect inguinal hernia. During the development of th...
5. `medquad_ans_0000222-4` (score=0.0328) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or c...

**Answer**:

The provided context does not contain information about the genetic changes related to congenital diaphragmatic hernia. [medquad_ans_0000464-1__dup1] mentions congenital diaphragmatic hernia as a birth defect that requires surgery, but it does not specify any genetic changes. Therefore, based on the given context, the genetic changes related to congenital diaphragmatic hernia cannot be determined.

**Supported citations**: `medquad_ans_0000464-1__dup1`
**Unsupported citations**: _none_

### Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.075, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000011-13` (score=0.0754) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...
2. `medquad_ans_0002533-3` (score=0.0656) - What causes Gerstmann-Straussler-Scheinker disease? Gerstmann-Straussler-Scheinker disease (GSS) is usually caused by certain changes (mutations) in the PRNP gene. PRNP encodes a protein called prion protein. Although...
3. `medquad_ans_0004453-1` (score=0.0604) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004456-1` (score=0.0602) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004450-1` (score=0.0595) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

The provided context does not contain information about Omenn syndrome. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000097-1` (score=0.0492) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...
2. `medquad_ans_0005792-1` (score=0.0481) - Spinocerebellar ataxia autosomal recessive 7, also called SCAR7, is a slowly progressive hereditary form of spinocerebellar ataxia. Symptoms of SCAR7 can include difficulty walking and writing, speech difficulties (dy...
3. `medquad_ans_0000032-3__dup1` (score=0.0469) - The prognosis for individuals with ataxia and cerebellar/spinocerebellar degeneration varies depending on its underlying cause.
4. `medquad_ans_0005761-1` (score=0.0462) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal...
5. `medquad_ans_0000097-5` (score=0.0459) - These resources address the diagnosis or management of ARCA1: - Gene Review: Gene Review: SYNE1-Related Autosomal Recessive Cerebellar Ataxia - Genetic Testing Registry: Spinocerebellar ataxia, autosomal recessive 8 -...

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive movement problems due to a loss (atrophy) of nerve cells in the cerebellum, the part of the brain that coordinates movement. Signs and symptoms first appear in early to mid-adulthood and include impaired speech (dysarthria), problems with coordination and balance (ataxia), or both. Individuals may also experience difficulty with movements that involve judging distance or scale (dysmetria), abnormal eye movements (nystagmus), and problems following the movements of objects with the eyes. The movement problems are slowly progressive and may eventually result in the need for a cane, walker, or wheelchair [medquad_ans_0000097-1].

**Supported citations**: `medquad_ans_0000097-1`
**Unsupported citations**: _none_

### Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Scores**: context_relevance=0.750, groundedness=0.961, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000049-2__dup1` (score=0.0802) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
2. `medquad_ans_0000050-2__dup1` (score=0.0799) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
3. `medquad_ans_0000123-1__dup2` (score=0.0799) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Symptoms may include - A limp or paralyzed...
4. `medquad_ans_0000050-3__dup1` (score=0.0782) - The site and type of brachial plexus injury determines the prognosis. For avulsion and rupture injuries, there is no potential for recovery unless surgical reconnection is made in a timely manner. The potential for re...
5. `medquad_ans_0000049-3__dup1` (score=0.0770) - The site and type of brachial plexus injury determines the prognosis. For avulsion and rupture injuries, there is no potential for recovery unless surgical reconnection is made in a timely manner. The potential for re...

**Answer**:

Treatment for brachial plexus injuries includes physical therapy and, in some cases, surgery [medquad_ans_0000049-2__dup1].

**Supported citations**: `medquad_ans_0000049-2__dup1`
**Unsupported citations**: _none_

### Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Scores**: context_relevance=0.833, groundedness=0.945, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000216-7` (score=0.0820) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
2. `medquad_ans_0000216-8` (score=0.0804) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PU...
3. `medquad_ans_0000216-3__dup1` (score=0.0796) - Before leaving the hospital, a baby with urine blockage may urinate only small amounts or may not urinate at all. As part of the routine newborn exam, the health care provider may feel an enlarged kidney or find a clo...
4. `medquad_ans_0000216-4__dup1` (score=0.0781) - When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis. Hydronephrosis is the most common problem found during prenatal ...
5. `medquad_ans_0000154-7` (score=0.0767) - The complications of kidney dysplasia can include - hydronephrosis of the working kidney. A baby with kidney dysplasia in only one kidney might have other urinary tract defects. When other defects in the urinary tract...

**Answer**:

Urine blockage in newborns can be caused by various defects in the urinary tract, including vesicoureteral reflux (VUR), ureteropelvic junction (UPJ) obstruction, bladder outlet obstruction (BOO) such as posterior urethral valves (PUV), and ureterocele. Additionally, genetic conditions affecting multiple body systems, such as prune belly syndrome (PBS), esophageal atresia (EA), and congenital heart defects, can contribute to urine blockage. Spina bifida and other birth defects affecting the spinal cord may also cause urine blockage [medquad_ans_0000216-8].

**Supported citations**: `medquad_ans_0000216-8`
**Unsupported citations**: _none_

### Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Scores**: context_relevance=0.667, groundedness=0.945, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000526-2` (score=0.0492) - Intranuclear rod myopathy is a rare disorder that has been identified in only a small number of individuals. Its exact prevalence is unknown.
2. `medquad_ans_0000526-1` (score=0.0484) - Intranuclear rod myopathy is a disorder that primarily affects skeletal muscles, which are muscles that the body uses for movement. People with intranuclear rod myopathy have severe muscle weakness (myopathy) and poor...
3. `medquad_ans_0000526-5` (score=0.0476) - These resources address the diagnosis or management of intranuclear rod myopathy: - Genetic Testing Registry: Nemaline myopathy 3 These resources from MedlinePlus offer information about the diagnosis and management o...
4. `medquad_ans_0000705-2` (score=0.0466) - The prevalence of myofibrillar myopathy is unknown.
5. `medquad_ans_0000165-2` (score=0.0464) - Centronuclear myopathy is a rare condition; its exact prevalence is unknown.

**Answer**:

The exact prevalence of intranuclear rod myopathy is unknown [medquad_ans_0000526-2].

**Supported citations**: `medquad_ans_0000526-2`
**Unsupported citations**: _none_

### Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Scores**: context_relevance=0.833, groundedness=0.954, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000036-11` (score=0.0653) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. There are two main types of high blood pressure: primary and secondary high blood pressure...
2. `medquad_ans_0000058-3__dup4` (score=0.0648) - Blood pressure is the force of blood flow inside your blood vessels. Blood pressure is written with two numbers separated by a slash. For example, a blood pressure result of 130/80 is said as 130 over 80. The first nu...
3. `medquad_ans_0000036-9` (score=0.0619) - Blood pressure is the force of blood pushing against the walls of the blood vessels as the heart pumps blood. If your blood pressure rises and stays high over time, its called high blood pressure. High blood pressure ...
4. `medquad_ans_0000150-1__dup3` (score=0.0614) - Blood pressure is the force of blood pushing against blood vessel walls as the heart pumps out blood, and high blood pressure, also called hypertension, is an increase in the amount of force that blood places on blood...
5. `medquad_ans_0000071-1__dup2` (score=0.0609) - Espaol High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. Measuring Blood Pressure Blood pressure is the force of blood pushing against the w...

**Answer**:

High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. It is defined as an increase in the force of blood pushing against the walls of blood vessels as the heart pumps blood. This increased force can strain the heart and damage blood vessels, increasing the risk of heart attack, stroke, eye problems, and kidney problems. High blood pressure is written with two numbers separated by a slash: the first number (systolic) represents the pressure when the heart beats and pushes blood through the vessels, and the second number (diastolic) represents the pressure when the vessels relax between heartbeats. It is most commonly classified as primary or essential high blood pressure, which develops over years as a person ages, or secondary high blood pressure, which is caused by another medical condition or the use of certain medicines [medquad_ans_0000036-11]medquad_ans

**Supported citations**: `medquad_ans_0000036-11`
**Unsupported citations**: _none_

### Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000692-1__dup2` (score=0.0320) - Parkinson's disease (PD) is a type of movement disorder. It happens when nerve cells in the brain don't produce enough of a brain chemical called dopamine. Sometimes it is genetic, but most cases do not seem to run in...
2. `medquad_ans_0000307-1__dup1` (score=0.0315) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...
3. `medquad_ans_0000139-2__dup2` (score=0.0311) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...
4. `medquad_ans_0000052-31` (score=0.0305) - Genetics is one of the most exciting areas of Parkinson's disease research. Studying the genes responsible for inherited cases can help researchers understand both inherited and sporadic cases of the disease. Sporadic...
5. `medquad_ans_0000052-8` (score=0.0300) - Parkinson's disease is a brain disorder that leads to shaking, stiffness, and difficulty with walking, balance, and coordination. It currently affects about half a million people in the United States, although the num...

**Answer**:

The provided context does not mention X-linked dystonia-parkinsonism or its inheritance pattern. Therefore, it is not possible to determine based on the given information whether X-linked dystonia-parkinsonism is inherited.

Answer: Insufficient context [medquad_ans_0000307-1__dup1, medquad_ans_0000692-1__dup2]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000307-1__dup1, medquad_ans_0000692-1__dup2`

### Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000120-1` (score=0.0656) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process keto...
2. `medquad_ans_0000120-2` (score=0.0643) - Beta-ketothiolase deficiency appears to be very rare. It is estimated to affect fewer than 1 in 1 million newborns.
3. `medquad_ans_0000702-2__dup1` (score=0.0630) - What are the signs and symptoms of Beta ketothiolase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Beta ketothiolase deficiency. If the information is available, the ta...
4. `medquad_ans_0000702-1__dup1` (score=0.0625) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ket...
5. `medquad_ans_0000120-5` (score=0.0623) - These resources address the diagnosis or management of beta-ketothiolase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of acetyl-CoA acetyltransferase These resources from MedlinePlus offer in...

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. The signs and symptoms typically appear between the ages of 6 months and 24 months and include episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, known as ketoacidotic attacks, are frequently triggered by infections, periods without food (fasting), or increased intake of protein-rich foods [medquad_ans_0000120-1]. It is estimated to affect fewer than 1 in 1 million newborns [medquad_ans_0000120-2].

**Supported citations**: `medquad_ans_0000120-1`, `medquad_ans_0000120-2`
**Unsupported citations**: _none_

### Example 285 (`0000006_5-7`)

**Question**: What are the stages of Childhood Brain Stem Glioma ?

**Scores**: context_relevance=0.778, groundedness=0.817, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000006_5-1` (score=0.0656) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - T...
2. `medquad_ans_0000006_5-7` (score=0.0643) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the...
3. `medquad_ans_0000006_5-4` (score=0.0637) - The signs and symptoms of brain stem glioma are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain. - The size of the tumor and whether it has spread all thro...
4. `medquad_ans_0000006_5-3` (score=0.0615) - Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not having risk factors doesnt mean that you will not get cancer. Talk with ...
5. `medquad_ans_0000006_3-1` (score=0.0611) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...

**Answer**:

The provided context does not mention any stages of childhood brain stem glioma. Therefore, it is not possible to determine the stages based on the given information. [medquad_ans_0000006_5-1]

**Supported citations**: `medquad_ans_0000006_5-1`
**Unsupported citations**: _none_

### Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.528, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000266-1__dup2` (score=0.0653) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal c...
2. `medquad_ans_0000266-3__dup1` (score=0.0648) - With treatment, individuals with tethered spinal cord syndrome have a normal life expectancy. However, some neurological and motor impairments may not be fully correctable. Surgery soon after symptoms emerge appears t...
3. `medquad_ans_0004257-2` (score=0.0630) - What are the signs and symptoms of myelomeningocele? A baby born with a myelomeningocele may have a sac sticking out of the mid to lower back that the doctor cannot see through when shining a light behind it. Symptoms...
4. `medquad_ans_0000266-2__dup1` (score=0.0616) - MRI imaging is often used to evaluate individuals with these symptoms, and can be used to diagnose the location of the tethering, lower than normal position of the conus medullaris, or presence of a tumor or fatty mas...
5. `medquad_ans_0000257-1__dup2` (score=0.0611) - Spinal cord infarction is a stroke either within the spinal cord or the arteries that supply it. It is caused by arteriosclerosis or a thickening or closing of the major arteries to the spinal cord. Frequently spinal ...

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course. Symptoms in children may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is often linked to improper growth of the neural tube during fetal development and is closely associated with spina bifida. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. Delayed symptom presentation is related to chronic strain on the spinal cord, which may be exacerbated

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.558, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000058-1__dup1` (score=0.0656) - Carpal tunnel syndrome (CTS) occurs when the median nerve, which runs from the forearm into the palm of the hand, becomes pressed or squeezed at the wrist. The carpal tunnel is a narrow, rigid passageway of ligament a...
2. `medquad_ans_0000058-3__dup1` (score=0.0645) - In general, carpal tunnel syndrome responds well to treatment, but less than half of individuals report their hand(s) feeling completely normal following surgery. Some residual numbness or weakness is common. At work,...
3. `medquad_ans_0000155-1__dup3` (score=0.0628) - You're working at your desk, trying to ignore the tingling or numbness you've had for some time in your hand and wrist. Suddenly, a sharp, piercing pain shoots through the wrist and up your arm. Just a passing cramp? ...
4. `medquad_ans_0000469-1` (score=0.0616) - Hereditary neuropathy with liability to pressure palsies is a disorder that affects peripheral nerves. These nerves connect the brain and spinal cord to muscles as well as sensory cells that detect touch, pain, and te...
5. `medquad_ans_0000970-1` (score=0.0602) - Tarsal-carpal coalition syndrome is a rare, inherited bone disorder that affects primarily the hands and feet. Several individual bones make up each wrist (carpal bones) and ankle (tarsal bones). In tarsal-carpal coal...

**Answer**:

The outlook for carpal tunnel syndrome is generally positive, as the condition responds well to treatment. However, less than half of individuals report their hand(s) feeling completely normal following surgery, and some residual numbness or weakness is common [medquad_ans_0000058-3__dup1]. Early diagnosis and treatment are important to prevent permanent nerve damage [medquad_ans_00000155-1__dup3]. With appropriate management, such as splints, rest, medications, or surgery, symptoms can be effectively managed, although full recovery is not guaranteed in all cases.

**Supported citations**: `medquad_ans_0000058-3__dup1`
**Unsupported citations**: `medquad_ans_00000155-1__dup3`

### Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.020, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0006133-6` (score=0.0593) - Is there treatment for Townes-Brocks syndrome? Treatment is directed towards the specific symptoms, including immediate surgical intervention for imperforate anus; surgery for severe malformations of the hands; routin...
2. `medquad_ans_0000080-2__dup1` (score=0.0581) - There is no cure and no standard course of treatment for Coffin-Lowry syndrome. Treatment is symptomatic and supportive, and may include physical and speech therapy and educational services.
3. `medquad_ans_0003799-5` (score=0.0565) - How might Marden-Walker syndrome be treated? Very little information is available regarding the treatment of Marden-Walker syndrome. In general, treatment is symptomatic, with a multidisciplinary approach. The team of...
4. `medquad_ans_0000094-2__dup1` (score=0.0552) - Treatment for individuals with Dandy-Walker Syndrome generally consists of treating the associated problems, if needed. A surgical procedure called a shunt may be required to drain off excess fluid within the brain, w...
5. `medquad_ans_0005761-1` (score=0.0547) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal...

**Answer**:

The provided context does not contain any information about the treatments for Donnai-Barrow syndrome. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 289 (`0000811-1`)

**Question**: What is (are) porphyria ?

**Scores**: context_relevance=0.750, groundedness=0.951, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000729-1__dup1` (score=0.0648) - Porphyrias are a group of genetic disorders caused by problems with how your body makes a substance called heme. Heme is found throughout the body, especially in your blood and bone marrow, where it carries oxygen. Th...
2. `medquad_ans_0000811-1` (score=0.0638) - Porphyria is a group of disorders caused by abnormalities in the chemical steps that lead to heme production. Heme is a vital molecule for all of the body's organs, although it is most abundant in the blood, bone marr...
3. `medquad_ans_0001134-1` (score=0.0625) - Chester porphyria is a unique type of porphyria with the signs and symptoms of acute intermittent porphyria (AIP) and the biochemical defects of both AIP and variegate porphyria (VP). Chester porphyria does not confor...
4. `medquad_ans_0000189-7` (score=0.0621) - A health care provider diagnoses porphyria with blood, urine, and stool tests. These tests take place at a health care providers office or a commercial facility. A blood test involves drawing blood and sending the sam...
5. `medquad_ans_0000811-2` (score=0.0611) - The exact prevalence of porphyria is unknown, but it probably ranges from 1 in 500 to 1 in 50,000 people worldwide. Overall, porphyria cutanea tarda is the most common type of porphyria. For some forms of porphyria, t...

**Answer**:

Porphyria is a group of genetic disorders caused by abnormalities in the chemical steps that lead to heme production. Heme is a vital molecule found throughout the body, especially in the blood, bone marrow, and liver, and is a component of hemoproteins such as hemoglobin, which carries oxygen in the blood. There are two main types: cutaneous porphyrias, which primarily affect the skin and cause blistering, itching, and fragility of skin exposed to sunlight; and acute porphyrias, which primarily affect the nervous system and cause symptoms such as abdominal pain, vomiting, constipation, muscle weakness, seizures, mental changes, and pain in the chest, abdomen, limbs, or back. These symptoms can come and go and are often triggered by factors like certain medications, alcohol, smoking, infections, stress, or sun exposure [medquad_ans_0000811-1]medquad_ans_0

**Supported citations**: `medquad_ans_0000811-1`
**Unsupported citations**: _none_

### Example 290 (`0000585-1__dup1`)

**Question**: What is (are) Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000934-1__dup1` (score=0.0353) - Usher syndrome is an inherited disease that causes serious hearing loss and retinitis pigmentosa, an eye disorder that causes your vision to get worse over time. It is the most common condition that affects both heari...
2. `medquad_ans_0000585-1__dup1` (score=0.0328) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
3. `medquad_ans_0000585-2__dup1` (score=0.0325) - What are the signs and symptoms of Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal recessive opt...
4. `medquad_ans_0001900-1` (score=0.0323) - Dominant optic atrophy (DOA) is an inherited optic nerve disorder characterized by degeneration of the optic nerves. It typically starts during the first decade of life. Affected people usually develop moderate visual...
5. `medquad_ans_0004557-1` (score=0.0323) - Optic atrophy 1 is a condition that mainly affects vision, but may include other features. Vision loss typically begins within the first decade of life; severity varies widely among affected people (from nearly normal...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. It is inherited in an autosomal recessive pattern [medquad_ans_0000585-1__dup1].

**Supported citations**: `medquad_ans_0000585-1__dup1`
**Unsupported citations**: _none_

### Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.055, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000167-2__dup3` (score=0.0631) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up...
2. `medquad_ans_0000167-11` (score=0.0531) - Autosomal recessive PKD is a rare genetic disorder that affects the liver as well as the kidneys. The signs of autosomal recessive PKD frequently appear in the earliest months of life, even in the womb, so health care...
3. `medquad_ans_0002833-3` (score=0.0481) - What causes hereditary leiomyomatosis and renal cell cancer? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is caused by changes (mutations) in the FH gene. This gene gives the body instructions for making an...
4. `medquad_ans_0001047-3` (score=0.0460) - X-linked chondrodysplasia punctata 1 is caused by genetic changes involving the ARSE gene. This gene provides instructions for making an enzyme called arylsulfatase E. The function of this enzyme is unknown, although ...
5. `medquad_ans_0004167-3` (score=0.0459) - What causes Muir-Torre syndrome? Muir-Torre syndrome is a subtype of Lynch syndrome and may be caused by changes (mutations) in either the MLH1, MSH2, or MSH6 gene. These genes give the body instructions to make prote...

**Answer**:

The provided context does not mention phosphoglycerate kinase deficiency or any genetic changes related to this condition. Therefore, it is not possible to answer the question based on the given information. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Scores**: context_relevance=0.600, groundedness=0.117, answer_relevance=0.760

**Top docs**:

1. `medquad_ans_0000109-4__dup2` (score=0.0455) - Pulmonary hypertension (PH) begins with inflammation and changes in the cells that line your pulmonary arteries. Other factors also can affect the pulmonary arteries and cause PH. For example, the condition may develo...
2. `medquad_ans_0000323-1__dup2` (score=0.0447) - Emphysema is a type of COPD involving damage to the air sacs (alveoli) in the lungs. As a result, your body does not get the oxygen it needs. Emphysema makes it hard to catch your breath. You may also have a chronic c...
3. `medquad_ans_0000367-4` (score=0.0430) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000038-1__dup2` (score=0.0429) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chroni...
5. `medquad_ans_0000022-5__dup1` (score=0.0423) - Many babies who develop bronchopulmonary dysplasia (BPD) are born with serious respiratory distress syndrome (RDS). The signs and symptoms of RDS at birth are: Rapid, shallow breathing Sharp pulling in of the chest be...

**Answer**:

The provided context does not contain information about genetic changes related to pulmonary veno-occlusive disease. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Scores**: context_relevance=0.625, groundedness=0.875, answer_relevance=0.825

**Top docs**:

1. `medquad_ans_0000189-5__dup1` (score=0.0608) - Most porphyrias are inherited disorders. Scientists have identified genes for all eight enzymes in the heme biosynthetic pathway. Most porphyrias result from inheriting an abnormal gene, also called a gene mutation, f...
2. `medquad_ans_0006262-1` (score=0.0492) - Tyrosinemia type 3 is a genetic disorder characterized by elevated blood levels of the amino acid tyrosine, a building block of most proteins. This condition is caused by a deficiency of the enzyme 4-hydroxyphenylpyru...
3. `medquad_ans_0006260-1` (score=0.0484) - Tyrosinemia type 1 is a genetic disorder characterized by elevated blood levels of the amino acid tyrosine, a building block of most proteins. This condition is caused by a shortage of the enzyme fumarylacetoacetate h...
4. `medquad_ans_0006260-3` (score=0.0476) - How might tyrosinemia type 1 be treated? There is currently no cure for tyrosinemia type 1. Individuals with this condition need to be on a special diet restricted in two amino acids, tyrosine and phenylalanine, throu...
5. `medquad_ans_0001006-5` (score=0.0469) - These resources address the diagnosis or management of tyrosinemia: - Baby's First Test: Tyrosinemia, Type I - Baby's First Test: Tyrosinemia, Type II - Baby's First Test: Tyrosinemia, Type III - Gene Review: Gene Rev...

**Answer**:

The genetic changes related to tyrosinemia are mutations in specific genes that lead to enzyme deficiencies in the tyrosine breakdown pathway. In tyrosinemia type 1, mutations in the FAH gene cause a shortage of the enzyme fumarylacetoacetate hydrolase [medquad_ans_0006260-1]. In tyrosinemia type 3, mutations in the HPD gene result in a deficiency of the enzyme 4-hydroxyphenylpyruvate dioxygenase [medquad_ans_0006262-1]. Both conditions are inherited in an autosomal recessive manner, meaning a person must inherit two abnormal copies of the gene (one from each parent) to develop the disorder [medquad_ans_0006260-1][medquad_ans_0006262-1].

**Supported citations**: `medquad_ans_0006260-1`, `medquad_ans_0006262-1`
**Unsupported citations**: _none_

### Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Scores**: context_relevance=0.857, groundedness=0.479, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000436-5` (score=0.0797) - These resources address the diagnosis or management of Graves disease: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Graves disease 2 - Genetic Testing Registry: Graves disease 3 -...
2. `medquad_ans_0000004-6__dup2` (score=0.0794) - People with Graves disease have three treatment options: radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves disease in the United States. Graves disease ...
3. `medquad_ans_0000004-2__dup3` (score=0.0794) - People with Graves disease may have common symptoms of hyperthyroidism such as - nervousness or irritability - fatigue or muscle weakness - heat intolerance - trouble sleeping - hand tremors - rapid and irregular hear...
4. `medquad_ans_0000004-3__dup4` (score=0.0772) - Graves ophthalmopathy is a condition associated with Graves disease that occurs when cells from the immune system attack the muscles and other tissues around the eyes. The result is inflammation and a buildup of tissu...
5. `medquad_ans_0000004-9` (score=0.0770) - - Graves disease is the most common cause of hyperthyroidism in the United States. - In Graves disease, the immune system stimulates the thyroid gland to make too much thyroid hormone. - Common symptoms of hyperthyroi...

**Answer**:

The treatments for Graves' disease include radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves' disease in the United States. In this therapy, patients take radioactive iodine-131 by mouth, which is collected by the thyroid gland and gradually destroys thyroid cells, reducing hormone production. The dose may be adjusted to either shut down the thyroid completely or bring hormone production into the normal range. Multiple rounds may be needed, and improvements in symptoms may take several weeks or months. Radioiodine therapy is not used in pregnant women or women who are breastfeeding due to potential harm to the fetus or passage into breast milk. Experts recommend waiting at least one year after treatment before becoming pregnant. Almost everyone who receives radioiodine therapy eventually develops hypothyroidism. Other treatments, such as prescription steroids, may help prevent worsening of Graves' ophthalmopathy (GO) associated with radioiodine therapy [med

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Scores**: context_relevance=0.800, groundedness=0.470, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000039-1__dup2` (score=0.0653) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, ...
2. `medquad_ans_0000039-2__dup1` (score=0.0648) - As yet, no specific treatment is known that can halt or reverse the symptoms of Batten disease. However, seizures can sometimes be reduced or controlled with anticonvulsant drugs, and other medical problems can be tre...
3. `medquad_ans_0000039-3__dup1` (score=0.0625) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten dise...
4. `medquad_ans_0000545-5` (score=0.0620) - These resources address the diagnosis or management of juvenile Batten disease: - Batten Disease Diagnostic and Clinical Research Center at the University of Rochester Medical Center - Batten Disease Support and Resea...
5. `medquad_ans_0000545-2` (score=0.0620) - Juvenile Batten disease is the most common type of NCL, but its exact prevalence is unknown. Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, wher...

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. It is characterized by a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. Early signs may include personality and behavior changes, slow learning, clumsiness, or stumbling, and vision loss is often an early symptom, leading to suspicion during an eye exam. The disease progresses over time, resulting in cognitive impairment, worsening seizures, progressive loss of sight and motor skills, and eventually blindness, bedridden state, and dementia. It is often fatal by the late teens or twenties. Juvenile Batten disease is the most common type of Neuronal Ceroid-Lipofuscinosis (NCL), though its exact prevalence is unknown; collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide, with higher prevalence in Finland (approximately 1 in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Scores**: context_relevance=1.000, groundedness=0.767, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000318-1__dup1` (score=0.0640) - Ehlers-Danlos syndrome (EDS) is a group of inherited disorders that weaken connective tissues. Connective tissues are proteins that support skin, bones, blood vessels, and other organs. EDS usually affects your skin, ...
2. `medquad_ans_0000146-3__dup2` (score=0.0473) - More detailed information on the diagnosis, management, and treatment of ehrlichiosis is available in Diagnosis and Management of Tickborne Rickettsial Diseases: Rocky Mountain Spotted Fever, Ehrlichioses, and Anaplas...
3. `medquad_ans_0002061-5` (score=0.0325) - How might Ehlers-Danlos syndrome progeroid type be treated? Individuals with Ehlers-Danlos Syndrome progeroid type can benefit from a variety of treatments depending on their symptoms. Affected children with weak musc...
4. `medquad_ans_0002061-2` (score=0.0325) - What are the signs and symptoms of Ehlers-Danlos syndrome, progeroid type? Ehlers-Danlos syndrome refers to a group of connective tissue disorders characterized by stretchy or kneadable skin, double jointedness, and d...
5. `medquad_ans_0005116-1` (score=0.0323) - Proteus-like syndrome describes people who do not meet the diagnostic criteria for Proteus syndrome but who share many of the characteristic signs and symptoms associated with the condition. Affected people may experi...

**Answer**:

The provided context does not mention "Ehlers-Danlos syndrome, progeroid type." Therefore, based on the given information, it is not possible to answer the question. [medquad_ans_0000318-1__dup1] discusses general Ehlers-Danlos syndrome but does not specify the progeroid type.

**Supported citations**: `medquad_ans_0000318-1__dup1`
**Unsupported citations**: _none_

### Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=0.900, groundedness=0.966, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.0656) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
2. `medquad_ans_0000416-3` (score=0.0645) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
3. `medquad_ans_0002601-1` (score=0.0630) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...
4. `medquad_ans_0000416-5` (score=0.0630) - These resources address the diagnosis or management of glutamate formiminotransferase deficiency: - Baby's First Test - Genetic Testing Registry: Glutamate formiminotransferase deficiency These resources from MedlineP...
5. `medquad_ans_0000416-1` (score=0.0615) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with ...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the last two steps of histidine metabolism and in the production of one form of the vitamin folate [medquad_ans_0000416-3].

**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

### Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Scores**: context_relevance=0.889, groundedness=0.949, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001047-2` (score=0.0492) - The prevalence of X-linked chondrodysplasia punctata 1 is unknown. Several dozen affected males have been reported in the scientific literature.
2. `medquad_ans_0001048-2` (score=0.0481) - X-linked chondrodysplasia punctata 2 has been estimated to affect fewer than 1 in 400,000 newborns. However, the disorder may actually be more common than this estimate because it is likely underdiagnosed, particularl...
3. `medquad_ans_0001169-1` (score=0.0476) - Chondrodysplasia punctata 1, X-linked recessive (CDPX1) is a genetic disorder present from birth that affects bone and cartilage development. On x-ray, affected infants have characteristic spots at the ends of their b...
4. `medquad_ans_0001047-1` (score=0.0469) - X-linked chondrodysplasia punctata 1 is a disorder of cartilage and bone development that occurs almost exclusively in males. Chondrodysplasia punctata is an abnormality that appears on x-rays as spots (stippling) nea...
5. `medquad_ans_0001048-5` (score=0.0459) - These resources address the diagnosis or management of X-linked chondrodysplasia punctata 2: - Gene Review: Gene Review: Chondrodysplasia Punctata 2, X-Linked - Genetic Testing Registry: Chondrodysplasia punctata 2 X-...

**Answer**:

Symptoms of Chondrodysplasia punctata 1, X-linked recessive (CDPX1) include characteristic stippled epiphyses seen on x-ray at the ends of bones, particularly in the ankles, toes, and fingers, which typically disappear between ages 2 and 3 [medquad_ans_0001169-1]. Other features include shortened fingers and unusually short fingertips and ends of the toes, a flattened nose with crescent-shaped nostrils and a flat nasal bridge, and in some cases, breathing abnormalities due to abnormal thickening (stenosis) of airway cartilage, hearing loss, abnormalities of the spinal bones in the neck, and delayed intellectual development [medquad_ans_0001169-1][medquad_ans_0001047-1]. Most individuals have normal intelligence and a normal life expectancy, although some may experience life-threatening complications [med

**Supported citations**: `medquad_ans_0001169-1`, `medquad_ans_0001047-1`
**Unsupported citations**: _none_

### Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Scores**: context_relevance=0.714, groundedness=0.980, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000165-2__dup1` (score=0.0807) - The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfa...
2. `medquad_ans_0000132-9` (score=0.0801) - Experts have not yet found a way to prevent Whipple disease.
3. `medquad_ans_0000132-1__dup3` (score=0.0784) - Whipple disease is a rare bacterial infection that primarily affects the small intestine. The infection may spread to any organ in the body; however, it more commonly affects the - joints - central nervous system, whi...
4. `medquad_ans_0000132-6` (score=0.0784) - People with Whipple disease may have complications caused by malnutrition, which is due to damaged villi in the small intestine. As a result of delayed diagnosis or treatment, people may experience the following compl...
5. `medquad_ans_0000132-10` (score=0.0777) - A person with Whipple disease and malabsorption may need - a diet high in calories and protein - vitamins - nutritional supplements

**Answer**:

The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfadizine or solfamethoxazole can treat neurological symptoms. Relapsing neurologic Whipple's disease, marked by bouts of worsening of symptoms, is sometimes treated with a combination of antibiotics and weekly injections of interferon gamma, a substance made by the body that activates the immune system [medquad_ans_0000165-2__dup1].

**Supported citations**: `medquad_ans_0000165-2__dup1`
**Unsupported citations**: _none_

### Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Scores**: context_relevance=0.857, groundedness=0.916, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000676-4__dup1` (score=0.0817) - How might Bell's palsy be treated?
2. `medquad_ans_0000676-3__dup1` (score=0.0807) - What causes Bell's palsy?
3. `medquad_ans_0000676-2__dup1` (score=0.0791) - What are the symptoms of Bell's palsy?
4. `medquad_ans_0000042-3__dup2` (score=0.0786) - The prognosis for individuals with Bell's palsy is generally very good. The extent of nerve damage determines the extent of recovery. With or without treatment, most individuals begin to get better within 2 weeks afte...
5. `medquad_ans_0000676-1__dup1` (score=0.0762) - Bell's palsy is a form of temporary facial paralysis which results from damage or trauma to one of the facial nerves. This disorder is characterized by the sudden onset of facial paralysis that often affects just one ...

**Answer**:

Treatments for Bell's palsy include steroids such as prednisone, which may reduce inflammation and swelling of the 7th cranial nerve; antiviral medications like acyclovir, which are used to fight viral infections; and pain relievers such as aspirin, acetaminophen, or ibuprofen to relieve pain. Additionally, physical therapy, facial massage, and acupuncture have been used as complementary treatments [medquad_ans_0000676-1__dup1].

**Supported citations**: `medquad_ans_0000676-1__dup1`
**Unsupported citations**: _none_
