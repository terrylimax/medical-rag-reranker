# Generation Evaluation

## Summary

- `avg_answer_chars`: 552.0800
- `avg_answer_relevance`: 0.7519
- `avg_citation_presence_rate`: 0.8500
- `avg_context_relevance`: 0.8586
- `avg_empty_answer`: 0.0000
- `avg_end_to_end_latency_ms`: 24144.3023
- `avg_generation_latency_ms`: 23973.7142
- `avg_groundedness`: 0.7296
- `avg_insufficient_context`: 0.0367
- `avg_num_retrieved_docs`: 5.0000
- `avg_rerank_latency_ms`: 0.0000
- `avg_retrieval_latency_ms`: 170.3539
- `avg_supported_citation_rate`: 0.7006
- `avg_unsupported_citation_rate`: 0.1494
- `num_examples`: 300
- `reranker_enabled_rate`: 0.0000

## Examples

### Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006510-1` (score=0.7143) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the informatio...
2. `medquad_ans_0006511-1` (score=0.7073) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the informatio...
3. `medquad_ans_0006509-2` (score=0.7003) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is...
4. `medquad_ans_0002834-2` (score=0.6062) - What are the signs and symptoms of Hereditary lymphedema type II? Hereditary lymphedema type II is characterized by the abnormal transport of lymph fluid. This causes the lymph fluid to build up, causing swelling (lym...
5. `medquad_ans_0002781-2` (score=0.5900) - What are the signs and symptoms of Hemophagocytic lymphohistiocytosis? The signs and symptoms of hemophagocytic lymphohistiocytosis typically develop during the first months or years of life. However, in rare cases, a...

**Answer**:

The symptoms of X-linked lymphoproliferative syndrome 1 include cellular immunodeficiency (90%), decreased antibody level in blood (50%), hepatomegaly (50%), lymphadenopathy (50%), lymphoma (50%), splenomegaly (50%), anemia (7.5%), encephalitis, fulminant hepatitis, hepatic encephalopathy, IgG deficiency, immunodeficiency, increased IgM level, meningitis, pancytopenia, recurrent pharyngitis, reduced natural killer cell activity, thrombocytopenia, and X-linked recessive inheritance [medquad_ans_0006510-1].

**Supported citations**: `medquad_ans_0006510-1`
**Unsupported citations**: _none_

### Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Scores**: context_relevance=0.600, groundedness=0.803, answer_relevance=0.920

**Top docs**:

1. `medquad_ans_0000307-1__dup1` (score=0.5938) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...
2. `medquad_ans_0004750-4` (score=0.5616) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is pass...
3. `medquad_ans_0003601-2` (score=0.5354) - How is limb-girdle muscular dystrophy inherited? Limb-girdle muscular dystrophy (LGMD) is most often inherited in an autosomal recessive manner; less commonly, rare sub-types may be inherited in an autosomal dominant ...
4. `medquad_ans_0000139-2__dup2` (score=0.5220) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...
5. `medquad_ans_0000673-4` (score=0.5133) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.

**Answer**:

The provided context does not specify whether early-onset primary dystonia is inherited. While it states that "some people inherit dystonia," it does not differentiate between early-onset primary dystonia and other forms or specify the inheritance pattern for early-onset cases [medquad_ans_0000307-1__dup1]. Therefore, based on the available information, the inheritance of early-onset primary dystonia cannot be determined.

**Supported citations**: `medquad_ans_0000307-1__dup1`
**Unsupported citations**: _none_

### Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Scores**: context_relevance=0.857, groundedness=0.912, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000397-1__dup2` (score=0.6910) - Taeniasis in humans is a parasitic infection caused by the tapeworm species Taenia saginata (beef tapeworm), Taenia solium (pork tapeworm), and Taenia asiatica (Asian tapeworm). Humans can become infected with these t...
2. `medquad_ans_0000397-2__dup1` (score=0.6613) - The tapeworms that cause taeniasis (Taenia saginata, T. solium, and T. asiatica) are found worldwide. Eating raw or undercooked beef or pork is the primary risk factor for acquiring taeniasis. Persons who don't eat ra...
3. `medquad_ans_0000689-1__dup2` (score=0.5942) - Parasites are living things that use other living things - like your body - for food and a place to live. You can get them from contaminated food or water, a bug bite, or sexual contact. Some parasitic diseases are ea...
4. `medquad_ans_0000366-1__dup2` (score=0.5753) - Schistosomiasis, also known as bilharzia, is a disease caused by parasitic worms. Infection with Schistosoma mansoni, S. haematobium, and S. japonicum causes illness in humans; less commonly, S. mekongi and S. interca...
5. `medquad_ans_0000121-2__dup3` (score=0.5734) - Cysticercosis is an infection caused by the larvae of the parasite Taenia solium. This infection occurs after a person swallows tapeworm eggs. The larvae get into tissues such as muscle and brain, and form cysts there...

**Answer**:

People at risk for taeniasis include those who eat raw or undercooked beef (particularly infected with Taenia saginata) or pork (with Taenia solium or Taenia asiatica). Individuals in under-developed communities with poor sanitation, especially in Latin America, Eastern Europe, sub-Saharan Africa, India, and Asia, are at higher risk. Additionally, people in the United States, particularly Latin American immigrants, may be at risk due to consumption of raw or undercooked pork or exposure to contaminated environments. Those with poor hygiene who shed tapeworm eggs in feces may also contaminate their environment and risk transmitting the infection to themselves or others [medquad_ans_0000397-1\_\_dup2][medquad_ans_0000397-2__dup1].

**Supported citations**: `medquad_ans_0000397-1__dup2`, `medquad_ans_0000397-2__dup1`
**Unsupported citations**: _none_

### Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.955, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000194-2__dup1` (score=0.7888) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
2. `medquad_ans_0000196-1__dup1` (score=0.7390) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
3. `medquad_ans_0000197-1__dup1` (score=0.7390) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000029-2` (score=0.7249) - Aicardi-Goutieres syndrome is a rare disorder. Its exact prevalence is unknown.
5. `medquad_ans_0000193-2__dup1` (score=0.7068) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 1. If the information is availab...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy [medquad_ans_0000194-2__dup1]. These symptoms are based on the Human Phenotype Ontology, which compiles data on the frequency of signs and symptoms in this condition [medquad_ans_0000194-2__dup1].

**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

### Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003421-1` (score=0.7551) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0005502-2` (score=0.6672) - What are the signs and symptoms of Schnitzler syndrome? The signs and symptoms of Schnitzler syndrome vary but may include: Red raised patches of skin (urticaria) that may become itchy Recurrent fevers Join pain and i...
3. `medquad_ans_0003339-2` (score=0.6490) - What are the signs and symptoms of Kallmann syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kallmann syndrome. If the information is available, the table below includes how...
4. `medquad_ans_0003423-1` (score=0.6444) - What are the signs and symptoms of Kozlowski Celermajer Tink syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kozlowski Celermajer Tink syndrome. If the information is avail...
5. `medquad_ans_0003418-1` (score=0.6391) - What are the signs and symptoms of Kosztolanyi syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kosztolanyi syndrome. If the information is available, the table below includ...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

### Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.912, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000069-1__dup3` (score=0.5652) - Espaol Hemophilia (heem-o-FILL-ee-ah) is a rare bleeding disorder in which the blood doesn't clot normally. If you have hemophilia, you may bleed for a longer time than others after an injury. You also may bleed insid...
2. `medquad_ans_0000069-3__dup1` (score=0.5549) - A defect in one of the genes that determines how the body makes blood clotting factor VIII or IX causes hemophilia. These genes are located on the X chromosomes (KRO-muh-somz). Chromosomes come in pairs. Females have ...
3. `medquad_ans_0000455-1__dup1` (score=0.5544) - Hemophilia is a rare disorder in which the blood does not clot normally. It is usually inherited. Hemophilia usually occurs in males. If you have hemophilia, you have little or no clotting factor. Clotting factor is a...
4. `medquad_ans_0000067-8` (score=0.5296) - You can't prevent primary, or inherited, hemochromatosis. However, not everyone who inherits hemochromatosis genes develops symptoms or complications of the disease. In those who do, treatments can keep the disease fr...
5. `medquad_ans_0001090-4` (score=0.5146) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

Yes, hemophilia is inherited. It is usually inherited through genes located on the X chromosomes, and individuals are born with the disorder due to a defect in the genes that determine how the body makes blood clotting factor VIII or IX [medquad_ans_0000069-1__dup3]. Males are more commonly affected because they have only one X chromosome, so if that X chromosome carries the hemophilia gene, they will have the disorder. Females with the gene on one X chromosome are typically carriers and can pass the gene to their children [medquad_ans_0000069-1__dup3].

**Supported citations**: `medquad_ans_0000069-1__dup3`
**Unsupported citations**: _none_

### Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.6311) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often t...
2. `medquad_ans_0003103-2` (score=0.5947) - What are the signs and symptoms of IBIDS syndrome? The most common symptoms of Tay syndrome are brittle hair (trichothiodystrophy); dry, thickened, scaling skin (ichthyosis); photosensitivity (abnormal light sensitivi...
3. `medquad_ans_0004163-2` (score=0.5870) - What are the signs and symptoms of Mucopolysaccharidosis type IVA? The Human Phenotype Ontology provides the following list of signs and symptoms for Mucopolysaccharidosis type IVA. If the information is available, th...
4. `medquad_ans_0004162-1` (score=0.5685) - What are the signs and symptoms of Mucopolysaccharidosis type IV? The Human Phenotype Ontology provides the following list of signs and symptoms for Mucopolysaccharidosis type IV. If the information is available, the ...
5. `medquad_ans_0004163-1` (score=0.5634) - Mucopolysaccharidosis type IVA (MPS IVA, also called Morquio syndrome, type A) is a metabolic condition that primarily affects the skeleton. The severity, age of onset, and associated symptoms vary significantly from ...

**Answer**:

The symptoms of IVIC syndrome include hearing impairment (90%), limitation of joint mobility (90%), short stature (90%), strabismus (90%), abnormal dermatoglyphics (50%), aplasia/hypoplasia of the thumb (50%), radioulnar synostosis (50%), scoliosis (50%), synostosis of carpal bones (50%), triphalangeal thumb (50%), abnormality of the clavicle (7.5%), arrhythmia (7.5%), leukocytosis (7.5%), preaxial hand polydactyly (7.5%), thrombocytopenia (7.5%), and urogenital fistula (7.5%) [medquad_ans_0003284-1].

**Supported citations**: `medquad_ans_0003284-1`
**Unsupported citations**: _none_

### Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000775-2` (score=0.8180) - Paroxysmal nocturnal hemoglobinuria is a rare disorder, estimated to affect between 1 and 5 per million people.
2. `medquad_ans_0000775-1` (score=0.7460) - Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. The disorder affects red blood cells (erythrocytes), which carry oxygen; white bloo...
3. `medquad_ans_0000775-5` (score=0.6952) - These resources address the diagnosis or management of paroxysmal nocturnal hemoglobinuria: - Duke University School of Medicine: Hemostasis & Thrombosis Center - Genetic Testing Registry: Paroxysmal nocturnal hemoglo...
4. `medquad_ans_0004747-1` (score=0.6096) - Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It can occur at any age, but is usually diagnosed in young adulthood. People ...
5. `medquad_ans_0004744-1` (score=0.5821) - What are the signs and symptoms of Paroxysmal cold hemoglobinuria? The Human Phenotype Ontology provides the following list of signs and symptoms for Paroxysmal cold hemoglobinuria. If the information is available, th...

**Answer**:

Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. It affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes), which protect the body from infection; and platelets (thrombocytes), which are involved in blood clotting. The disorder affects both sexes equally and can occur at any age, although it is most often diagnosed in young adulthood. People with paroxysmal nocturnal hemoglobinuria experience sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses on the body, such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to hemoglobinuria—dark-colored urine due to the presence of hemoglobin. In many, but not all cases, hemoglobinuria is most noticeable in the morning. The premature destruction

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.972, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000010-10` (score=0.7244) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the li...
2. `medquad_ans_0000010-1__dup5` (score=0.6878) - How Tumors Form The body is made up of many types of cells. Normally, cells grow, divide, and produce more cells as needed to keep the body healthy and functioning properly. Sometimes, however, the process goes wrong ...
3. `medquad_ans_0000037_3-1` (score=0.6715) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...
4. `medquad_ans_0000221-1__dup2` (score=0.6660) - The colon and rectum are part of the large intestine. Colorectal cancer occurs when tumors form in the lining of the large intestine. It is common in both men and women. The risk of developing colorectal cancer rises ...
5. `medquad_ans_0000037_4-1` (score=0.6654) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...

**Answer**:

Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the lining of the large intestine, also called the large bowel [medquad_ans_0000010-10]. It is the second leading cause of death from cancer in the United States [medquad_ans_0000037_3-1].

**Supported citations**: `medquad_ans_0000010-10`, `medquad_ans_0000037_3-1`
**Unsupported citations**: _none_

### Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Scores**: context_relevance=0.833, groundedness=0.993, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000544-2` (score=0.7666) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.
2. `medquad_ans_0000544-5` (score=0.7142) - These resources address the diagnosis or management of junctional epidermolysis bullosa: - Epidermolysis Bullosa Center, Cincinnati Children's Hospital Medical Center - Gene Review: Gene Review: Junctional Epidermolys...
3. `medquad_ans_0002121-1` (score=0.6967) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main ty...
4. `medquad_ans_0002122-5` (score=0.6624) - How might epidermolysis bullosa acquisita be treated?
5. `medquad_ans_0003317-1` (score=0.6558) - Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz ...

**Answer**:

Junctional epidermolysis bullosa (JEB) is a type of epidermolysis bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz type and the Non-Herlitz type. The Herlitz type is very severe, and individuals with this condition often do not survive infancy. The Non-Herlitz type includes several subtypes that cause mild to severe blistering of the skin present at birth or shortly thereafter. JEB is inherited in an autosomal recessive pattern and is caused by mutations in the LAMB3, COL17A1, or LAMC2, and LAMA3 genes. There is no cure for JEB; treatment is focused on management of blistering and prevention of secondary infections [medquad_ans_0003317-1]. Both types

**Supported citations**: `medquad_ans_0003317-1`
**Unsupported citations**: _none_

### Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.880, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000668-2` (score=0.7758) - Molybdenum cofactor deficiency is a rare condition that is estimated to occur in 1 in 100,000 to 200,000 newborns worldwide. More than 100 cases have been reported in the medical literature, although it is thought tha...
2. `medquad_ans_0000668-3` (score=0.7509) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same ...
3. `medquad_ans_0000668-5` (score=0.6905) - These resources address the diagnosis or management of molybdenum cofactor deficiency: - Genetic Testing Registry: Combined molybdoflavoprotein enzyme deficiency - Genetic Testing Registry: Molybdenum cofactor deficie...
4. `medquad_ans_0000668-1` (score=0.6887) - Molybdenum cofactor deficiency is a rare condition characterized by brain dysfunction (encephalopathy) that worsens over time. Babies with this condition appear normal at birth, but within a week they have difficulty ...
5. `medquad_ans_0004143-3` (score=0.5879) - What causes Mowat-Wilson syndrome? Mowat-Wilson syndrome is caused by mutations in the ZEB2 (also known as ZFHX1B or SIP-1) gene. This gene provides instructions for making a protein that plays a critical role in the ...

**Answer**:

The genetic changes related to molybdenum cofactor deficiency are mutations in the MOCS1, MOCS2, or GPHN gene. These mutations cause type A, B, or C of the disorder, respectively, and result in reduced or eliminated function of the associated proteins involved in the biosynthesis of the molybdenum cofactor [medquad_ans_0000668-3].

**Supported citations**: `medquad_ans_0000668-3`
**Unsupported citations**: _none_

### Example 12 (`0000756-1`)

**Question**: What is (are) ornithine translocase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.963, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000756-2` (score=0.8160) - Ornithine translocase deficiency is a very rare disorder. Fewer than 100 affected individuals have been reported worldwide.
2. `medquad_ans_0000756-5` (score=0.6747) - These resources address the diagnosis or management of ornithine translocase deficiency: - Baby's First Test - Gene Review: Gene Review: Hyperornithinemia-Hyperammonemia-Homocitrullinuria Syndrome - Gene Review: Gene ...
3. `medquad_ans_0000755-4` (score=0.6562) - Ornithine transcarbamylase deficiency is an X-linked disorder. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes. A characte...
4. `medquad_ans_0004581-1` (score=0.6514) - Ornithine transcarbamylase (OTC) deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become to...
5. `medquad_ans_0000755-2` (score=0.6496) - Ornithine transcarbamylase deficiency is believed to occur in approximately 1 in every 80,000 people.

**Answer**:

Ornithine translocase deficiency is a very rare disorder, with fewer than 100 affected individuals reported worldwide [medquad_ans_0000756-2].

**Supported citations**: `medquad_ans_0000756-2`
**Unsupported citations**: _none_

### Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000362-1__dup2` (score=0.4115) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
2. `medquad_ans_0000036-8` (score=0.4022) - Medicaid is a state health insurance program for those with low incomes and few assets. Each state runs its own program. The Federal Government requires that Medicaid programs cover a specific set of services; however...
3. `medquad_ans_0000045-28` (score=0.3998) - The general number for Medicare is 1-800-Medicare (1-800-633-4227). TTY users should call 1-877-486-2048. You can also visit http://www.medicare.gov. The "Medicare & You" handbook is mailed out to all Medicare enrolle...
4. `medquad_ans_0000003-6__dup1` (score=0.3776) - Most people with Alzheimers disease are cared for at home by family members. Within families, caregiving is provided most often by wives and husbands, followed by daughters. As Alzheimers disease gets worse, the perso...
5. `medquad_ans_0000036-16` (score=0.3759) - - Diabetes management and treatment is expensive. Many people who have diabetes need help paying for their care. For those who qualify, a variety of government and nongovernment programs can help cover health care exp...

**Answer**:

Yes, there is information about financial assistance. If you don't have health insurance or need help with costs that aren't covered, financial assistance might be available through certain government programs and nonprofit organizations. You can also discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2]. Additionally, Medicaid and CHIP provide financial assistance to individuals with low incomes. Medicaid is a state health insurance program for those with low incomes and few assets, and CHIP offers free or low-cost coverage to children whose parents earn too much for Medicaid but not enough to afford a health plan [medquad_ans_0000036-8].

**Supported citations**: `medquad_ans_0000362-1__dup2`, `medquad_ans_0000036-8`
**Unsupported citations**: _none_

### Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0004105-1` (score=0.7122) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About hal...
2. `medquad_ans_0004105-2` (score=0.5871) - What are the symptoms of Mollaret meningitis? The symptoms of Mollaret meningitis are the same as those found in other types of meningitis. In Mollaret meningitis, however, the symptoms are recurring and are often acc...
3. `medquad_ans_0000598-1__dup2` (score=0.5207) - Meningitis is inflammation of the thin tissue that surrounds the brain and spinal cord, called the meninges. There are several types of meningitis. The most common is viral meningitis, which you get when a virus enter...
4. `medquad_ans_0000115-1__dup1` (score=0.5054) - Meningitis is an infection of the meninges, the membranes that surround the brain and spinal cord. Encephalitis is inflammation of the brain itself. Causes of encephalitis and meningitis include viruses, bacteria, fun...
5. `medquad_ans_0004749-4` (score=0.4895) - How is pars planitis diagnosed? Pars planitis is typically diagnosed based on a specialized eye examination. During the exam, the ophthalmologist will typically see clusters of white blood cells trapped within the eye...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches, separated by weeks or months of no symptoms. About half of affected individuals may also experience long-term abnormalities of the nervous system that come and go, such as seizures, double vision, abnormal reflexes, paralysis of a cranial nerve (paresis), hallucinations, or coma. The exact cause remains unknown, but recent data suggests that herpes simplex virus (HSV-2 and, less frequently, HSV-1) may cause some, if not most cases. Other potential causes include trauma and viral infections other than herpes simplex [medquad_ans_0004105-1]. The symptoms of Mollaret meningitis are similar to those of other types of meningitis and include high fever, severe headache, nausea, vomiting, stiff neck, photophobia (s

**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

### Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.780, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003331-1` (score=0.7766) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
2. `medquad_ans_0003331-3` (score=0.7693) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...
3. `medquad_ans_0003319-1` (score=0.7658) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. U...
4. `medquad_ans_0003319-2` (score=0.7163) - What are the signs and symptoms of juvenile amyotrophic lateral sclerosis? Signs and symptoms of juvenile ALS vary but include slowly to very slowly progressive muscle weakness, increased muscle tone, Babinski reflex,...
5. `medquad_ans_0000551-2` (score=0.7139) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.

**Answer**:

Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be inherited in an autosomal dominant or autosomal recessive fashion [medquad_ans_0003319-1]. However, the specific genetic mutations or molecular mechanisms causing juvenile ALS are not detailed in the provided context. Therefore, based on the available information, the cause is described as genetic with possible inheritance patterns of autosomal dominant or autosomal recessive [medquad_ans_0003319-1].

**Supported citations**: `medquad_ans_0003319-1`
**Unsupported citations**: _none_

### Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Scores**: context_relevance=0.778, groundedness=0.928, answer_relevance=0.689

**Top docs**:

1. `medquad_ans_0000219-11` (score=0.6974) - - Erectile dysfunction (ED) is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last...
2. `medquad_ans_0000334-1__dup2` (score=0.6926) - Erectile dysfunction (ED) is a common type of male sexual dysfunction. It is when a man has trouble getting or keeping an erection. ED becomes more common as you get older. But it's not a natural part of aging. Some p...
3. `medquad_ans_0000219-8` (score=0.6524) - Your doctor can offer you a number of treatments for ED. For many men, the answer is as simple as taking a pill. Other men have to try two or three options before they find a treatment that works for them. Dont give u...
4. `medquad_ans_0000219-1__dup4` (score=0.6117) - Erectile dysfunction\* is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last long ...
5. `medquad_ans_0000219-5__dup2` (score=0.5989) - Having ED can cause you to feel depressed or anxious. ED may also cause low self-esteem. When you have ED, you may not have a satisfying sex life. You may not feel as close with your sexual partner, which may strain y...

**Answer**:

To understand erectile dysfunction (ED), know that it is when you cannot get or keep an erection firm enough for sex. This can occur if you sometimes get an erection, but it does not last long enough, or if you are unable to get an erection at all [medquad_ans_0000219-11]. An erection happens when blood flow into the penis increases, involving hormones, blood vessels, nerves, and muscles working together [medquad_ans_0000219-11]. ED often has multiple causes, such as diseases that damage nerves, arteries, or muscles [medquad_ans_0000219-11].

To determine the cause, your doctor may take your medical and sexual history, ask about your mental health, perform a physical exam, test your blood, conduct a nighttime erection test, or perform an injection or Doppler penile ultrasound [medquad*ans*

**Supported citations**: `medquad_ans_0000219-11`
**Unsupported citations**: _none_

### Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.946, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0006267-3` (score=0.7648) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
2. `medquad_ans_0001537-1` (score=0.6434) - Congenital muscular dystrophy (CMD) refers to a group of inherited conditions that affect the muscles and are present at birth or in early infancy. The severity of the condition, the associated signs and symptoms and ...
3. `medquad_ans_0004514-4` (score=0.6379) - How is oculopharyngeal muscular dystrophy inherited?
4. `medquad_ans_0001026-5` (score=0.6311) - These resources address the diagnosis or management of Walker-Warburg syndrome: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Genetic Testing Registry: Walker-Warburg congenital muscular dystrop...
5. `medquad_ans_0000631-1__dup1` (score=0.6275) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...

**Answer**:

Physical therapy, including early mobilization, regular stretching, and splinting, is the main focus of supportive care for Ullrich congenital muscular dystrophy. Respiratory support and night-time ventilation may become necessary in the first or second decade of life. Prevention of chest infections can be achieved with the use of antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that cyclosporin A may benefit individuals with Ullrich congenital muscular dystrophy, although more studies are needed to confirm its effectiveness [medquad_ans_0006267-3].

**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

### Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004005-1` (score=0.7779) - Microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) is a genetic condition that is mainly characterized by intrauterine and post-natal growth retardation; an abnormally small head size (microcephaly); abn...
2. `medquad_ans_0004006-1` (score=0.7283) - Microcephalic osteodysplastic primordial dwarfism type 2 (MOPD2) is a condition characterized by short stature (dwarfism), skeletal abnormalities and an unusually small head size (microcephaly). Other signs and sympto...
3. `medquad_ans_0004005-5` (score=0.6822) - How might microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) be treated? At this time there are no specific treatments for MOPD1. Treatment is generally supportive. The prognosis is poor for affected ind...
4. `medquad_ans_0000653-1` (score=0.6698) - Microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is a condition characterized by short stature (dwarfism) with other skeletal abnormalities (osteodysplasia) and an unusually small head size (microcep...
5. `medquad_ans_0004006-2` (score=0.6515) - What are the signs and symptoms of Microcephalic osteodysplastic primordial dwarfism type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephalic osteodysplastic primordial ...

**Answer**:

Yes, microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is inherited in an autosomal recessive manner [medquad_ans_0004006-1].

**Supported citations**: `medquad_ans_0004006-1`
**Unsupported citations**: _none_

### Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.719, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000742-2` (score=0.7684) - Ochoa syndrome is a rare disorder. About 150 cases have been reported in the medical literature.
2. `medquad_ans_0006597-4` (score=0.6594) - Can 48,XXYY syndrome be inherited?
3. `medquad_ans_0000742-5` (score=0.6363) - These resources address the diagnosis or management of Ochoa syndrome: - Gene Review: Gene Review: Urofacial Syndrome - Genetic Testing Registry: Ochoa syndrome - National Institute of Diabetes and Digestive and Kidne...
4. `medquad_ans_0000324-4` (score=0.6103) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
5. `medquad_ans_0000883-4` (score=0.6052) - This condition is typically inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutatio...

**Answer**:

The provided context does not explicitly state whether Ochoa syndrome is inherited. While it mentions that about 150 cases have been reported and lists resources for diagnosis and management, there is no direct information on the inheritance pattern of Ochoa syndrome. Therefore, based on the given context, it is not possible to determine if Ochoa syndrome is inherited.

[medquad_ans_0000742-2]

**Supported citations**: `medquad_ans_0000742-2`
**Unsupported citations**: _none_

### Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.833, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000531-2` (score=0.7437) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
2. `medquad_ans_0000531-3` (score=0.6812) - Isodicentric chromosome 15 syndrome results from the presence of an abnormal extra chromosome, called an isodicentric chromosome 15, in each cell. An isodicentric chromosome contains mirror-image segments of genetic m...
3. `medquad_ans_0000531-5` (score=0.6607) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management...
4. `medquad_ans_0000531-1` (score=0.6115) - Isodicentric chromosome 15 syndrome is a developmental disorder with a broad spectrum of features. The signs and symptoms vary among affected individuals. Poor muscle tone is commonly seen in individuals with isodicen...
5. `medquad_ans_0004132-4` (score=0.6111) - How might mosaic trisomy 14 be treated? Treatment for signs and symptoms of mosaic trisomy 14 focuses on the specific features present in each individual. Infants with congenital heart defects may need surgery or othe...

**Answer**:

The provided context does not specify treatments for isodicentric chromosome 15 syndrome. It mentions resources that address the diagnosis and management of the condition, including diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care [medquad_ans_0000531-5], but does not detail specific treatments. Therefore, based on the given information, a definitive answer about treatments cannot be provided.

**Supported citations**: `medquad_ans_0000531-5`
**Unsupported citations**: _none_

### Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.006, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0004414-3` (score=0.5375) - What causes Nicolaides-Baraitser syndrome? Nicolaides-Baraitser syndrome (NCBRS) is caused by mutations in the SMARCA2 gene, which is located on the small arm of chromosome 9. All mutations that have been identified i...
2. `medquad_ans_0002439-1` (score=0.5064) - Friedreich ataxia is an inherited condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features...
3. `medquad_ans_0001143-1` (score=0.5043) - CHILD syndrome, also known as congenital hemidysplasia with ichthyosiform erythroderma and limb defects, is a genetic condition that is typically characterized by large patches of skin that are red and inflamed (eryth...
4. `medquad_ans_0000115-1__dup4` (score=0.4946) - Mntriers disease causes the ridges along the inside of the stomach wallcalled rugaeto enlarge, forming giant folds in the stomach lining. The rugae enlarge because of an overgrowth of mucous cells in the stomach wall....
5. `medquad_ans_0002049-3` (score=0.4899) - What causes Ehlers-Danlos syndrome? Ehlers-Danlos syndrome can be caused by changes (mutations) in several different genes (COL5A1, COL5A2, COL1A1, COL3A1, TNXB, PLOD1, COL1A2, and ADAMTS2). However, the underlying ge...

**Answer**:

The provided context does not mention Troyer syndrome or its cause. Therefore, based on the available information, it is not possible to determine what causes Troyer syndrome. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.930, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000983-4` (score=0.7516) - This condition is considered to have an autosomal dominant pattern of inheritance, which means one copy of the altered CACNA1C gene in each cell is sufficient to cause the disorder. Most cases result from new mutation...
2. `medquad_ans_0003672-1` (score=0.6943) - Timothy syndrome is a type of long QT syndrome. It affects many parts of the body including the heart, fingers, toes, face, and the nervous system. It is characterized by long QT syndrome, although some people with Ti...
3. `medquad_ans_0000983-5` (score=0.6633) - These resources address the diagnosis or management of Timothy syndrome: - Gene Review: Gene Review: Timothy Syndrome - Genetic Testing Registry: Timothy syndrome - MedlinePlus Encyclopedia: Arrhythmias - MedlinePlus ...
4. `medquad_ans_0000983-1` (score=0.6524) - Timothy syndrome is a rare disorder that affects many parts of the body including the heart, digits (fingers and toes), and the nervous system. Timothy syndrome is characterized by a heart condition called long QT syn...
5. `medquad_ans_0000983-2` (score=0.6189) - Timothy syndrome is a rare condition; fewer than 20 people with this disorder have been reported worldwide. The classic type of Timothy syndrome appears to be more common than the atypical type, which has been identif...

**Answer**:

The genetic changes related to Timothy syndrome are changes in the CACNA1C gene. This condition is caused by alterations in the CACNA1C gene, and all cases of Timothy syndrome appear to be due to changes in this gene [medquad_ans_0003672-1]. The disorder follows an autosomal dominant pattern of inheritance, meaning one copy of the altered CACNA1C gene in each cell is sufficient to cause the disorder [medquad_ans_0000983-4]. Most cases result from new mutations in the gene and occur in people with no family history of the disorder, while less commonly, individuals inherit the altered gene from an unaffected parent who is mosaic for a CACNA1C mutation [medquad_ans_0000983-4].

**Supported citations**: `medquad_ans_0003672-1`, `medquad_ans_0000983-4`
**Unsupported citations**: _none_

### Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000061-11` (score=0.7374) - Rheumatoid arthritis can be difficult to diagnose in its early stages because the full range of symptoms develops over time, and only a few symptoms may be present in the early stages. As part of the diagnosis, your d...
2. `medquad_ans_0000875-1` (score=0.7163) - Rheumatoid arthritis is a disease that causes chronic abnormal inflammation, primarily affecting the joints. The most common signs and symptoms are pain, swelling, and stiffness of the joints. Small joints in the hand...
3. `medquad_ans_0000061-7__dup1` (score=0.6907) - Rheumatoid arthritis is an inflammatory disease that causes pain, swelling, stiffness, and loss of function in the joints. It can cause mild to severe symptoms. People with rheumatoid arthritis may feel sick, tired, a...
4. `medquad_ans_0005261-2` (score=0.6627) - What are the signs and symptoms of Reactive arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Reactive arthritis. If the information is available, the table below includes h...
5. `medquad_ans_0000061-3__dup3` (score=0.6515) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by i...

**Answer**:

The symptoms of rheumatoid arthritis include pain, swelling, and stiffness of the joints, with small joints in the hands and feet being most commonly affected. The disease often affects joints in a symmetrical pattern, such as both hands or both knees. Joint pain and stiffness are typically worse in the morning or after prolonged rest. Additional signs and symptoms may include fatigue, low fever, weight loss, anemia, and inflammation of other tissues and organs such as the eyes, lungs, and blood vessels. Some individuals may develop rheumatoid nodules, which are firm, noncancerous lumps under the skin. The symptoms usually appear in mid- to late adulthood and may come in episodes of flares followed by periods of remission [medquad_ans_0000875-1]. The disease can cause mild to severe symptoms and may lead to significant joint damage and disability over time [medquad_ans_00000

**Supported citations**: `medquad_ans_0000875-1`
**Unsupported citations**: _none_

### Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Scores**: context_relevance=1.000, groundedness=0.515, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0002796-1` (score=0.5937) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is no...
2. `medquad_ans_0000149-1__dup3` (score=0.5791) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, d...
3. `medquad_ans_0002796-2` (score=0.5598) - What are the signs and symptoms of Henoch-Schonlein purpura? The Human Phenotype Ontology provides the following list of signs and symptoms for Henoch-Schonlein purpura. If the information is available, the table belo...
4. `medquad_ans_0002796-4` (score=0.5513) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...
5. `medquad_ans_0000149-2__dup2` (score=0.5239) - Henoch-Schnlein purpura is caused by an abnormal immune system response in which the bodys immune system attacks the bodys own cells and organs. Usually, the immune system makes antibodies, or proteins, to protect the...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). It occurs when small blood vessels in the body become inflamed and leak, leading to a rash that appears as many small raised bruises. HSP may develop as an immune response to an infection and is most common in children between 2 and 6 years of age, though it can affect people of any age. Most cases resolve on their own without treatment, with the main goal of treatment being symptom relief such as joint pain, abdominal pain, or swelling. In some cases, over-the-counter medicines are used; in severe arthritis, prednisone, a steroid medicine, may be prescribed. Kidney damage is the most likely long-term complication, and in adults, HSP can lead to chronic kidney disease (CK

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0003166-1` (score=0.7685) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information ...
2. `medquad_ans_0003162-2` (score=0.6968) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 1? Symptoms and physical findings associated with hyper IgM syndrome usually become apparent in the first or second year of life. This condition ...
3. `medquad_ans_0003163-1` (score=0.6805) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information ...
4. `medquad_ans_0003162-1` (score=0.6662) - Hyper IgM syndrome is a type of primary immunodeficiency syndrome. Primary immunodeficiency occurs when part of a persons immune system is missing or does not work correctly. The bodies of people with primary immunode...
5. `medquad_ans_0001446-2` (score=0.6628) - What are the signs and symptoms of Common variable immunodeficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Common variable immunodeficiency. If the information is available...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [medquad_ans_0003166-1].

**Supported citations**: `medquad_ans_0003166-1`
**Unsupported citations**: _none_

### Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Scores**: context_relevance=0.556, groundedness=0.000, answer_relevance=0.556

**Top docs**:

1. `medquad_ans_0002533-4` (score=0.5825) - How is Gerstmann-Straussler-Scheinker disease inherited? Gerstmann-Straussler-Scheinker disease (GSS) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutatio...
2. `medquad_ans_0000883-3__dup1` (score=0.5611) - How is Buschke Ollendorff syndrome inherited? Buschke Ollendorff syndrome (BOS) is caused by mutations in the LEMD3 gene and is inherited in an autosomal dominant manner. This means that only one changed (mutated) cop...
3. `medquad_ans_0000700-4` (score=0.5578) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
4. `medquad_ans_0000516-4` (score=0.5557) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
5. `medquad_ans_0000149-4` (score=0.5555) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

The provided context does not contain any information about Senior-Loken syndrome or its associated genetic changes. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.968, answer_relevance=0.475

**Top docs**:

1. `medquad_ans_0000610-2` (score=0.6590) - The prevalence of lymphedema-distichiasis syndrome is unknown. Because the extra eyelashes can be overlooked during a medical examination, researchers believe that some people with this condition may be misdiagnosed a...
2. `medquad_ans_0003700-2` (score=0.6446) - What are the signs and symptoms of Lymphedema-distichiasis syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lymphedema-distichiasis syndrome. If the information is available...
3. `medquad_ans_0000195-2` (score=0.6119) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0000113-2` (score=0.5761) - Beckwith-Wiedemann syndrome affects an estimated 1 in 13,700 newborns worldwide. The condition may actually be more common than this estimate because some people with mild symptoms are never diagnosed.
5. `medquad_ans_0003700-1` (score=0.5706) - Lymphedema distichiasis syndrome is a condition that affects the normal function of the lymphatic system (part of the immune system that produces and transports fluids and immune cells throughout the body). People wit...

**Answer**:

The incidence of lymphedema-distichiasis syndrome is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people [medquad_ans_0000195-2].

**Supported citations**: `medquad_ans_0000195-2`
**Unsupported citations**: _none_

### Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Scores**: context_relevance=0.857, groundedness=0.397, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000013-22` (score=0.6879) - In a genetic test, a small sample of blood, saliva, or tissue is taken to examine a person's genes. Sometimes, genetic testing can detect diseases that may be preventable or treatable. This type of testing is availabl...
2. `medquad_ans_0000399-1__dup1` (score=0.6621) - Summary : Genetic tests are tests on blood and other tissue to find genetic disorders. Over 2000 tests are available. Doctors use genetic tests for several reasons. These include - Finding genetic diseases in unborn b...
3. `medquad_ans_0000013-25` (score=0.6177) - Genetic testing may be helpful whether the test identifies a mutation or not. Test results can - serve as a relief, eliminating some of the uncertainty about a health condition - help doctors make recommendations for ...
4. `medquad_ans_0000013-26` (score=0.5891) - Finding out your test results can affect you emotionally. Learning that you are someone in your family has or is at risk for a disease can be scary. Some people can also feel guilty, angry, anxious, or depressed when ...
5. `medquad_ans_0000397-1__dup1` (score=0.5408) - Summary : Genetic counseling provides information and support to people who have, or may be at risk for, genetic disorders. A genetic counselor meets with you to discuss genetic risks. The counseling may be for yourse...

**Answer**:

Yes, I have information about genetic testing. Genetic testing involves taking a small sample of blood, saliva, or tissue to examine a person's genes. It can detect genetic disorders and is available for thousands of conditions. Genetic tests are used for various purposes, including identifying genetic diseases in unborn babies, determining if individuals carry a gene that could be passed to their children, screening embryos for disease, testing adults before symptoms appear, making a diagnosis in someone with symptoms, and determining the best type or dose of medicine for a person. Test results can provide relief by reducing uncertainty, help doctors make treatment or monitoring recommendations, support decision-making about personal and family health, encourage preventive actions such as earlier screenings or lifestyle changes, and assist in making future life decisions like family planning. However, test results can also have emotional impacts, such as fear, guilt, anger, anxiety, or depression, and the cost of testing can vary from less than $100 to higher

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.520, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000149-1` (score=0.7519) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxy...
2. `medquad_ans_0000654-2` (score=0.6583) - Microcephaly-capillary malformation syndrome is rare. About a dozen people have been diagnosed with the disorder.
3. `medquad_ans_0000654-5` (score=0.6521) - These resources address the diagnosis or management of microcephaly-capillary malformation syndrome: - Gene Review: Gene Review: Microcephaly-Capillary Malformation Syndrome - Genetic Testing Registry: Microcephaly-ca...
4. `medquad_ans_0006335-1` (score=0.6395) - Vein of Galen aneurysm is a rare form of arteriovenous malformation in which a particular vein at the base of the brain, the vein of Galen, dilates causing too much blood to rush to the heart and leading to congestive...
5. `medquad_ans_0000056-1__dup3` (score=0.6340) - Arteriovenous malformations (AVMs) are defects in your vascular system. The vascular system includes arteries, veins, and capillaries. Arteries carry blood away from the heart to other organs; veins carry blood back t...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin. These malformations appear as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs, and may be visible at birth or during childhood. While capillary malformations alone usually do not cause health problems, some individuals with CM-AVM also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs). These abnormal connections between arteries, veins, and capillaries can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure, some of which can be life-threatening. Complications from AVMs and AVFs often appear in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001992-1` (score=0.6992) - What are the signs and symptoms of Dystonia 16? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 16. If the information is available, the table below includes how often the s...
2. `medquad_ans_0002000-1` (score=0.6802) - What are the signs and symptoms of Dystonia 7, torsion? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 7, torsion. If the information is available, the table below includes...
3. `medquad_ans_0001986-1` (score=0.6691) - What are the signs and symptoms of Dystonia 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 1. If the information is available, the table below includes how often the sym...
4. `medquad_ans_0001988-1` (score=0.6639) - What are the signs and symptoms of Dystonia 11? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 11. If the information is available, the table below includes how often the s...
5. `medquad_ans_0001994-1` (score=0.6638) - What are the signs and symptoms of Dystonia 18? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 18. If the information is available, the table below includes how often the s...

**Answer**:

The symptoms of Dystonia 16 include cognitive impairment (5%), abnormal pyramidal signs, bradykinesia, delayed speech and language development, dysarthria, dysphagia, gait disturbance, hyperreflexia, involuntary movements, laryngeal dystonia, limb dystonia, lower limb pain, morphological abnormality of the pyramidal tract, motor delay, parkinsonism, postural tremor, progressive features, retrocollis [medquad_ans_0001992-1].

**Supported citations**: `medquad_ans_0001992-1`
**Unsupported citations**: _none_

### Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Scores**: context_relevance=0.857, groundedness=0.980, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0005548-5` (score=0.8162) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not tr...
2. `medquad_ans_0000899-2` (score=0.7453) - Septo-optic dysplasia has a reported incidence of 1 in 10,000 newborns.
3. `medquad_ans_0000899-5` (score=0.6951) - These resources address the diagnosis or management of septo-optic dysplasia: - Genetic Testing Registry: Septo-optic dysplasia sequence - MedlinePlus Encyclopedia: Growth Hormone Deficiency - MedlinePlus Encyclopedia...
4. `medquad_ans_0005548-2` (score=0.6868) - What are the signs and symptoms of Septo-optic dysplasia? Symptoms may include blindness in one or both eyes, pupil dilation in response to light, nystagmus (a rapid, involuntary to-and-fro movement of the eyes), inwa...
5. `medquad_ans_0005548-1` (score=0.6601) - Septo-optic dysplasia is a disorder of early brain development. The signs and symptoms vary from person to person; however, underdevelopment (hypoplasia) of the optic nerve, abnormal formation of structures along the ...

**Answer**:

Treatment for septo-optic dysplasia is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not treatable, and vision, physical, and occupational therapies may be required [medquad_ans_0005548-5].

**Supported citations**: `medquad_ans_0005548-5`
**Unsupported citations**: _none_

### Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Scores**: context_relevance=0.900, groundedness=0.883, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000208-1__dup4` (score=0.7198) - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. Benign prostatic hyperplasia is also called benign prostatic hypertrophy or benign prostatic ...
2. `medquad_ans_0000036_3-1` (score=0.6515) - Key Points - Prostate cancer is a disease in which malignant (cancer) cells form in the tissues of the prostate. - Prostate cancer is the most common nonskin cancer among men in the United States. - Different factors ...
3. `medquad_ans_0000208-14` (score=0.6407) - - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. - The prostate is a walnut-shaped gland that is part of the male reproductive system. - The...
4. `medquad_ans_0000752-1__dup1` (score=0.6317) - The prostate is a gland in men. It helps make semen, the fluid that contains sperm. The prostate surrounds the tube that carries urine away from the bladder and out of the body. A young man's prostate is about the siz...
5. `medquad_ans_0000058-19` (score=0.6290) - Yes. Any of the symptoms caused by prostate cancer may also be due to enlargement of the prostate, which is not cancer. If you have any of the symptoms mentioned in question #10, see your doctor or a urologist to find...

**Answer**:

The complications of prostate enlargement due to benign prostatic hyperplasia (BPH) include narrowing of the urethra and urinary retention, which is the inability to empty the bladder completely. This can lead to the bladder wall becoming thicker and eventually weakening, resulting in incomplete bladder emptying and leaving some urine in the bladder [medquad_ans_0000208-1__dup4]. These issues contribute to many of the symptoms and problems associated with BPH [medquad_ans_0000208-1__dup4].

**Supported citations**: `medquad_ans_0000208-1__dup4`
**Unsupported citations**: _none_

### Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Scores**: context_relevance=0.556, groundedness=0.000, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0004784-5` (score=0.4651) - How might Pelizaeus-Merzbacher disease be treated?
2. `medquad_ans_0000029-18` (score=0.4581) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
3. `medquad_ans_0004770-2` (score=0.4464) - How might patulous eustacian tube be treated? While no standard treatment has been found to work for every patient, there are several options that have been used to successfully manage the symptoms in a number of case...
4. `medquad_ans_0006236-5` (score=0.4430) - How might tubular aggregate myopathy be treated?
5. `medquad_ans_0000860-2__dup1` (score=0.4419) - How might Brown-Sequard syndrome be treated?

**Answer**:

The provided context does not contain information about the treatments for Absence of the Septum Pellucidum. Therefore, no treatments can be identified from the given sources. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Scores**: context_relevance=1.000, groundedness=0.518, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.7724) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (...
2. `medquad_ans_0002322-1` (score=0.6422) - Febrile ulceronecrotic Mucha-Habermann disease (FUMHD) is a rare and severe form of pityriasis lichenoides et varioliformis acuta (PLEVA). PLEVA is characterized by skin lesions that ulcerate, breakdown, form open sor...
3. `medquad_ans_0002322-4` (score=0.6421) - How is febrile ulceronecrotic Mucha-Habermann disease definitively diagnosed? FUMHD is diagnosed based upon the clinical symptoms in the patient, with confirmation by skin biopsy. Skin biopsy findings suggestive of FU...
4. `medquad_ans_0002322-5` (score=0.5524) - How is febrile ulceronecrotic Mucha-Habermann disease (FUMHD) treated? It is important that FUMHD is diagnosed and treated as soon as possible. While a number of treatments have been tried, it is hard to asses the ben...
5. `medquad_ans_0000114-2__dup1` (score=0.5509) - What are the signs and symptoms of Acute febrile neutrophilic dermatosis? The most obvious signs of acute febrile neutrophilic dermatosis are distinctive skin lesions that usually develop according to a specific patte...

**Answer**:

The symptoms of Febrile Ulceronecrotic Mucha-Habermann disease (FUMHD) include initial red scaly skin lesions (papules) that ulcerate, break down, form open sores, and then develop a red-brown crust (i.e., PLEVA). These lesions can rapidly progress to large, destructive ulcers with extensive, painful loss of skin tissue. Skin lesions may become infected, leading to pus and a putrid odor. FUMHD is often associated with high fever (up to 104°F), which may be persistent or intermittent. Other symptoms may include feeling ill, sore throat, congestion, muscle soreness or pain, joint pain, diarrhea, central nervous system symptoms, abdominal pain, enlarged spleen, arthritis, megaloblastic anemia, interstitial pneumonitis, lymphocytic (viral) myocarditis, and sepsis. FUMHD can

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Scores**: context_relevance=0.500, groundedness=0.196, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003951-3` (score=0.6129) - Is Merkel cell carcinoma inherited? To our knowledge, there currently is no evidence that Merkel cell carcinoma (MCC) is inherited. While DNA changes (mutations) found in the cells of MCC tumors can lead to MCC, these...
2. `medquad_ans_0000167-5__dup1` (score=0.5263) - Autosomal dominant PKD is the most common form of PKD and the most common inherited disorder of the kidneys.3 The term autosomal dominant means a child can get the disorder by inheriting the gene mutation from only on...
3. `medquad_ans_0000615-4` (score=0.5052) - Maffucci syndrome is not inherited. The mutations that cause this disorder are somatic, which means they occur during a person's lifetime. A somatic mutation occurs in a single cell. As that cell continues to grow and...
4. `medquad_ans_0000167-2__dup3` (score=0.4765) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up...
5. `medquad_ans_0000676-4` (score=0.4730) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain information about hereditary leiomyomatosis and renal cell cancer (HLRCC). Therefore, based on the given context, it is not possible to determine whether HLRCC is inherited.

Answer: Insufficient context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002445-1` (score=0.7618) - What are the signs and symptoms of Frontometaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontometaphyseal dysplasia. If the information is available, the tabl...
2. `medquad_ans_0000389-2` (score=0.7061) - Frontometaphyseal dysplasia is a rare disorder; only a few dozen cases have been reported worldwide.
3. `medquad_ans_0000389-1` (score=0.6977) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which al...
4. `medquad_ans_0002444-1` (score=0.6733) - What are the signs and symptoms of Frontofacionasal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontofacionasal dysplasia. If the information is available, the table ...
5. `medquad_ans_0000389-5` (score=0.6653) - These resources address the diagnosis or management of frontometaphyseal dysplasia: - Gene Review: Gene Review: Otopalatodigital Spectrum Disorders - Genetic Testing Registry: Frontometaphyseal dysplasia These resourc...

**Answer**:

The symptoms of Frontometaphyseal dysplasia include abnormality of dental morphology, abnormality of frontal sinus, abnormality of the metaphyses, bowing of the long bones, camptodactyly of finger, craniofacial hyperostosis, hypertelorism, limitation of joint mobility, prominent supraorbital ridges, abnormal form of the vertebral bodies, abnormality of the palate, accelerated skeletal maturation, aplasia/hypoplasia of the thumb, arachnodactyly, conductive hearing impairment, elbow dislocation, scoliosis, sensorineural hearing impairment, skeletal muscle atrophy, synostosis of carpal bones, ulnar deviation of finger, abnormality of the larynx, abnormality of the urethra, complete atrioventricular canal defect, craniosynostosis, tracheal stenosis, ureteral stenosis, ankle

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Scores**: context_relevance=0.818, groundedness=0.925, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000128-7` (score=0.7321) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding,...
2. `medquad_ans_0000128-11` (score=0.7117) - - Diverticular disease is more common in people as they grow older. - A low-fiber diet is the most likely cause of the disease. - Most people are treated with a high-fiber diet and pain medication. - Add whole grain f...
3. `medquad_ans_0000128-10` (score=0.6938) - Eat a high-fiber diet to help prevent problems. Talk to your doctor about using fiber products like Benefiber, Citrucel, or Metamucil. Daily use can help you get the fiber you need if you do not get it through your di...
4. `medquad_ans_0000097-8` (score=0.6471) - A health care provider may treat the symptoms of diverticulosis with a high-fiber diet or fiber supplements, medications, and possibly probiotics. Treatment for diverticular disease varies, depending on whether a pers...
5. `medquad_ans_0000128-5__dup1` (score=0.6305) - Many people get diverticular disease. Starting at age 40, the chance of getting it increases about every 10 years. About half of people between the ages of 60 and 80 have diverticular disease. Almost everyone over 80 ...

**Answer**:

Treatments for diverticular disease depend on the severity of the condition and whether the person has diverticulosis or diverticulitis. Most people improve with dietary changes, particularly a high-fiber diet. This includes consuming whole grain foods, high-fiber fruits, and vegetables such as raw apples, peaches, pears, tangerines, fresh broccoli, squash, carrots, brussels sprouts, potatoes, baked beans, kidney beans, lima beans, whole-wheat bread, brown rice, bran flake cereal, and oatmeal [medquad_ans_0000128-10].

A health care provider may also recommend fiber supplements like methylcellulose (Citrucel) or psyllium (Metamucil), taken one to three times a day with at least 8 ounces of water, to help meet fiber needs [medquad_ans_0000097

**Supported citations**: `medquad_ans_0000128-10`
**Unsupported citations**: _none_

### Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000619-1__dup1` (score=0.7896) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is avail...
2. `medquad_ans_0000676-2__dup1` (score=0.6506) - What are the symptoms of Bell's palsy?
3. `medquad_ans_0006066-1` (score=0.6074) - What are the signs and symptoms of Thompson Baraitser syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thompson Baraitser syndrome. If the information is available, the tabl...
4. `medquad_ans_0004414-2` (score=0.5815) - What are the signs and symptoms of Nicolaides-Baraitser syndrome? Nicolaides-Baraitser syndrome (NCBRS) is typically characterized by intellectual disability, seizures, short stature, sparse hair, distinctive facial f...
5. `medquad_ans_0004414-1` (score=0.5806) - Nicolaides-Baraitser syndrome (NCBRS) is a very rare condition characterized by severe intellectual disability and various physical features. Signs and symptoms may include seizures, short stature, sparse hair, distin...

**Answer**:

The symptoms of Baraitser Brett Piesowicz syndrome include cerebral calcification (90%), hyperreflexia (90%), hypertonia (90%), microcephaly (90%), seizures (90%), abnormality of movement (50%), cerebral cortical atrophy (50%), cataract (5%), opacification of the corneal stroma (5%), renal insufficiency (5%), anteverted nares, autosomal recessive inheritance, cerebellar hypoplasia, decreased liver function, elevated hepatic transaminases, failure to thrive, hepatomegaly, high palate, increased CSF protein, intellectual disability, profound, jaundice, lissencephaly, long philtrum, low-set ears, microretrognathia, muscular hypotonia of the trunk, nystagmus, pachygyria, petechiae, phenotypic

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Scores**: context_relevance=0.833, groundedness=0.025, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000052-2__dup1` (score=0.5567) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
2. `medquad_ans_0000029-18` (score=0.5257) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
3. `medquad_ans_0000273-2__dup2` (score=0.5091) - There is no cure for TSC, although treatment is available for a number of the symptoms. Rapamycin and related drugs are not yet approved by the U.S. Food and Drug Administration (FDA) for any purpose in individuals wi...
4. `medquad_ans_0004053-4` (score=0.5020) - What is the treatment for microscopic polyangiitis (MPA)? MPA is treated with medications that suppress the immune system, which can lower an individual's resistance to infections. There are a variety of immune suppre...
5. `medquad_ans_0000205-2__dup1` (score=0.5007) - There is no specific treatment to stop or reverse any form of MD. Treatment may include physical therapy, respiratory therapy, speech therapy, orthopedic appliances used for support, and corrective orthopedic surgery....

**Answer**:

The provided context does not contain information about the treatments for meningioma. Therefore, a specific answer cannot be given based on the available context. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000006_7-2` (score=0.7419) - Key Points - Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and s...
2. `medquad_ans_0000006_7-8` (score=0.7110) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a te...
3. `medquad_ans_0000006_7-3` (score=0.6832) - The cause of most childhood CNS germ cell tumors is not known.
4. `medquad_ans_0000015_1-2` (score=0.6464) - The cause of most childhood extracranial germ cell tumors is unknown.
5. `medquad_ans_0000006_7-1` (score=0.6460) - Key Points - Childhood central nervous system (CNS) germ cell tumors form from germ cells. - There are different types of childhood CNS germ cell tumors. - Germinomas - Nongerminomas - The cause of most childhood CNS ...

**Answer**:

The treatments for Childhood Central Nervous System (CNS) Germ Cell Tumors include radiation therapy, chemotherapy, surgery, high-dose chemotherapy with stem cell rescue, and new treatments being tested in clinical trials. Treatment is planned by a team of healthcare providers who are experts in treating cancer in children. Patients may consider participating in a clinical trial, which can occur before, during, or after starting treatment [medquad_ans_0000006_7-8].

**Supported citations**: `medquad_ans_0000006_7-8`
**Unsupported citations**: _none_

### Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Scores**: context_relevance=1.000, groundedness=0.457, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0000729-2` (score=0.8922) - Neutral lipid storage disease with myopathy is a rare condition; its incidence is unknown.
2. `medquad_ans_0000729-1` (score=0.8792) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the ...
3. `medquad_ans_0004401-1` (score=0.8629) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (my...
4. `medquad_ans_0004401-5` (score=0.7746) - How might neutral lipid storage disease with myopathy be treated? To date, there is no treatment for the underlying metabolic problem. Current therapies include adhering to strict dietary guidelines and utilizing trea...
5. `medquad_ans_0004401-3` (score=0.7308) - What causes neutral lipid storage disease with myopathy? Neutral lipid storage disease with myopathy is caused by mutations in the PNPLA2 gene. This gene provides instructions for making an enzyme called adipose trigl...

**Answer**:

Neutral lipid storage disease with myopathy is a rare condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. This abnormal accumulation leads to muscle weakness (myopathy) due to fat buildup in muscle tissue. Other possible features include a fatty liver, weakened and enlarged heart (cardiomyopathy), inflammation of the pancreas (pancreatitis), reduced thyroid activity (hypothyroidism), and type 2 diabetes mellitus. The condition is caused by mutations in the PNPLA2 gene, which results in impaired function of the adipose triglyceride lipase (ATGL) enzyme, leading to reduced breakdown of triglycerides and their subsequent accumulation in tissues. It is inherited in an autosomal recessive pattern, and there is currently no treatment to correct the underlying metabolic problem. Management focuses on dietary guidelines and symptom-related treatments, with emerging evidence suggesting potential benefit from bezafibrate in reducing lipid accumulation and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Scores**: context_relevance=0.800, groundedness=0.954, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0006546-1` (score=0.6267) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. ...
2. `medquad_ans_0000981-1__dup1` (score=0.6055) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been repo...
3. `medquad_ans_0000193-1__dup2` (score=0.6042) - Microcephaly is a medical condition in which the circumference of the head is smaller than normal because the brain has not developed properly or has stopped growing. Microcephaly can be present at birth or it may dev...
4. `medquad_ans_0006546-2` (score=0.5432) - How might a Zika virus infection be treated? There is no vaccine to prevent Zika virus infections, nor is there a specific medicine to treat Zika. Individuals infected with the Zika virus should get plenty of rest, dr...
5. `medquad_ans_0000174-1__dup3` (score=0.4864) - Chikungunya is a virus that spread by the same kinds of mosquitoes that spread dengue and Zika virus. Rarely, it can spread from mother to newborn around the time of birth. It may also possibly spread through infected...

**Answer**:

Zika virus is a virus spread primarily through the bite of an infected mosquito. It can also be transmitted from a pregnant mother to her child during pregnancy or around the time of birth, through sexual contact with an affected male partner, and through blood transfusions. The illness associated with Zika virus infection is usually mild, with symptoms such as fever, rash, joint pain, and conjunctivitis, which typically last for several days to a week. In rare cases, Zika virus infection has been linked to serious birth defects, including microcephaly and other brain anomalies, as well as Guillain-Barre syndrome. There is currently no vaccine or medicine to prevent or treat Zika virus infection [medquad_ans_0006546-1][medquad_ans_0000981-1__dup1].

**Supported citations**: `medquad_ans_0006546-1`, `medquad_ans_0000981-1__dup1`
**Unsupported citations**: _none_

### Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000019-1__dup2` (score=0.7971) - Broken heart syndrome is a condition in which extreme stress can lead to heart muscle failure. The failure is severe, but often short-term. Most people who experience broken heart syndrome think they may be having a h...
2. `medquad_ans_0000019-3__dup2` (score=0.7818) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack...
3. `medquad_ans_0000019-8` (score=0.6744) - Researchers are still learning about broken heart syndrome, and no treatments have been shown to prevent it. For people who have experienced the condition, the risk of recurrence is low. An emotionally upsetting or se...
4. `medquad_ans_0000029-1__dup3` (score=0.6625) - Cardiomyopathy refers to diseases of the heart muscle. These diseases have many causes, signs and symptoms, and treatments. In cardiomyopathy, the heart muscle becomes enlarged, thick, or rigid. In rare cases, the mus...
5. `medquad_ans_0000019-4__dup2` (score=0.6473) - Broken heart syndrome affects women more often than men. Often, people who experience broken heart syndrome have previously been healthy. Research shows that the traditional risk factors for heart disease may not appl...

**Answer**:

The cause of broken heart syndrome is not fully known. In most cases, symptoms are triggered by extreme emotional or physical stress, such as intense grief, anger, or surprise. Researchers think that the stress releases hormones that stun the heart and affect its ability to pump blood to the body. [medquad_ans_0000019-1__dup2]

**Supported citations**: `medquad_ans_0000019-1__dup2`
**Unsupported citations**: _none_

### Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Scores**: context_relevance=1.000, groundedness=0.967, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000005-1__dup5` (score=0.7158) - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. An autoimmune disease is a disorder in which the bodys immune system attacks the bodys own cells and...
2. `medquad_ans_0000005-9__dup1` (score=0.7038) - - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. - Hashimotos disease often leads to reduced thyroid function, or hypothyroidism. Hypothyroidism is...
3. `medquad_ans_0000009-3__dup4` (score=0.6970) - Hypothyroidism has several causes, including - Hashimotos disease - thyroiditis, or inflammation of the thyroid - congenital hypothyroidism, or hypothyroidism that is present at birth - surgical removal of part or all...
4. `medquad_ans_0002735-1` (score=0.6837) - Hashimoto's encephalitis (HE) is a condition characterized by onset of confusion with altered level of consciousness; seizures; and jerking of muscles (myoclonus). Psychosis, including visual hallucinations and parano...
5. `medquad_ans_0000005-3__dup4` (score=0.6255) - Many people with Hashimotos disease have no symptoms at first. As the disease slowly progresses, the thyroid usually enlarges and may cause the front of the neck to look swollen. The enlarged thyroid, called a goiter,...

**Answer**:

Hashimoto thyroiditis is an autoimmune disease in which the body's immune system attacks the thyroid gland, causing inflammation and interfering with its ability to produce thyroid hormones. This condition is also known as chronic lymphocytic thyroiditis or autoimmune thyroiditis. Large numbers of white blood cells called lymphocytes accumulate in the thyroid and produce antibodies that initiate the autoimmune process. Hashimoto thyroiditis often leads to reduced thyroid function, or hypothyroidism, which occurs when the thyroid does not make enough thyroid hormone for the body's needs. Thyroid hormones regulate metabolism and affect nearly every organ in the body. Hashimoto thyroiditis is the most common cause of hypothyroidism in the United States and is much more common in women than men, typically appearing between 30 and 50 years of age [medquad_ans_0000005-1__dup5]medquad_ans_0000005-9\_\_

**Supported citations**: `medquad_ans_0000005-1__dup5`
**Unsupported citations**: _none_

### Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Scores**: context_relevance=1.000, groundedness=0.942, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001043-1` (score=0.7396) - Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). In affected individuals, harmful amounts of lipids accumulate in the spleen, liver, ...
2. `medquad_ans_0006463-1` (score=0.7386) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include...
3. `medquad_ans_0000002-3__dup1` (score=0.6623) - Wolmans disease is usually fatal by age 1. The onset and course of cholesteryl ester storage disease varies, and individuals may live into adulthood.
4. `medquad_ans_0001043-5` (score=0.5959) - These resources address the diagnosis or management of Wolman disease: - Genetic Testing Registry: Lysosomal acid lipase deficiency These resources from MedlinePlus offer information about the diagnosis and management...
5. `medquad_ans_0006463-2` (score=0.5757) - What are the signs and symptoms of Wolman disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Wolman disease. If the information is available, the table below includes how often...

**Answer**:

Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). It is a type of lysosomal storage disorder caused by mutations in the LIPA gene and is inherited in an autosomal recessive manner. The condition leads to a buildup of lipids in body organs such as the spleen, liver, bone marrow, small intestine, adrenal glands, and lymph nodes, along with calcium deposits in the adrenal glands. Infants are typically healthy at birth but soon develop symptoms including enlarged liver and spleen (hepatosplenomegaly), poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor nutrient absorption. The disease is severe and life-threatening, with most affected individuals not surviving past early childhood; mortality typically occurs by age 1 [medquad_ans_0001043-1]medquad_ans

**Supported citations**: `medquad_ans_0001043-1`
**Unsupported citations**: _none_

### Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002753-3` (score=0.7168) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma a...
2. `medquad_ans_0002753-1` (score=0.6833) - Hemangiopericytoma is a term used to described a group of tumors that are derived from pericytes, the cells normally arranged along specific types of blood vessels called capillaries and venules. These types of tumors...
3. `medquad_ans_0002750-1` (score=0.5971) - The term hemangioendothelioma describes several types of vascular neosplasms and includes both non-cancerous (benign) and cancerous (malignant) growths. The term has also been applied to those that show "borderline" b...
4. `medquad_ans_0002753-2` (score=0.5650) - What are the signs and symptoms of Hemangiopericytoma? The Human Phenotype Ontology provides the following list of signs and symptoms for Hemangiopericytoma. If the information is available, the table below includes h...
5. `medquad_ans_0002826-1` (score=0.5619) - Hereditary hemorrhagic telangiectasia (HHT) is an inherited disorder of the blood vessels that can cause excessive bleeding. People with this condition can develop abnormal blood vessels called arteriovenous malformat...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a possible relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

### Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Scores**: context_relevance=0.750, groundedness=0.098, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000396-1__dup2` (score=0.4504) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...
2. `medquad_ans_0000588-3` (score=0.4359) - Mutations in the MT-ND1, MT-ND4, MT-ND4L, or MT-ND6 gene can cause LHON. These genes are found in the DNA of cellular structures called mitochondria, which convert the energy from food into a form that cells can use. ...
3. `medquad_ans_0000181-3__dup2` (score=0.4348) - Alagille syndrome is caused by a gene mutation, or defect. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the DNA sequence that makes up a gene. DNA, or deoxyribon...
4. `medquad_ans_0004086-1` (score=0.4336) - Mitochondrial genetic disorders refer to a group of conditions that affect the mitochondria (the structures in each cell of the body that are responsible for making energy). People with these conditions can present at...
5. `medquad_ans_0000398-1__dup1` (score=0.4312) - Genes are the building blocks of heredity. They are passed from parent to child. They hold DNA, the instructions for making proteins. Proteins do most of the work in cells. They move molecules from one place to anothe...

**Answer**:

The provided context does not contain information about genetic changes related to microphthalmia. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Scores**: context_relevance=0.833, groundedness=0.958, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000028-13` (score=0.6971) - Gout is a form of arthritis that frequently affects joints in the lower part of the body such as the knees, ankles, or toes. The affected joint may become swollen, red, or warm. Attacks usually occur at night. Sometim...
2. `medquad_ans_0000028-1__dup4` (score=0.6481) - Sudden, Intense Joint Pain Gout is a form of arthritis that causes the sudden onset of intense pain and swelling in the joints, which also may be warm and red. Attacks frequently occur at night and can be triggered by...
3. `medquad_ans_0000409-1__dup1` (score=0.6439) - Gout is a common, painful form of arthritis. It causes swollen, red, hot and stiff joints. Gout happens when uric acid builds up in your body. Uric acid comes from the breakdown of substances called purines. Purines a...
4. `medquad_ans_0000028-8` (score=0.6332) - Gout is a form of arthritis that causes sudden onset of intense pain and swelling in the joints, which also may be warm and red. Attacks frequently occur at night and can be triggered by stressful events, alcohol or d...
5. `medquad_ans_0000028-18` (score=0.6062) - The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH has more information about gout. Check out the information here. Also, see "Key Words" related to gout.

**Answer**:

The symptoms of gout include swollen, red, or warm joints, sudden and intense joint pain, attacks that develop in a day, and arthritis affecting only one joint, often the toe, ankle, or knee. Hyperuricemia—high levels of uric acid in the body—is associated with gout, as is the presence of uric acid crystals in joint fluid. More than one attack of acute arthritis may occur, and attacks frequently happen at night [medquad_ans_0000028-13]. [medquad_ans_0000028-1__dup4]

**Supported citations**: `medquad_ans_0000028-13`, `medquad_ans_0000028-1__dup4`
**Unsupported citations**: _none_

### Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Scores**: context_relevance=0.667, groundedness=0.036, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0001903-1` (score=0.6637) - Dopa-responsive dystonia (DRD) is an inherited type of dystonia that typically begins during childhood but may begin in adolescence or adulthood. Depending on the specific type of DRD, specific symptoms can vary. Feat...
2. `medquad_ans_0000307-1__dup1` (score=0.6454) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...
3. `medquad_ans_0003335-1` (score=0.6426) - Juvenile-onset dystonia is a form of dystonia, which is a movement disorder characterized by involuntary muscle contractions that cause repetitive movements and/or abnormal postures. The severity and frequency of the ...
4. `medquad_ans_0000112-1__dup1` (score=0.6362) - The dystonias are movement disorders in which sustained muscle contractions cause twisting and repetitive movements or abnormal postures. The movements, which are involuntary and sometimes painful, may affect a single...
5. `medquad_ans_0002400-1` (score=0.6295) - Focal dystonia is a movement disorder that is localized to a specific part of the body. The dystonias are a group of movement problems characterized by involuntary, sustained muscle contractions, tremors, and other un...

**Answer**:

The provided context does not mention "Dystonia 8" or its specific cause. Therefore, based on the available information, the cause of Dystonia 8 cannot be determined. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Scores**: context_relevance=0.778, groundedness=0.873, answer_relevance=0.511

**Top docs**:

1. `medquad_ans_0000612-2` (score=0.7891) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unk...
2. `medquad_ans_0003719-1` (score=0.7130) - Lysinuric protein intolerance is a metabolic disorder caused by the body's inability to digest and use the amino acids lysine, arginine, and ornithine. Because the body cannot effectively break down these amino acids,...
3. `medquad_ans_0000195-2` (score=0.5237) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0001101-2` (score=0.5021) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
5. `medquad_ans_0000136-2` (score=0.4990) - Researchers estimate that BOR/BO syndrome affects about 1 in 40,000 people.

**Answer**:

Lysinuric protein intolerance affects an estimated 1 in 40,000 to 50,000 people [medquad_ans_0000195-2]. In specific populations, the incidence is 1 in 60,000 in Finland and 1 in 57,000 in Japan [medquad_ans_0000612-2]. The prevalence is estimated to be 1 in 7,500 to 20,000 people [medquad_ans_0001101-2]. However, the exact global incidence is unknown [medquad_ans_0000612-2].

**Supported citations**: `medquad_ans_0000195-2`, `medquad_ans_0000612-2`, `medquad_ans_0001101-2`
**Unsupported citations**: _none_

### Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Scores**: context_relevance=0.875, groundedness=0.937, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000170-10` (score=0.5721) - - Renal artery stenosis (RAS) is the narrowing of one or both renal arteries. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood vessel that carries blood from the heart ...
2. `medquad_ans_0000150-4__dup2` (score=0.5649) - Most people with high blood pressure do not have symptoms. In rare cases, high blood pressure can cause headaches. Kidney disease also does not have symptoms in the early stages. A person may have swelling called edem...
3. `medquad_ans_0000170-1__dup3` (score=0.5646) - Renal artery stenosis is the narrowing of one or both renal arteries. Renal means kidney and stenosis means narrowing. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood ...
4. `medquad_ans_0000145-3__dup3` (score=0.5615) - The symptoms of Goodpasture syndrome may initially include fatigue, nausea, vomiting, and weakness. The lungs are usually affected before or at the same time as the kidneys, and symptoms can include shortness of breat...
5. `medquad_ans_0000071-5__dup1` (score=0.5567) - Because diagnosis is based on blood pressure readings, this condition can go undetected for years, as symptoms do not usually appear until the body is damaged from chronic high blood pressure. Complications of High Bl...

**Answer**:

Renal artery stenosis (RAS) often has no symptoms until it becomes severe. The first symptoms are usually either high blood pressure or decreased kidney function, or both [medquad_ans_0000170-10]. High blood pressure in RAS is typically detected without symptoms in most individuals, though in rare cases, it may cause headaches [medquad_ans_0000150-4__dup2]. Decreased kidney function may present with symptoms such as edema (swelling in the legs, feet, ankles, hands, or face), appetite loss, nausea, vomiting, drowsiness or fatigue, trouble concentrating, sleep problems, increased or decreased urination, generalized itching or numbness, dry skin, headaches, weight loss, darkened skin, muscle cramps, shortness of breath, and chest pain [medquad_ans_0000150-4\_\_dup

**Supported citations**: `medquad_ans_0000170-10`, `medquad_ans_0000150-4__dup2`
**Unsupported citations**: _none_

### Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Scores**: context_relevance=0.800, groundedness=0.987, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0005502-5` (score=0.6760) - How might Schnitzler syndrome be treated? The treatment of Schnitzler syndrome is aimed at alleviating the signs and symptoms associated with the condition. The following medications have been used with variable succe...
2. `medquad_ans_0000712-5` (score=0.6312) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis -...
3. `medquad_ans_0004303-4` (score=0.6246) - Is there a treatment for Naegeli syndrome? Treatment for Naegeli syndrome is based on an individual's symptoms. Dry skin can be moisturized with creams. To avoid overheating, affected individuals should wear appropria...
4. `medquad_ans_0000712-1` (score=0.6186) - Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis (NFJS/DPR) represents a rare type of ectodermal dysplasia, a group of about 150 conditions characterized by abnormal development of ectoder...
5. `medquad_ans_0002049-6` (score=0.6184) - How might Ehlers-Danlos syndrome be treated? There is no specific cure for Ehlers-Danlos syndrome (EDS). The treatment and management is focused on preventing serious complications and relieving associated signs and s...

**Answer**:

Treatment for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis (NFJS/DPR) is based on an individual's symptoms. Dry skin can be moisturized with creams. To avoid overheating, affected individuals should wear appropriate clothing and use wet dressings. Dental care is needed to treat cavities and tooth loss [medquad_ans_0004303-4].

**Supported citations**: `medquad_ans_0004303-4`
**Unsupported citations**: _none_

### Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.7876) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0000353-5` (score=0.6919) - These resources address the diagnosis or management of familial hyperaldosteronism: - Genetic Testing Registry: Familial hyperaldosteronism type 1 - Genetic Testing Registry: Familial hyperaldosteronism type 3 - Hormo...
3. `medquad_ans_0005138-2` (score=0.6850) - What are the signs and symptoms of Pseudohypoaldosteronism type 2? The most common symptom of pseudohypoaldosteronism type 2 is high blood pressure in adolescents or young adults. In its most severe form, it is associ...
4. `medquad_ans_0000353-1` (score=0.6624) - Familial hyperaldosteronism is a group of inherited conditions in which the adrenal glands, which are small glands located on top of each kidney, produce too much of the hormone aldosterone. Aldosterone helps control ...
5. `medquad_ans_0002593-2` (score=0.6589) - What are the signs and symptoms of Glucocorticoid-remediable aldosteronism? The Human Phenotype Ontology provides the following list of signs and symptoms for Glucocorticoid-remediable aldosteronism. If the informatio...

**Answer**:

The symptoms of Familial hyperaldosteronism type III include hypercalciuria (5%), metabolic acidosis (5%), polydipsia (5%), polyuria (5%), adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

**Supported citations**: `medquad_ans_0002259-1`
**Unsupported citations**: _none_

### Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Scores**: context_relevance=1.000, groundedness=0.117, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004530-2` (score=0.5985) - Is olivopontocerebellar atrophy inherited? Olivopontocerebellar atrophy (OPCA) may be associated with conditions that are inherited (genetic), or it may occur sporadically. Genetic forms of OPCA may be inherited in an...
2. `medquad_ans_0004750-4` (score=0.5471) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is pass...
3. `medquad_ans_0000673-4` (score=0.5409) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
4. `medquad_ans_0000700-4` (score=0.5315) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
5. `medquad_ans_0000786-4` (score=0.5310) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

The provided context does not contain information about the inheritance pattern of Spinocerebellar ataxia 11. Therefore, based on the given context, it is not possible to determine whether Spinocerebellar ataxia 11 is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Scores**: context_relevance=1.000, groundedness=0.918, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000503-2` (score=0.8110) - Hypochromic microcytic anemia with iron overload is likely a rare disorder; at least five affected families have been reported in the scientific literature.
2. `medquad_ans_0003040-1` (score=0.7952) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload....
3. `medquad_ans_0000503-1` (score=0.7552) - Hypochromic microcytic anemia with iron overload is a condition that impairs the normal transport of iron in cells. Iron is an essential component of hemoglobin, which is the substance that red blood cells use to carr...
4. `medquad_ans_0000503-5` (score=0.7519) - These resources address the diagnosis or management of hypochromic microcytic anemia with iron overload: - Genetic Testing Registry: Hypochromic microcytic anemia with iron overload These resources from MedlinePlus of...
5. `medquad_ans_0000185-10` (score=0.7183) - - Hemochromatosis is the most common form of iron overload disease. Too much iron in the body causes hemochromatosis. - Inherited genetic defects cause primary hemochromatosis. - Primary hemochromatosis mainly affects...

**Answer**:

The symptoms of Hypochromic microcytic anemia with iron overload include pale skin (pallor), tiredness (fatigue), and slow growth. Additionally, the condition leads to anemia characterized by abnormally small (microcytic) and pale (hypochromic) red blood cells. Iron accumulation in the liver can impair its function over time, with liver problems typically becoming apparent in adolescence or early adulthood [medquad_ans_0000503-1].

**Supported citations**: `medquad_ans_0000503-1`
**Unsupported citations**: _none_

### Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Scores**: context_relevance=0.600, groundedness=0.748, answer_relevance=0.720

**Top docs**:

1. `medquad_ans_0000698-1__dup2` (score=0.5751) - Pemphigus is an autoimmune disorder. If you have it, your immune system attacks healthy cells in your skin and mouth, causing blisters and sores. No one knows the cause. Pemphigus does not spread from person to person...
2. `medquad_ans_0000405-4` (score=0.5003) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0001090-4` (score=0.5003) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000741-4` (score=0.5003) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000891-4` (score=0.5003) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention "benign chronic pemphigus" or specify whether it is inherited. While the context states that pemphigus is an autoimmune disorder and does not spread from person to person or appear to be inherited, it also notes that "some people's genes put them more at risk for pemphigus" [medquad_ans_0000698-1__dup2]. However, this does not confirm inheritance of a specific form like "benign chronic pemphigus." Therefore, based on the available information, it cannot be determined if benign chronic pemphigus is inherited.

Answer: Insufficient context to determine if benign chronic pemphigus is inherited.

**Supported citations**: `medquad_ans_0000698-1__dup2`
**Unsupported citations**: _none_

### Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.962, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000916-4__dup1` (score=0.8072) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is...
2. `medquad_ans_0000916-1__dup1` (score=0.6847) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
3. `medquad_ans_0000915-1__dup1` (score=0.6847) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
4. `medquad_ans_0004784-4` (score=0.5789) - How is Pelizaeus-Merzbacher disease inherited?
5. `medquad_ans_0002823-3` (score=0.5738) - How is hereditary endotheliopathy with retinopathy, nephropathy, and stroke (HERNS) inherited? Hereditary endotheliopathy with retinopathy, nephropathy, and stroke (HERNS) is inherited in an autosomal dominant manner....

**Answer**:

Yes, Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a mutation in only one copy of the responsible gene in each cell is sufficient to cause the condition. [medquad_ans_0000916-4__dup1]

**Supported citations**: `medquad_ans_0000916-4__dup1`
**Unsupported citations**: _none_

### Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Scores**: context_relevance=0.700, groundedness=0.882, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0002895-1` (score=0.6625) - Histiocytosis-lymphadenopathy plus syndrome is a group of conditions with overlapping signs and symptoms that affect many parts of the body. This group of disorders includes H syndrome, pigmented hypertrichosis with i...
2. `medquad_ans_0006446-4` (score=0.5663) - Is Williams syndrome inherited?
3. `medquad_ans_0000673-4` (score=0.5317) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
4. `medquad_ans_0002533-4` (score=0.5249) - How is Gerstmann-Straussler-Scheinker disease inherited? Gerstmann-Straussler-Scheinker disease (GSS) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutatio...
5. `medquad_ans_0001030-4` (score=0.5196) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene and occur in pe...

**Answer**:

The genetic changes related to histiocytosis-lymphadenopathy plus syndrome are mutations in the SLC29A3 gene [medquad_ans_0002895-1].

**Supported citations**: `medquad_ans_0002895-1`
**Unsupported citations**: _none_

### Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.052, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000261-4` (score=0.5760) - In most cases of cyclic vomiting syndrome, affected people have no known history of the disorder in their family. However, many affected individuals have a family history of related conditions, such as migraines, irri...
2. `medquad_ans_0000646-4` (score=0.5572) - Menkes syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome)...
3. `medquad_ans_0000139-2__dup2` (score=0.5538) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...
4. `medquad_ans_0000557-4` (score=0.5485) - This condition is generally not inherited but arises from mutations in the body's cells that occur after conception. This alteration is called a somatic mutation and is present only in certain cells. Rarely, this cond...
5. `medquad_ans_0003743-4` (score=0.5327) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition w...

**Answer**:

The provided context does not contain information about whether Noonan syndrome is inherited. Therefore, based on the given documents, it is not possible to determine if Noonan syndrome is inherited. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Scores**: context_relevance=0.750, groundedness=0.782, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0001093-2` (score=0.8090) - 3-methylglutaconyl-CoA hydratase deficiency is a rare disorder; at least 20 cases have been reported in the scientific literature.
2. `medquad_ans_0001093-5` (score=0.6877) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Aci...
3. `medquad_ans_0001093-1` (score=0.6489) - 3-methylglutaconyl-CoA hydratase deficiency is an inherited condition that causes neurological problems. Beginning in infancy to early childhood, children with this condition often have delayed development of mental a...
4. `medquad_ans_0001093-3` (score=0.6256) - Mutations in the AUH gene cause 3-methylglutaconyl-CoA hydratase deficiency. This gene provides instructions for producing 3-methylglutaconyl-CoA hydratase, an enzyme that is involved in breaking down a protein buildi...
5. `medquad_ans_0006372-6` (score=0.5934) - How might VLCAD deficiency be treated? Management of VLCAD deficiency depends on many factors, including the form of the condition and the specific signs and symptoms present. For example, people affected by the sever...

**Answer**:

The provided context does not mention any specific treatments for 3-methylglutaconyl-CoA hydratase deficiency. While it discusses diagnostic resources and symptoms, including elevated levels of 3-methylglutaconic acid and metabolic acidosis, there is no information on therapeutic interventions or management strategies. [medquad_ans_0001093-5] lists resources related to diagnosis and management, but does not specify treatments. Therefore, based on the given context, the treatments for this condition cannot be determined.

**Supported citations**: `medquad_ans_0001093-5`
**Unsupported citations**: _none_

### Example 61 (`0003000-1`)

**Question**: What is (are) Hyperkalemic periodic paralysis ?

**Scores**: context_relevance=0.833, groundedness=0.981, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.8100) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0000494-5` (score=0.7540) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis ...
3. `medquad_ans_0000126-1__dup1` (score=0.7256) - Familial periodic paralyses are a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. They are characterized by episodes in which the affect...
4. `medquad_ans_0000494-1` (score=0.6981) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to...
5. `medquad_ans_0003000-1` (score=0.6920) - Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move...

**Answer**:

Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms and legs. Episodes tend to increase in frequency until mid-adulthood, after which they occur less frequently. Factors that can trigger attacks include rest after exercise, potassium-rich foods such as bananas and potatoes, stress, fatigue, alcohol, pregnancy, exposure to cold temperatures, certain medications, and periods without food (fasting). Muscle strength usually returns to normal between attacks, although many affected people continue to experience mild stiffness (myotonia), particularly in [medquad_ans_0000126-1__dup1]. It is estimated to affect 1 in 200,000 people [medquad_ans_0000494-2].

**Supported citations**: `medquad_ans_0000126-1__dup1`, `medquad_ans_0000494-2`
**Unsupported citations**: _none_

### Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Scores**: context_relevance=0.800, groundedness=0.828, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000225-5` (score=0.7008) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of...
2. `medquad_ans_0000225-1` (score=0.6426) - Congenital fibrosis of the extraocular muscles is a disorder that affects the muscles that surround the eyes. These muscles control eye movement and the position of the eyes (for example, looking straight ahead). Cong...
3. `medquad_ans_0000225-2` (score=0.6416) - CFEOM1 is the most common form of congenital fibrosis of the extraocular muscles, affecting at least 1 in 230,000 people. CFEOM1 and CFEOM3 have been reported worldwide, whereas CFEOM2 has been seen in only a few fami...
4. `medquad_ans_0002381-3` (score=0.5787) - How might Filippi syndrome be treated? The treatment of Filippi syndrome is directed toward the specific symptoms that are apparent in each individual. Treatment may require the coordinated efforts of a team of medica...
5. `medquad_ans_0002374-1` (score=0.5552) - Fibrous dysplasia is a skeletal disorder that is characterized by the replacement of normal bone with fibrous bone tissue. It may involve one bone (monostotic) or multiple bones (polyostotic). Fibrous dysplasia can af...

**Answer**:

The provided context does not specify treatments for congenital fibrosis of the extraocular muscles. While it mentions resources related to diagnosis and management, such as genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, palliative care, and strabismus, it does not detail specific treatment approaches for congenital fibrosis of the extraocular muscles [medquad_ans_0000225-5]. Therefore, based on the available information, the treatments for congenital fibrosis of the extraocular muscles cannot be determined.

**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

### Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Scores**: context_relevance=0.800, groundedness=0.520, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000152-1__dup1` (score=0.7305) - Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. Symptoms of the disease, which gets progressiv...
2. `medquad_ans_0002944-1` (score=0.7147) - Huntington disease (HD) is an inherited condition that causes progressive degeneration of neurons in the brain. Signs and symptoms usually develop between ages 35 to 44 years and may include uncontrolled movements, lo...
3. `medquad_ans_0000491-1__dup2` (score=0.7106) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms...
4. `medquad_ans_0000152-3__dup1` (score=0.6920) - Huntingtons disease causes disability that gets worse over time. People with this disease usually die within 15 to 20 years following diagnosis. At this time, no treatment is available to slow, stop or reverse the cou...
5. `medquad_ans_0000489-1` (score=0.6763) - Huntington disease is a progressive brain disorder that causes uncontrolled movements, emotional problems, and loss of thinking ability (cognition). Adult-onset Huntington disease, the most common form of this disorde...

**Answer**:

Huntington's disease (HD) is an inherited disorder that causes progressive degeneration of brain cells, called neurons, in motor control regions of the brain and other areas. It is caused by a mutation in the gene for a protein called huntingtin, leading to an abnormal repetition of cytosine, adenine, and guanine (CAG) building blocks in DNA. The disease typically begins between ages 30 and 50, with an earlier onset form called juvenile HD occurring under age 20. Symptoms include uncontrolled movements (chorea), abnormal body postures, impaired coordination, slurred speech, difficulty feeding and swallowing, changes in behavior, emotion, judgment, and cognition. HD is inherited in an autosomal dominant manner, meaning each child of a parent with HD has a 50-50 chance of inheriting the mutated gene. A person who inherits the HD gene will eventually develop the disease. Diagnosis is aided

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.556

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.8201) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
2. `medquad_ans_0002601-2` (score=0.6823) - What are the signs and symptoms of Glutamate formiminotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Glutamate formiminotransferase deficiency. If the inform...
3. `medquad_ans_0002601-1` (score=0.6530) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...
4. `medquad_ans_0000416-1` (score=0.6453) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with ...
5. `medquad_ans_0000416-5` (score=0.6408) - These resources address the diagnosis or management of glutamate formiminotransferase deficiency: - Baby's First Test - Genetic Testing Registry: Glutamate formiminotransferase deficiency These resources from MedlineP...

**Answer**:

Approximately 20 affected individuals have been identified with glutamate formiminotransferase deficiency [medquad_ans_0000416-2].

**Supported citations**: `medquad_ans_0000416-2`
**Unsupported citations**: _none_

### Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000738-2` (score=0.8692) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
2. `medquad_ans_0000738-1` (score=0.6507) - Noonan syndrome is a condition that affects many areas of the body. It is characterized by mildly unusual facial characteristics, short stature, heart defects, bleeding problems, skeletal malformations, and many other...
3. `medquad_ans_0004452-2` (score=0.6459) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004453-3` (score=0.6459) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004451-3` (score=0.6459) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people [medquad_ans_0000738-2].

**Supported citations**: `medquad_ans_0000738-2`
**Unsupported citations**: _none_

### Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Scores**: context_relevance=0.571, groundedness=0.989, answer_relevance=0.571

**Top docs**:

1. `medquad_ans_0000339-5__dup1` (score=0.7285) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on t...
2. `medquad_ans_0000339-1__dup3` (score=0.7215) - Also called crab lice or "crabs," pubic lice are parasitic insects found primarily in the pubic or genital area of humans. Pubic lice infestation is found worldwide and occurs in all races, ethnic groups, and levels o...
3. `medquad_ans_0000339-2__dup1` (score=0.6662) - Pubic ("crab") lice infestation is found worldwide and occurs in all races and ethnic groups and in all levels of society. Pubic lice usually are spread through sexual contact and are most common in adults. Occasional...
4. `medquad_ans_0000557-1__dup2` (score=0.6646) - Summary : Lice are parasitic insects that can be found on people's heads and bodies. They survive by feeding on human blood. Lice found on each area of the body are different from each other. The three types of lice t...
5. `medquad_ans_0000339-7` (score=0.6448) - Pubic ("crab") lice most commonly are spread directly from person to person by sexual contact. Pubic lice very rarely may be spread by clothing, bedding, or a toilet seat. The following are steps that can be taken to ...

**Answer**:

Pubic lice infestation is diagnosed by finding a “crab” louse or eggs (nits) on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp) [medquad_ans_0000339-5__dup1]. Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

### Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.938, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005571-3` (score=0.8232) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
2. `medquad_ans_0005571-1` (score=0.6935) - Shapiro syndrome is a rare disease affecting about 50 people worldwide that is typically characterized by recurrent episodes of excessive sweating and hypothermia and the agenesis of the corpus callosum. The duration ...
3. `medquad_ans_0000860-2__dup1` (score=0.5907) - How might Brown-Sequard syndrome be treated?
4. `medquad_ans_0005571-2` (score=0.5593) - What are the signs and symptoms of Shapiro syndrome? Shapiro syndrome generally consists of three findings: spontaneous periodic hypothermia, excessive sweating, and agenesis of the corpus callosum. However, there has...
5. `medquad_ans_0002533-6` (score=0.5560) - How might Gerstmann-Straussler-Scheinker disease be treated? The treatment of Gerstmann-Straussler-Scheinker disease (GSS) is based on the signs and symptoms present in each person. There is currently no cure for the ...

**Answer**:

Treatment options for Shapiro syndrome that have been attempted include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, and sympathectomy. However, these treatments have resulted in varying responses, and some have been unsuccessful or of doubtful efficacy due to the small number of diagnosed cases [medquad_ans_0005571-3]. The cause of Shapiro syndrome is unknown, and therefore, effective treatment remains challenging. It is recommended that treatment options be discussed with a healthcare provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

### Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.964, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002359-1` (score=0.7685) - FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. People wi...
2. `medquad_ans_0000373-1` (score=0.6945) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects ...
3. `medquad_ans_0002359-3` (score=0.6865) - How might FG syndrome be treated? Treatment is aimed at addressing the individual symptoms present in each case. This often involves care by a team of providers which may include pediatricians, neurologists, cardiolog...
4. `medquad_ans_0000373-5` (score=0.6245) - These resources address the diagnosis or management of FG syndrome: - Gene Review: Gene Review: MED12-Related Disorders - Genetic Testing Registry: FG syndrome - Genetic Testing Registry: FG syndrome 2 - Genetic Testi...
5. `medquad_ans_0000373-2` (score=0.6120) - The prevalence of FG syndrome is unknown, although several hundred cases have been reported worldwide. Researchers suspect that FG syndrome may be overdiagnosed because many of its signs and symptoms are also seen wit...

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. It is characterized by intellectual disability ranging from mild to severe, hypotonia, constipation and/or anal anomalies, a distinctive facial appearance, broad thumbs and great toes, relative macrocephaly, and abnormalities of the corpus callosum. Medical problems such as heart defects, seizures, undescended testicles, and inguinal hernia have also been reported. The condition is inherited in an X-linked recessive pattern, and mutations in genes such as MED12 (FGS1), FLNA (FGS2), CASK (FGS4), UPF3B (FGS6), and BRWD3 (FGS7) are associated with the disorder [medquad_ans_0002359-1]. Affected individuals often

**Supported citations**: `medquad_ans_0002359-1`
**Unsupported citations**: _none_

### Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006242-1` (score=0.7039) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0001335-1` (score=0.6423) - What are the signs and symptoms of Chudley Rozdilsky syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Chudley Rozdilsky syndrome. If the information is available, the table ...
3. `medquad_ans_0000676-2__dup1` (score=0.6410) - What are the symptoms of Bell's palsy?
4. `medquad_ans_0003339-2` (score=0.6332) - What are the signs and symptoms of Kallmann syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kallmann syndrome. If the information is available, the table below includes how...
5. `medquad_ans_0003459-2` (score=0.6246) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficu...

**Answer**:

The symptoms of Tukel syndrome include carpal bone aplasia, carpal synostosis, compensatory chin elevation, congenital fibrosis of extraocular muscles, nonprogressive restrictive external ophthalmoplegia, postaxial oligodactyly, ptosis, restrictive external ophthalmoplegia, syndactyly [medquad_ans_0006242-1].

**Supported citations**: `medquad_ans_0006242-1`
**Unsupported citations**: _none_

### Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Scores**: context_relevance=0.500, groundedness=0.670, answer_relevance=0.633

**Top docs**:

1. `medquad_ans_0000199-1__dup1` (score=0.6185) - The mucolipidoses (ML) are a group of inherited metabolic diseases that affect the bodys ability to carry out the normal turnover of various materials within cells. In ML, abnormal amounts of carbohydrates and fatty m...
2. `medquad_ans_0000674-5` (score=0.5980) - These resources address the diagnosis or management of mucolipidosis II alpha/beta: - Gene Review: Gene Review: Mucolipidosis II - Genetic Testing Registry: I cell disease - MedlinePlus Encyclopedia: Clubfoot - Medlin...
3. `medquad_ans_0004153-1` (score=0.5937) - Mucolipidosis type 4 is a metabolic condition that affects the body's ability to process certain carbohydrates and fats. As a result, these materials accumulate in cells leading to the various signs and symptoms of th...
4. `medquad_ans_0000677-5` (score=0.5715) - These resources address the diagnosis or management of mucolipidosis type IV: - Gene Review: Gene Review: Mucolipidosis IV - Genetic Testing Registry: Ganglioside sialidase deficiency - MedlinePlus Encyclopedia: Gastr...
5. `medquad_ans_0000676-5` (score=0.5515) - These resources address the diagnosis or management of mucolipidosis III gamma: - Gene Review: Gene Review: Mucolipidosis III Gamma - Genetic Testing Registry: Mucolipidosis III Gamma - MedlinePlus Encyclopedia: Cloud...

**Answer**:

The provided context does not mention any specific treatments for mucolipidoses. While it describes the disease mechanisms, symptoms, and genetic basis, it does not include information on therapeutic interventions or management strategies. Therefore, based on the given context, treatments for mucolipidoses cannot be determined.

[medquad_ans_0000199-1__dup1]

**Supported citations**: `medquad_ans_0000199-1__dup1`
**Unsupported citations**: _none_

### Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001157-1` (score=0.7136) - What are the signs and symptoms of Cholestasis, progressive familial intrahepatic 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Cholestasis, progressive familial intrahepatic 4....
2. `medquad_ans_0005084-2` (score=0.7091) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis...
3. `medquad_ans_0000118-1` (score=0.6843) - Benign recurrent intrahepatic cholestasis (BRIC) is characterized by episodes of liver dysfunction called cholestasis. During these episodes, the liver cells have a reduced ability to release a digestive fluid called ...
4. `medquad_ans_0005084-1` (score=0.6836) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching...
5. `medquad_ans_0000118-5` (score=0.6746) - These resources address the diagnosis or management of benign recurrent intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Benign recurrent intrahepatic cholestasis 1 -...

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 include autosomal recessive inheritance, cirrhosis, conjugated hyperbilirubinemia, death in childhood, diarrhea, elevated alkaline phosphatase, failure to thrive, fat malabsorption, hepatocellular carcinoma, hepatomegaly, infantile onset, intermittent jaundice, and intrahepatic cholestasis [medquad_ans_0005084-2].

**Supported citations**: `medquad_ans_0005084-2`
**Unsupported citations**: _none_

### Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Scores**: context_relevance=0.800, groundedness=0.796, answer_relevance=0.720

**Top docs**:

1. `medquad_ans_0000099-2__dup2` (score=0.6859) - Nearly 18 million U.S. adultsabout one in 12have fecal incontinence.1 People of any age can have a bowel control problem, though fecal incontinence is more common in older adults. Fecal incontinence is slightly more c...
2. `medquad_ans_0000211-6` (score=0.6835) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he i...
3. `medquad_ans_0000517-2` (score=0.6350) - Incontinentia pigmenti is an uncommon disorder. Between 900 and 1,200 affected individuals have been reported in the scientific literature. Most of these individuals are female, but several dozen males with incontinen...
4. `medquad_ans_0000211-1__dup3` (score=0.6187) - Urinary incontinence is the loss of bladder control, resulting in the accidental leakage of urine from the body. For example, a man may feel a strong, sudden need, or urgency, to urinate just before losing a large amo...
5. `medquad_ans_0000211-4__dup2` (score=0.5992) - Urinary incontinence in men results when the brain does not properly signal the bladder, the sphincters do not squeeze strongly enough, or both. The bladder muscle may contract too much or not enough because of a prob...

**Answer**:

Urinary incontinence occurs in 11 to 34 percent of older men [medquad_ans_0000211-6]. Two to 11 percent of older men report daily urinary incontinence [medquad_ans_0000211-6]. However, the exact number of individuals affected is not specified in the provided context. Therefore, a precise total number of people affected by urinary incontinence in men cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000211-6`
**Unsupported citations**: _none_

### Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.476, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000882-4` (score=0.7549) - As discussed above, Robinow syndrome can have either an autosomal recessive or an autosomal dominant pattern of inheritance. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. T...
2. `medquad_ans_0000882-1` (score=0.7185) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
3. `medquad_ans_0000882-2` (score=0.7131) - Both the autosomal recessive and autosomal dominant forms of Robinow syndrome are rare. Fewer than 200 people with autosomal recessive Robinow syndrome have been described in the medical literature. This form of the c...
4. `medquad_ans_0005415-1` (score=0.6741) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
5. `medquad_ans_0000882-5` (score=0.6684) - These resources address the diagnosis or management of Robinow syndrome: - Gene Review: Gene Review: Autosomal Dominant Robinow Syndrome - Gene Review: Gene Review: ROR2-Related Robinow Syndrome - Genetic Testing Regi...

**Answer**:

Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. It is characterized by skeletal abnormalities, distinctive facial features, and other developmental issues. The syndrome is classified into two major types based on inheritance pattern: autosomal recessive and autosomal dominant. Autosomal recessive Robinow syndrome is marked by skeletal abnormalities such as shortening of long bones, brachydactyly, hemivertebrae leading to spinal curvature, fused or missing ribs, and short stature. It also presents with fetal-like facial features, including a broad forehead, widely spaced eyes, a short nose with an upturned tip, a wide nasal bridge, and a broad triangular mouth. Additional features include underdeveloped genitalia, dental problems, and possible kidney and heart defects. Delayed development occurs in 10 to 15 percent of individuals, though intelligence is usually normal. In contrast, autosomal dominant Robin

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 74 (`0000551-1`)

**Question**: What is (are) juvenile primary lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0003331-1` (score=0.8390) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
2. `medquad_ans_0000551-2` (score=0.8229) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.
3. `medquad_ans_0003319-1` (score=0.7541) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. U...
4. `medquad_ans_0000551-1` (score=0.7517) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. The features of this disorder are caused by damage to moto...
5. `medquad_ans_0003331-2` (score=0.6833) - What are the signs and symptoms of Juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. ...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. It is caused by damage to motor neurons, which are specialized nerve cells in the brain and spinal cord that control muscle movement. Symptoms begin in early childhood and progress over a period of 15 to 20 years. Early symptoms include clumsiness, muscle weakness and spasticity in the legs, and difficulty with balance. As symptoms progress, they include weakness and stiffness in the arms and hands, slurred speech, drooling, difficulty swallowing, and an inability to walk. The disorder is inherited in an autosomal recessive pattern and is caused by mutations in the ALS2 gene [medquad_ans_0003331-1]. Few cases have been reported, highlighting its rarity [medquad_ans_0000551-2].

**Supported citations**: `medquad_ans_0003331-1`, `medquad_ans_0000551-2`
**Unsupported citations**: _none_

### Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Scores**: context_relevance=0.750, groundedness=0.958, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000040-9` (score=0.7054) - Leukemia is a cancer of the blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white ...
2. `medquad_ans_0000010-1__dup4` (score=0.6542) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. Thes...
3. `medquad_ans_0000188-1__dup2` (score=0.6391) - Leukemia is cancer of the white blood cells. It is the most common type of childhood cancer. Your blood cells form in your bone marrow. White blood cells help your body fight infection. In leukemia, the bone marrow pr...
4. `medquad_ans_0000040-1__dup3` (score=0.6355) - Leukemia is a cancer of the blood cells. It is the most common type of blood cancer and affects 10 times as many adults as children. Most people diagnosed with leukemia are over 50 years old. Leukemia Starts in Bone M...
5. `medquad_ans_0000009-1__dup3` (score=0.6150) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. Thes...

**Answer**:

Leukemia is a cancer of the white blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white blood cells builds up in the blood, they crowd out healthy blood cells, making it difficult for the blood to carry out its normal functions [medquad_ans_0000040-9]. Leukemia is the most common type of childhood cancer and can develop quickly or slowly, with acute leukemia being a fast-growing type and chronic leukemia growing slowly [medquad_ans_0000188-1__dup2]. In acute myeloid leukemia (AML), there are too many myeloblasts, a specific type of white blood cell, and AML is the most common type of acute leukemia in adults [medquad_ans_0000010-1\_\_

**Supported citations**: `medquad_ans_0000040-9`, `medquad_ans_0000188-1__dup2`
**Unsupported citations**: _none_

### Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Scores**: context_relevance=0.500, groundedness=0.044, answer_relevance=0.300

**Top docs**:

1. `medquad_ans_0000135-1__dup1` (score=0.5870) - The gangliosidoses are a group of inherited metabolic diseases caused by a deficiency of the different proteins needed to break down fatty substances called lipids. Excess buildup of these fatty materials (oils, waxes...
2. `medquad_ans_0004525-1` (score=0.5725) - Oligoastrocytoma is a brain tumor that forms when two types of cells in the brain, called oligodendrocytes and astrocytes, rapidly increase in number to form a mass. These brain cells are known as glial cells, which n...
3. `medquad_ans_0000737-1` (score=0.5526) - Paraganglioma is a type of noncancerous (benign) tumor that occurs in structures called paraganglia. Paraganglia are groups of cells that are found near nerve cell bunches called ganglia. Paragangliomas are usually fo...
4. `medquad_ans_0002629-1` (score=0.5518) - GM1 gangliosidosis is an inherited lysosomal storage disorder that progressively destroys nerve cells (neurons) in the brain and spinal cord. The condition may be classified into three major types based on the general...
5. `medquad_ans_0002627-1` (score=0.5518) - GM1 gangliosidosis is an inherited lysosomal storage disorder that progressively destroys nerve cells (neurons) in the brain and spinal cord. The condition may be classified into three major types based on the general...

**Answer**:

The provided context does not mention "Gangliocytoma" or any condition with that name. Therefore, based on the given information, it is not possible to answer the question.

Answer: Insufficient context [medquad_ans_0000135-1, medquad_ans_0004525-1]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000135-1, medquad_ans_0004525-1`

### Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.321, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000130-1__dup1` (score=0.8360) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of ...
2. `medquad_ans_0004068-1` (score=0.6781) - Miller syndrome is a rare condition that mainly affects the development of the face and limbs. Characteristic features include underdeveloped cheek bones, a very small lower jaw, cleft lip and/or palate, abnormalities...
3. `medquad_ans_0000658-2` (score=0.6567) - Miller syndrome is a rare disorder; it is estimated to affect fewer than 1 in 1 million newborns. At least 30 cases have been reported in the medical literature.
4. `medquad_ans_0000130-2__dup1` (score=0.6541) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportiv...
5. `medquad_ans_0004069-1` (score=0.6377) - Miller-Dieker syndrome is a genetic condition characterized by lissencephaly, typical facial features, and severe neurologic abnormalities. Symptoms may include severe intellectual disability, developmental delay, sei...

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, absence of tendon reflexes, generalized muscle weakness, and may be preceded by a viral illness. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder. Treatment is identical to that for Guillain-Barr syndrome, involving intravenous immunoglobulin (IVIg) or plasmapheresis and supportive care [medquad_ans_0000130-1__dup1, medquad_ans_0000130-2__dup1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000130-1__dup1, medquad_ans_0000130-2__dup1`

### Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0001084-1` (score=0.8010) - 22q11.2 deletion syndrome (which is also known by several other names, listed below) is a disorder caused by the deletion of a small piece of chromosome 22. The deletion occurs near the middle of the chromosome at a l...
2. `medquad_ans_0006570-3` (score=0.7935) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location desig...
3. `medquad_ans_0001087-3` (score=0.7859) - 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at a location designated 2q37. The size...
4. `medquad_ans_0006579-3` (score=0.7816) - What causes 2q37 deletion syndrome? 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at ...
5. `medquad_ans_0001086-2` (score=0.7723) - At least 500 cases of 22q13.3 deletion syndrome are known.

**Answer**:

22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in the region designated q11.2 [medquad_ans_0001084-1].

**Supported citations**: `medquad_ans_0001084-1`
**Unsupported citations**: _none_

### Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Scores**: context_relevance=1.000, groundedness=0.876, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000062-12` (score=0.7562) - Shingles is caused by a virus called the varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores and other symptoms healed, but the virus remained....
2. `medquad_ans_0000062-2__dup3` (score=0.7539) - Caused By A Virus Shingles is caused by a virus called varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores healed and the other symptoms went a...
3. `medquad_ans_0000062-7__dup1` (score=0.7300) - Shingles -- also called varicella-zoster -- is a painful skin disease caused by a reactivation of the chickenpox virus. It is distinctive because it affects only one side of the body. The early signs of shingles usual...
4. `medquad_ans_0000148-1__dup2` (score=0.7274) - Shingles (herpes zoster) is an outbreak of rash or blisters on the skin that is caused by the same virus that causes chickenpox the varicella-zoster virus. The first sign of shingles is often burning or tingling pain ...
5. `medquad_ans_0000062-11` (score=0.6893) - The most common complication of shingles is pain -- a condition called post-herpetic neuralgia (PHN). People with PHN have severe pain in the areas where they had the shingles rash, even after the rash clears up. In m...

**Answer**:

Shingles is caused by the varicella-zoster virus, the same virus that causes chickenpox. After recovering from chickenpox, the virus remains in the body for life, hiding in nerve cells, usually in the spine. It can become active again when the immune system weakens, which may happen due to aging, certain medical conditions (such as cancer, HIV/AIDS, or transplant operations), or stress. When reactivated, the virus travels along a nerve path (dermatome) to the skin, causing a painful rash [medquad_ans_0000062-12][medquad_ans_0000062-2__dup3].

**Supported citations**: `medquad_ans_0000062-12`, `medquad_ans_0000062-2__dup3`
**Unsupported citations**: _none_

### Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.977, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002456-1` (score=0.7997) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the ta...
2. `medquad_ans_0002458-2` (score=0.6774) - What are the signs and symptoms of Fryns syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns syndrome. If the information is available, the table below includes how often...
3. `medquad_ans_0000333-1` (score=0.6757) - Fabry disease is an inherited disorder that results from the buildup of a particular type of fat, called globotriaosylceramide, in the body's cells. Beginning in childhood, this buildup causes signs and symptoms that ...
4. `medquad_ans_0002200-2` (score=0.6566) - What are the signs and symptoms of Fabry disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Fabry disease. If the information is available, the table below includes how often t...
5. `medquad_ans_0002458-1` (score=0.6506) - Fryns syndrome is a condition that affects the development of many parts of the body. Signs and symptoms vary widely among affected individuals. Many affected individuals have a defect in the diaphragm muscle such as ...

**Answer**:

The symptoms of Fryns Hofkens Fabry syndrome include ulnar deviation of the finger (90%), distal ulnar hypoplasia, dysplastic radii, hypoplasia of the radius, mesomelic arm shortening, radial bowing, and ulnar deviation of the hand [medquad_ans_0002456-1].

**Supported citations**: `medquad_ans_0002456-1`
**Unsupported citations**: _none_

### Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Scores**: context_relevance=0.600, groundedness=0.931, answer_relevance=0.440

**Top docs**:

1. `medquad_ans_0000050-8` (score=0.6594) - Scientists are pursuing a wide range of basic and clinical studies on osteoporosis. Significant advances in preventing and treating osteoporosis continue to be made. Such advances are the direct result of research foc...
2. `medquad_ans_0000552-5` (score=0.5825) - These resources address the diagnosis or management of juvenile primary osteoporosis: - Lucile Packard Children's Hospital at Stanford: Juvenile Osteoporosis - MedlinePlus Encyclopedia: Bone Mineral Density Test - Mer...
3. `medquad_ans_0000050-34` (score=0.5672) - A comprehensive osteoporosis treatment program includes a focus on proper nutrition, exercise, and safety issues to prevent falls that may result in fractures. In addition, your doctor may prescribe a medication to sl...
4. `medquad_ans_0000050-36` (score=0.5642) - Millions of Americans are able to lead healthy, productive lives while living with osteoporosis. If you have been feeling symptoms of depressionsuch as loss of appetite, hopelessness, feeling useless and helpless, or ...
5. `medquad_ans_0000050-6` (score=0.5512) - Who Treats Osteoporosis? Although there is no cure for osteoporosis, it can be treated. If your doctor does not specialize in osteoporosis, he or she can refer you to a specialist. There is not one type of doctor who ...

**Answer**:

Scientists are pursuing a wide range of basic and clinical studies on osteoporosis, focusing on determining the causes and consequences of bone loss at the cellular and tissue levels, assessing risk factors, developing new strategies to maintain and even enhance bone density and reduce fracture risk, and exploring the roles of genetics, hormones, calcium, vitamin D, drugs, and exercise on bone mass [medquad_ans_0000050-8]. These advances are directly resulting from research efforts aimed at improving prevention and treatment options for osteoporosis. For more information on ongoing research, the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH provides resources [medquad_ans_0000050-8].

**Supported citations**: `medquad_ans_0000050-8`
**Unsupported citations**: _none_

### Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Scores**: context_relevance=0.857, groundedness=0.891, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000430-1__dup1` (score=0.5966) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. ...
2. `medquad_ans_0000338-1__dup1` (score=0.5245) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the s...
3. `medquad_ans_0000927-1__dup2` (score=0.4508) - Summary : It seems to happen almost every day - you hear about the results of a new medical research study. Sometimes the results of one study seem to disagree with the results of another study. It's important to be c...
4. `medquad_ans_0000011-14` (score=0.4497) - Some parents did not tell their children about receiving treatment with pituitary hGH and the possible risk of CJD. These children are now adults. Although the HHS no longer sends annual information about the problem ...
5. `medquad_ans_0000705-1__dup1` (score=0.4169) - Summary : You've probably seen your chart at your doctor's office. In fact, you may have charts at several doctors' offices. If you've been in the hospital, you have a chart there, too. These charts are your medical r...

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. These scams often promise miracle cures, quick fixes, ancient remedies, secret ingredients, or scientific breakthroughs, which may sound too good to be true. Such scams can be dangerous, waste money, and prevent individuals from accessing effective treatments. They commonly target people with serious health conditions like cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. To protect yourself, recognize red flags and consult a doctor or healthcare professional before taking unproven treatments, especially when using prescription drugs [medquad_ans_0000430-1__dup1].

**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

### Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Scores**: context_relevance=1.000, groundedness=0.807, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000275-1__dup1` (score=0.7094) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
2. `medquad_ans_0000055-3__dup2` (score=0.6790) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision -...
3. `medquad_ans_0000016-6` (score=0.6480) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
4. `medquad_ans_0000015-17` (score=0.6417) - Diabetes is a very serious disease. Over time, diabetes that is not well managed causes serious damage to the eyes, kidneys, nerves, and heart, gums and teeth. If you have diabetes, you are more likely than someone wh...
5. `medquad_ans_0000054-1__dup4` (score=0.6319) - Diabetes problems are health problems that can happen when you have diabetes. If your diabetes is not under control, you will have too much glucose\*, also called sugar, in your blood. Having too much glucose in your b...

**Answer**:

The provided context does not contain information about symptoms of preventing diabetes problems or specifically about "keeping your eyes healthy" in terms of prevention. The context discusses symptoms of diabetic eye problems, such as blurry or double vision, rings or flashing lights, dark or floating spots, pain or pressure in the eyes, and trouble seeing out of the corners of the eyes, but these are symptoms of existing eye complications, not preventive measures [medquad_ans_0000275-1\_\_dup1][medquad_ans_0000055-3__dup2][medquad_ans_0000016-6]. Therefore, based on the given context, the symptoms of preventing diabetes problems or keeping eyes healthy cannot be answered.

**Supported citations**: `medquad_ans_0000275-1__dup1`, `medquad_ans_0000055-3__dup2`, `medquad_ans_0000016-6`
**Unsupported citations**: _none_

### Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.042, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0003743-4` (score=0.5672) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition w...
2. `medquad_ans_0000983-3__dup1` (score=0.5662) - Is Castleman disease inherited? Although the exact underlying cause of Castleman disease is unknown, it is thought to occur sporadically in people with no family history of the condition.
3. `medquad_ans_0006460-4` (score=0.5650) - Is Wolff-Parkinson-White syndrome inherited?
4. `medquad_ans_0000557-4` (score=0.5494) - This condition is generally not inherited but arises from mutations in the body's cells that occur after conception. This alteration is called a somatic mutation and is present only in certain cells. Rarely, this cond...
5. `medquad_ans_0000646-4` (score=0.5456) - Menkes syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome)...

**Answer**:

The provided context does not contain information about the inheritance of Aarskog-Scott syndrome. Therefore, it is not possible to determine based on the given text whether Aarskog-Scott syndrome is inherited. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Scores**: context_relevance=0.800, groundedness=0.967, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0006525-1` (score=0.7483) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin. People with X-linked sideroblastic anemia have mature red blood cells that...
2. `medquad_ans_0001062-1` (score=0.7250) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin, which is the protein that carries oxygen in the blood. People with X-linke...
3. `medquad_ans_0001062-4` (score=0.7178) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
4. `medquad_ans_0001063-2` (score=0.7166) - X-linked sideroblastic anemia and ataxia is a rare disorder; only a few affected families have been reported.
5. `medquad_ans_0001063-1` (score=0.7087) - X-linked sideroblastic anemia and ataxia is a rare condition characterized by a blood disorder called sideroblastic anemia and movement problems known as ataxia. This condition occurs only in males. Sideroblastic anem...

**Answer**:

The genetic changes related to X-linked sideroblastic anemia are mutations in the ALAS2 gene. In rare cases, mutations are also found in both the HFE gene and the ALAS2 gene, resulting in a more severe form of the disorder. The condition is inherited in an X-linked recessive pattern [medquad_ans_0006525-1].

**Supported citations**: `medquad_ans_0006525-1`
**Unsupported citations**: _none_

### Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Scores**: context_relevance=0.833, groundedness=0.935, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0003601-1` (score=0.8190) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
2. `medquad_ans_0004514-4` (score=0.6976) - How is oculopharyngeal muscular dystrophy inherited?
3. `medquad_ans_0000602-1` (score=0.6969) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifi...
4. `medquad_ans_0001935-1` (score=0.6845) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystroph...
5. `medquad_ans_0001537-1` (score=0.6817) - Congenital muscular dystrophy (CMD) refers to a group of inherited conditions that affect the muscles and are present at birth or in early infancy. The severity of the condition, the associated signs and symptoms and ...

**Answer**:

Limb-girdle muscular dystrophy is a group of disorders that affect the voluntary muscles around the hips and shoulders, causing weakness and wasting of the muscles, particularly the proximal muscles such as those in the shoulders, upper arms, pelvic area, and thighs [medquad_ans_0003601-1]. The conditions are progressive, leading to a loss of muscle strength and bulk over time, with onset possibly occurring in childhood, adolescence, young adulthood, or later in life [medquad_ans_0003601-1]. Males and females are affected equally, and most forms are inherited in an autosomal recessive manner, while several rare forms follow an autosomal dominant pattern [medquad_ans_0003601-1]. The severity, age of onset, and specific features vary among the subtypes and can differ even within the same family [medquad_ans_000

**Supported citations**: `medquad_ans_0003601-1`
**Unsupported citations**: _none_

### Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Scores**: context_relevance=1.000, groundedness=0.899, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000016-6` (score=0.7333) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
2. `medquad_ans_0000275-1__dup1` (score=0.7279) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
3. `medquad_ans_0000055-3__dup2` (score=0.6911) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision -...
4. `medquad_ans_0000016-1__dup3` (score=0.6715) - Can Cause Vision Loss, Blindness Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina in the back of the eye. A ...
5. `medquad_ans_0000016-8` (score=0.6657) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a com...

**Answer**:

Diabetic eye problems include diabetic retinopathy, cataracts, and glaucoma. Diabetic retinopathy is a complication of diabetes that occurs when high blood glucose levels damage the tiny blood vessels in the retina, leading to potential vision loss or blindness. It often affects both eyes and may initially show no symptoms, though later symptoms can include blurry or double vision, rings or flashing lights, dark or floating spots, pain or pressure in the eyes, and trouble seeing out of the corners of the eyes [medquad_ans_0000016-6][medquad_ans_0000275-1__dup1][medquad_ans_0000055-3__dup2]. Cataracts involve a clouding of the eye's lens and can be treated with surgery to restore clear vision. Glaucoma occurs when pressure builds up in the eye, damaging the main nerve, and

**Supported citations**: `medquad_ans_0000016-6`, `medquad_ans_0000275-1__dup1`, `medquad_ans_0000055-3__dup2`
**Unsupported citations**: _none_

### Example 88 (`0006449-1`)

**Question**: What is (are) Wilson disease ?

**Scores**: context_relevance=1.000, groundedness=0.676, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001037-2` (score=0.8276) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
2. `medquad_ans_0006449-1` (score=0.7802) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver,...
3. `medquad_ans_0001037-5` (score=0.7498) - These resources address the diagnosis or management of Wilson disease: - Gene Review: Gene Review: Wilson Disease - Genetic Testing Registry: Wilson's disease - MedlinePlus Encyclopedia: Wilson's disease - National Hu...
4. `medquad_ans_0006449-3` (score=0.7281) - What causes Wilson disease? Wilson disease is caused by changes (mutations) in the ATP7B gene. This gene encodes a protein that plays an important role in the transport of copper from the liver to the rest of the body...
5. `medquad_ans_0000133-1__dup3` (score=0.7200) - Wilson disease is a genetic disease that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filt...

**Answer**:

Wilson disease is a rare inherited disorder characterized by the accumulation of copper in the body due to mutations in the ATP7B gene. This gene encodes a protein responsible for transporting copper from the liver to other parts of the body and removing excess copper from the body. When the protein does not function properly, copper builds up in tissues and organs such as the liver, brain, and eyes, leading to damage. High levels of copper are toxic and can cause chronic liver disease, central nervous system abnormalities, and psychiatric disturbances. The disorder is inherited in an autosomal recessive manner and currently has no cure, but therapies exist to reduce or control copper accumulation [medquad_ans_0006449-1][medquad_ans_0003449-3][medquad_ans_0000133-1__dup3].

**Supported citations**: `medquad_ans_0006449-1`, `medquad_ans_0000133-1__dup3`
**Unsupported citations**: `medquad_ans_0003449-3`

### Example 89 (`0001503-2`)

**Question**: What are the symptoms of Congenital dyserythropoietic anemia type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001503-2` (score=0.8174) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to abs...
2. `medquad_ans_0001504-1` (score=0.7093) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital dyserythropoietic anemia type 3. If the info...
3. `medquad_ans_0001503-1` (score=0.6721) - Congenital dyserythropoietic anemia type 2 (CDA II) is an inherited blood disorder characterized by mild to severe anemia. It is usually diagnosed in adolescence or early adulthood. Many affected individuals have yell...
4. `medquad_ans_0005622-4` (score=0.6346) - How is sideroblastic anemia diagnosed? The principle feature of sideroblastic anemia is slowly progressive, mild, life-long anemia which often goes unnoticed. Symptoms of iron overload may lead to the discovery of thi...
5. `medquad_ans_0000223-1` (score=0.6322) - Congenital dyserythropoietic anemia (CDA) is an inherited blood disorder that affects the development of red blood cells. This disorder is one of many types of anemia, which is a condition characterized by a shortage ...

**Answer**:

The symptoms of Congenital dyserythropoietic anemia type 2 include jaundice, gallstones, an enlarged liver and spleen (splenomegaly), anemia of inadequate production, endopolyploidy on chromosome studies of bone marrow, reduced activity of N-acetylglucosaminyltransferase II, reticulocytosis, and iron overload. Iron overload can lead to complications such as an abnormal heart rhythm (arrhythmia), congestive heart failure, diabetes, and chronic liver disease (cirrhosis). Rarely, mediastinal tumors may occur [medquad_ans_0001503-2].

**Supported citations**: `medquad_ans_0001503-2`
**Unsupported citations**: _none_

### Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Scores**: context_relevance=0.800, groundedness=0.929, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000038-11` (score=0.7362) - Kidney disease -- also known as chronic kidney disease (CKD) -- occurs when kidneys can no longer remove wastes and extra water from the blood or perform other functions as they should. According to the Centers for Di...
2. `medquad_ans_0000140-2__dup2` (score=0.6857) - Chronic kidney disease is kidney damage that occurs slowly over many years, often due to diabetes or high blood pressure. Once damaged, the kidneys cant filter blood as they should. This damage can cause wastes to bui...
3. `medquad_ans_0000038-12` (score=0.6617) - Kidney disease is most often caused by diabetes or high blood pressure. Each kidney contains about one million tiny filters made up of blood vessels. These filters are called glomeruli. Diabetes and high blood pressur...
4. `medquad_ans_0000033-1__dup4` (score=0.6602) - Diabetic kidney disease, also called diabetic nephropathy, is kidney disease caused by diabetes. Even when well controlled, diabetes can lead to chronic kidney disease (CKD) and kidney failure, described as end-stage ...
5. `medquad_ans_0001009-1` (score=0.6331) - Uromodulin-associated kidney disease is an inherited condition that affects the kidneys. The signs and symptoms of this condition vary, even among members of the same family. Many individuals with uromodulin-associate...

**Answer**:

Kidney disease, also known as chronic kidney disease (CKD), occurs when kidneys can no longer remove wastes and extra water from the blood or perform other functions as they should [medquad_ans_0000038-11]. Chronic kidney disease is kidney damage that occurs slowly over many years, often due to diabetes or high blood pressure [medquad_ans_0000140-2__dup2]. This damage impairs the kidneys' ability to filter blood properly, leading to a buildup of wastes in the body and other health problems, including mineral and bone disorder [medquad_ans_0000140-2__dup2]. Each kidney contains about one million tiny filters called glomeruli, which are made up of blood vessels; diabetes and high blood pressure damage these blood vessels, reducing the kidneys' filtering capacity [medquad_ans_0000038-12

**Supported citations**: `medquad_ans_0000038-11`, `medquad_ans_0000140-2__dup2`
**Unsupported citations**: _none_

### Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Scores**: context_relevance=0.857, groundedness=0.962, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000058-23` (score=0.7426) - There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But ...
2. `medquad_ans_0000058-6__dup1` (score=0.7113) - Choosing Treatment There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade...
3. `medquad_ans_0000058-26` (score=0.6849) - Surgery is a common treatment for early stage prostate cancer. It is used to remove the cancer. The surgeon may remove the entire prostate -- a type of surgery called radical prostatectomy -- or, in some cases, remove...
4. `medquad_ans_0000036_1-6` (score=0.6823) - Key Points - There are different types of treatment for patients with prostate cancer. - Seven types of standard treatment are used: - Watchful waiting or active surveillance - Surgery - Radiation therapy and radiopha...
5. `medquad_ans_0000058-4__dup3` (score=0.6522) - If tests show that you have cancer, you should talk with your doctor in order to make treatment decisions. Working With a Team of Specialists A team of specialists often treats people with cancer. The team will keep t...

**Answer**:

The treatments for prostate cancer may involve watchful waiting, surgery, radiation therapy, or hormonal therapy. Some men receive a combination of these therapies. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor, as well as the man's age, general health, and his feelings about the treatments and their possible side effects [medquad_ans_0000058-23]. A cure is probable for men whose prostate cancer is diagnosed early [medquad_ans_0000058-23]. Surgery, such as radical prostatectomy, may be used to remove the cancer, and can sometimes preserve sexual function [medquad_ans_0000058-6__dup1]. Radiation therapy and hormonal therapy also have potential side effects, including impacts on sexual desire or performance and urinary function [medquad_ans_0000058-6\_\_dup

**Supported citations**: `medquad_ans_0000058-23`, `medquad_ans_0000058-6__dup1`
**Unsupported citations**: _none_

### Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Scores**: context_relevance=0.857, groundedness=0.931, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000431-1__dup1` (score=0.6248) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...
2. `medquad_ans_0000036-2__dup3` (score=0.5674) - Health insurance helps pay for medical care, including the cost of diabetes care. Health insurance options include the following: - private health insurance, which includes group and individual health insurance - gove...
3. `medquad_ans_0000036-4__dup3` (score=0.5611) - Insurance companies sell private health insurance plans. Two types of private health insurance are - Group health insurance. People may be eligible to purchase group health insurance through their employer or union or...
4. `medquad_ans_0000045-9` (score=0.5246) - Medicare Part B is medical insurance. It helps cover medical services such as doctor's services, outpatient care and other medical services that Part A doesn't cover. Part B also covers some preventive services, such ...
5. `medquad_ans_0000705-1__dup1` (score=0.5063) - Summary : You've probably seen your chart at your doctor's office. In fact, you may have charts at several doctors' offices. If you've been in the hospital, you have a chart there, too. These charts are your medical r...

**Answer**:

Yes, there is information about health insurance. Health insurance helps protect individuals from high medical care costs by providing coverage for medical expenses, including diabetes care. It is a contract between the individual and an insurance company, where the company agrees to pay part of the expenses when medical care is needed. Health insurance can be obtained through employers, where the employer often helps pay for the insurance, or through individual purchases. Private health insurance includes group and individual plans, while government health insurance options include Medicare, Medicaid, the Children's Health Insurance Program (CHIP), TRICARE, and veterans health care programs. The Affordable Care Act (ACA) prevents insurers from denying coverage or charging higher premiums to people with preexisting conditions and requires most people to have health insurance or pay a fee, with some exemptions. The ACA also expanded coverage for many people in the U.S. [medquad_ans_0000431-1__dup1]medquad_ans

**Supported citations**: `medquad_ans_0000431-1__dup1`
**Unsupported citations**: _none_

### Example 93 (`0003860-1`)

**Question**: What is (are) MECP2 duplication syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003860-1` (score=0.8701) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...
2. `medquad_ans_0000635-1` (score=0.7956) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infan...
3. `medquad_ans_0000635-2` (score=0.7841) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of...
4. `medquad_ans_0000635-3` (score=0.7779) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
5. `medquad_ans_0003860-3` (score=0.7709) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...

**Answer**:

MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. It is caused by an extra copy (duplication) of the MECP2 gene in each cell, which results from a duplication of genetic material on the long (q) arm of the X chromosome. The duplication size varies from 100,000 to 900,000 base pairs (100 to 900 kilobases), and the MECP2 gene is always included; other genes may be involved depending on the size of the duplicated segment. Extra copies of these other genes do not appear to affect the severity of the condition. The syndrome is associated with infantile hypotonia, delayed motor milestones (such as sitting up and walking), poor or absent speech, seizures, spasticity, and recurrent respiratory tract infections, which are a major cause of death in affected

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Scores**: context_relevance=0.833, groundedness=0.096, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000594-1__dup2` (score=0.6686) - Melanoma is the most serious type of skin cancer. Often the first sign of melanoma is a change in the size, shape, color, or feel of a mole. Most melanomas have a black or black-blue area. Melanoma may also appear as ...
2. `medquad_ans_0000063-6__dup1` (score=0.6312) - Many Treatment Options There are many treatment options for people with melanoma and non-melanoma skin cancer. The choice of treatment depends on your age and general health, the site of the cancer, the stage of the c...
3. `medquad_ans_0000017_1-5` (score=0.6102) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - How the melanoma cells look under a microscope. - The si...
4. `medquad_ans_0000028_2-1` (score=0.6079) - Key Points - Melanoma is a disease in which malignant (cancer) cells form in melanocytes (cells that color the skin). - There are different types of cancer that start in the skin. - Melanoma can occur anywhere on the ...
5. `medquad_ans_0000063-15` (score=0.6061) - Melanoma is the fastest growing and most invasive type of skin cancer. This cancer arises from overgrowth of melanocytes. Melanocytes are specialized skin cells that produce a pigment called melanin.

**Answer**:

The provided context does not contain information about the outlook or prognosis for melanoma. Therefore, it is not possible to answer the question based on the given text. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000139-2__dup2` (score=0.5111) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...
2. `medquad_ans_0002274-3` (score=0.5051) - How is familial isolated hyperparathyroidism inherited? Familial isolated hyperparathyroidism (FIHP) is typically inherited in an autosomal dominant manner. This means that having only one changed (mutated) copy of th...
3. `medquad_ans_0001090-4` (score=0.4689) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000741-4` (score=0.4689) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000045-4` (score=0.4689) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain information about the inheritance pattern of boomerang dysplasia. Therefore, it is not possible to determine whether boomerang dysplasia is inherited based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Scores**: context_relevance=0.800, groundedness=0.087, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000466-1` (score=0.7429) - Hereditary multiple osteochondromas is a condition in which people develop multiple benign (noncancerous) bone tumors called osteochondromas. The number of osteochondromas and the bones on which they are located vary ...
2. `medquad_ans_0000466-2` (score=0.7262) - The incidence of hereditary multiple osteochondromas is estimated to be 1 in 50,000 individuals. This condition occurs more frequently in some isolated populations: the incidence is approximately 1 in 1,000 in the Cha...
3. `medquad_ans_0002838-1` (score=0.7186) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called...
4. `medquad_ans_0002838-2` (score=0.7157) - What are the signs and symptoms of Hereditary multiple osteochondromas? The Human Phenotype Ontology provides the following list of signs and symptoms for Hereditary multiple osteochondromas. If the information is ava...
5. `medquad_ans_0000466-5` (score=0.6750) - These resources address the diagnosis or management of hereditary multiple osteochondromas: - Gene Review: Gene Review: Hereditary Multiple Osteochondromas - Genetic Testing Registry: Multiple congenital exostosis - G...

**Answer**:

The provided context does not mention specific genetic changes related to hereditary multiple osteochondromas. Therefore, based on the given information, the genetic changes are not specified [medquad_ans_0000466-1, medquad_ans_0002838-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000466-1, medquad_ans_0002838-1`

### Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.753, answer_relevance=0.475

**Top docs**:

1. `medquad_ans_0000746-5` (score=0.6730) - These resources address the diagnosis or management of oculofaciocardiodental syndrome: - Genetic Testing Registry: Oculofaciocardiodental syndrome These resources from MedlinePlus offer information about the diagnosi...
2. `medquad_ans_0004510-1` (score=0.6431) - Oculofaciocardiodental syndrome is a genetic syndrome that affects the eyes, heart, face, and teeth. Common signs and symptoms include abnormally small deep-set eyes, cataracts, long narrow face, a broad nasal tip tha...
3. `medquad_ans_0004510-2` (score=0.6162) - What are the signs and symptoms of Oculofaciocardiodental syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Oculofaciocardiodental syndrome. If the information is available, ...
4. `medquad_ans_0000344-2` (score=0.5862) - It is estimated that 750,000 people in the United States have dilated cardiomyopathy; roughly half of these cases are familial.
5. `medquad_ans_0000151-2` (score=0.5647) - Cardiofaciocutaneous syndrome is a very rare condition whose incidence is unknown. Researchers estimate that 200 to 300 people worldwide have this condition.

**Answer**:

The provided context does not specify the total number of people affected by oculofaciocardiodental syndrome. While it lists the frequency of various signs and symptoms in individuals with the condition, such as "90% with abnormality of the cardiac septa," it does not provide a population estimate or prevalence figure. Therefore, the number of people affected cannot be determined from the given information.

Answer: Insufficient context to determine the number of people affected by oculofaciocardiodental syndrome. [medquad_ans_0004510-2]

**Supported citations**: `medquad_ans_0004510-2`
**Unsupported citations**: _none_

### Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000189-1` (score=0.6532) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused b...
2. `medquad_ans_0001350-1` (score=0.6168) - Citrullinemia type I is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. This condition, also known as classic citrullinemia, belongs to a class of genetic diseases call...
3. `medquad_ans_0004334-1` (score=0.5790) - Neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) is a liver condition is also known as neonatal-onset type II citrullinemia. NICCD blocks the flow of bile (a digestive fluid produced by the liver)...
4. `medquad_ans_0000189-5` (score=0.5754) - These resources address the diagnosis or management of citrullinemia: - Baby's First Test: Citrullinemia, Type I - Baby's First Test: Citrullinemia, Type II - Gene Review: Gene Review: Citrin Deficiency - Gene Review:...
5. `medquad_ans_0001350-5` (score=0.5702) - What happens when citrullinemia type I is not treated? Untreated individuals with the severe form of citrullinemia type I have hyperammonemia (plasma ammonia concentration 1000-3000 mol/L). Without prompt intervention...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described: Type I (also known as classic citrullinemia), which is caused by mutations in the ASS1 gene and inherited in an autosomal recessive pattern, and typically becomes evident in the first few days of life with symptoms such as lethargy, poor feeding, vomiting, seizures, and loss of consciousness; and Type II, which chiefly affects the nervous system, causing confusion, restlessness, memory loss, abnormal behaviors, seizures, and coma, and may appear during adulthood, often triggered by medications, infections, surgery, or alcohol intake. Type II can also develop in individuals who had neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) as infants [medquad_ans_0000189-1]medquad_ans

**Supported citations**: `medquad_ans_0000189-1`
**Unsupported citations**: _none_

### Example 99 (`0000718-1__dup1`)

**Question**: What is (are) Bilateral frontal polymicrogyria ?

**Scores**: context_relevance=0.833, groundedness=0.968, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000718-1__dup1` (score=0.8054) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
2. `medquad_ans_0000720-1__dup1` (score=0.7570) - Bilateral generalized polymicrogyria is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). This is the most widespread form of polymicrogyria and typically affects the enti...
3. `medquad_ans_0000719-1__dup1` (score=0.7412) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides ...
4. `medquad_ans_0004956-1` (score=0.6898) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. Specifically, the surface of the brain develops too many folds which are unusually small. The signs and symptoms associate...
5. `medquad_ans_0000807-1` (score=0.6802) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. The surface of the brain normally has many ridges or folds, called gyri. In people with polymicrogyria, the brain develops...

**Answer**:

Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form that affects only the frontal lobes of the brain, without involving the areas located behind the Sylvius fissure or the Rolando sulcus. Signs and symptoms include delayed motor and language milestones; spastic hemiparesis or quadriparesis; and mild to moderate intellectual disability. Seizures may also be present. The cause of this form is still not known, unlike the frontoparietal form, which is associated with mutations in the GPR56 gene. Treatment is based on the signs and symptoms present in each individual [medquad_ans_0000718-1__dup1].

**Supported citations**: `medquad_ans_0000718-1__dup1`
**Unsupported citations**: _none_

### Example 100 (`0000944-5`)

**Question**: What are the treatments for spondylocostal dysostosis ?

**Scores**: context_relevance=0.857, groundedness=0.845, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000944-2` (score=0.6994) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.
2. `medquad_ans_0005860-4` (score=0.6878) - What treatment is available for spondylothoracic dysostosis? Many infants born with spondylothoracic dysostosis have difficulty breathing due to their small, malformed chests, and therefore are prone to repeated respi...
3. `medquad_ans_0000949-5` (score=0.6595) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from Medli...
4. `medquad_ans_0005820-1` (score=0.6290) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
5. `medquad_ans_0005819-1` (score=0.6215) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...

**Answer**:

Treatment for spondylocostal dysostosis usually consists of intensive medical care, including treatment of respiratory infections, bone surgery, and orthopedic treatment. Respiratory support may also be provided, and surgery may be used to correct inguinal hernia and scoliosis [medquad_ans_0005860-4]. Treatment is generally symptomatic and supportive, depending on the specific complications such as breathing difficulties or spinal deformities [medquad_ans_0005820-1].

**Supported citations**: `medquad_ans_0005860-4`, `medquad_ans_0005820-1`
**Unsupported citations**: _none_

### Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0004825-1` (score=0.7411) - What are the signs and symptoms of Perry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Perry syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0000786-1` (score=0.7112) - Perry syndrome is a progressive brain disease that is characterized by four major features: a pattern of movement abnormalities known as parkinsonism, psychiatric changes, weight loss, and abnormally slow breathing (h...
3. `medquad_ans_0002652-2` (score=0.6520) - What are the signs and symptoms of Gordon syndrome? Gordon syndrome belongs to a group of conditions known as the distal arthrogryposes, which are characterized by stiffness and impaired mobility of certain joints of ...
4. `medquad_ans_0002480-2` (score=0.6426) - What are the signs and symptoms of Galloway-Mowat syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Galloway-Mowat syndrome. If the information is available, the table below ...
5. `medquad_ans_0005016-2` (score=0.6326) - What are the signs and symptoms of Prader-Willi syndrome? In infancy, Prader-Willi syndrome (PWS) is characterized by weak muscle tone (hypotonia), feeding difficulties, poor growth, and delayed development. In later ...

**Answer**:

The symptoms of Perry syndrome include abnormality of extrapyramidal motor function (such as parkinsonism, bradykinesia, rigidity, tremor, and mask-like facies), respiratory insufficiency (including hypoventilation and central hypoventilation), sleep disturbance, weight loss, developmental regression, hallucinations, hypotension, apathy, inappropriate behavior, insomnia, dysarthria, weak voice, and psychiatric changes such as depression and suicidal thoughts [medquad_ans_0004825-1]. Parkinsonism and psychiatric changes are typically the earliest features of the disease, which usually appears in a person's forties or fifties [medquad_ans_0000786-1].

**Supported citations**: `medquad_ans_0004825-1`, `medquad_ans_0000786-1`
**Unsupported citations**: _none_

### Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Scores**: context_relevance=1.000, groundedness=0.935, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006399-1` (score=0.7121) - What are the signs and symptoms of Warfarin syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0006398-2` (score=0.6932) - What are the signs and symptoms of Warfarin sensitivity? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin sensitivity. If the information is available, the table below includ...
3. `medquad_ans_0001028-2` (score=0.6626) - The prevalence of warfarin sensitivity is unknown. However, it appears to be more common in people who are older, those with lower body weights, and individuals of Asian ancestry. Of the approximately 2 million people...
4. `medquad_ans_0001027-2` (score=0.6118) - Warfarin resistance is thought to be a rare condition, although its prevalence is unknown.
5. `medquad_ans_0006398-1` (score=0.5983) - Warfarin sensitivity is a condition that is characterized by a reduced tolerance for a "blood-thinning" medication called warfarin. Warfarin is an anticoagulant that is often prescribed to people who are at an increas...

**Answer**:

The symptoms of Warfarin sensitivity include abnormality of blood and blood-forming tissues, abnormality of metabolism/homeostasis, and autosomal dominant inheritance. [medquad_ans_0006398-2]

**Supported citations**: `medquad_ans_0006398-2`
**Unsupported citations**: _none_

### Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.528, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000701-2` (score=0.8040) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.
2. `medquad_ans_0004262-1` (score=0.7918) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, a...
3. `medquad_ans_0000701-1` (score=0.7511) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most...
4. `medquad_ans_0004262-2` (score=0.6545) - What are the signs and symptoms of Myhre syndrome? Myhre syndrome is a condition with features affecting many systems and functions of the body. Associated findings might include: Delayed development of language and m...
5. `medquad_ans_0000701-5` (score=0.6078) - These resources address the diagnosis or management of Myhre syndrome: - Centers for Disease Control and Prevention: Types of Hearing Loss - Genetic Testing Registry: Myhre syndrome - National Institute on Deafness an...

**Answer**:

Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement of the muscles (muscle hypertrophy), and/or joint stiffness. The condition is caused by mutations in the SMAD4 gene and is inherited in an autosomal dominant pattern, with most cases due to a new mutation. Features affecting multiple body systems include delayed development of language and motor skills, hearing loss (sensorineural, conductive, or mixed), low birth weight, reduced growth throughout life, skin stiffness, and muscular appearance. Skeletal abnormalities include thickened skull bones, flattened spine bones (platyspondyly), broad ribs, underdeveloped pelvic wings (hypoplastic iliac wings), and short fingers and toes (brachydactyly). Typical facial features include narrow eyelid openings, short philtrum, midface hyp

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Scores**: context_relevance=0.857, groundedness=0.686, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0004451-2` (score=0.5664) - What are the signs and symptoms of Noonan syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome 1. If the information is available, the table below includes how...
2. `medquad_ans_0002523-1` (score=0.5339) - What are the signs and symptoms of Genitopatellar syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Genitopatellar syndrome. If the information is available, the table below ...
3. `medquad_ans_0004826-1` (score=0.5334) - What are the signs and symptoms of Persistence of mullerian derivatives with lymphangiectasia and postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Persistence o...
4. `medquad_ans_0000673-2__dup1` (score=0.5289) - What are the signs and symptoms of Behcet's disease? Symptoms of Behcet's disease include recurrent ulcers in the mouth (resembling canker sores) and on the genitals, and eye inflammation (uveitis). The disorder may a...
5. `medquad_ans_0005640-1` (score=0.5289) - What are the signs and symptoms of Single upper central incisor? The Human Phenotype Ontology provides the following list of signs and symptoms for Single upper central incisor. If the information is available, the ta...

**Answer**:

The provided context does not contain information about the symptoms of Thanatophoric dysplasia. [medquad_ans_0004451-2] and [medquad_ans_0002523-1] discuss symptoms of Noonan syndrome 1 and Genitopatellar syndrome, respectively, but neither mentions Thanatophoric dysplasia. Therefore, the requested information is not available in the context.

**Supported citations**: `medquad_ans_0004451-2`, `medquad_ans_0002523-1`
**Unsupported citations**: _none_

### Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002417-2` (score=0.8171) - What are the signs and symptoms of Fragile X syndrome? Fragile X syndrome is characterized by developmental problems including intellectual disability and delayed speech and language development. Males are usually mor...
2. `medquad_ans_0000381-1__dup1` (score=0.7157) - Fragile X syndrome is the most common form of inherited developmental disability. A problem with a specific gene causes the disease. Normally, the gene makes a protein you need for brain development. But the problem c...
3. `medquad_ans_0000381-1` (score=0.6928) - Fragile X syndrome is a genetic condition that causes a range of developmental problems including learning disabilities and cognitive impairment. Usually, males are more severely affected by this disorder than females...
4. `medquad_ans_0002417-6` (score=0.6657) - How might fragile X syndrome be treated? There is no specific treatment available for fragile X syndrome. Management of this condition is generally supportive and may include: recognizing the need for special educatio...
5. `medquad_ans_0000381-5` (score=0.6528) - These resources address the diagnosis or management of fragile X syndrome: - Gene Review: Gene Review: FMR1-Related Disorders - GeneFacts: Fragile X Syndrome: Diagnosis - GeneFacts: Fragile X Syndrome: Management - Ge...

**Answer**:

The symptoms of Fragile X syndrome include cognitive impairment, joint hypermobility, macroorchidism, neurological speech impairment, otitis media, pes planus, abnormality of the pinna, attention deficit hyperactivity disorder, frontal bossing, intellectual disability, moderate, long face, macrocephaly, mandibular prognathia, muscular hypotonia, narrow face, sinusitis, abnormality of the mitral valve, autism, cerebral cortical atrophy, dilatation of the ascending aorta, seizures, self-injurious behavior, and strabismus [medquad_ans_0002417-2]. Additionally, males may have enlarged testicles (macroorchidism) after puberty, and physical features such as a long and narrow face, large ears, prominent jaw and forehead, unusually flexible fingers, and flat feet may become more apparent with age [medquad_ans_000241

**Supported citations**: `medquad_ans_0002417-2`
**Unsupported citations**: _none_

### Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Scores**: context_relevance=1.000, groundedness=0.495, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000166-2` (score=0.5606) - CADASIL is likely a rare condition; however, its prevalence is unknown.
2. `medquad_ans_0000166-5` (score=0.5074) - These resources address the diagnosis or management of CADASIL: - Butler Hospital: Treatment and Management of CADASIL - Gene Review: Gene Review: CADASIL - Genetic Testing Registry: Cerebral autosomal dominant arteri...
3. `medquad_ans_0000056-3__dup1` (score=0.4926) - Symptoms usually progress slowly. By age 65, the majority of persons with CADASIL have cognitive problems and dementia. Some will become dependent due to multiple strokes.
4. `medquad_ans_0000889-2__dup1` (score=0.4206) - What are the signs and symptoms of CADASIL? Strokes are the main feature of CADASIL and often occur repeatedly. Strokes may lead to severe disability such as an inability to walk and urinary incontinence. The average ...
5. `medquad_ans_0000166-1` (score=0.3585) - Cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy, usually called CADASIL, is an inherited condition that causes stroke and other impairments. This condition affects blood flow...

**Answer**:

CADASIL is a rare condition characterized by recurrent strokes, which are the main feature, often occurring repeatedly and leading to severe disability such as inability to walk and urinary incontinence. It typically presents with a slow decline in cognitive function, with over half of affected individuals experiencing a decline in thinking ability, beginning as early as 35 years of age. Approximately 75% of people with CADASIL eventually develop dementia, including significant difficulties with reasoning and memory. Migraines with aura occur in about 35% of individuals, with the first attack typically occurring at an average age of 26 years. Psychiatric issues, such as personality changes or severe depression, affect about 30% of people. Other common signs and symptoms include abnormality of temperature regulation (90%), abnormality of the retinal vasculature (90%), and memory impairment (50%). The condition is associated with cerebral autosomal dominant arter

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Scores**: context_relevance=1.000, groundedness=0.992, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000688-1` (score=0.7523) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate...
2. `medquad_ans_0000688-2` (score=0.6678) - Multiple endocrine neoplasia type 1 affects about 1 in 30,000 people; multiple endocrine neoplasia type 2 affects an estimated 1 in 35,000 people. Among the subtypes of type 2, type 2A is the most common form, followe...
3. `medquad_ans_0004187-2` (score=0.6151) - What are the signs and symptoms of Multiple endocrine neoplasia type 2A? The Human Phenotype Ontology provides the following list of signs and symptoms for Multiple endocrine neoplasia type 2A. If the information is a...
4. `medquad_ans_0004185-2` (score=0.6025) - What are the signs and symptoms of Multiple endocrine neoplasia type 1? Multiple endocrine neoplasia, type 1 (MEN1) is characterized primarily by several different types of endocrine tumors. People affected by MEN1 ty...
5. `medquad_ans_0000010-8` (score=0.5836) - - Multiple endocrine neoplasia type 1 (MEN1) is an inherited disorder that causes hormone-secreting tumors in the duodenum and the endocrine glands-most often the parathyroid, pancreas, and pituitary. - Overactive par...

**Answer**:

Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cells and tissues throughout the body. Multiple endocrine neoplasia typically involves tumors (neoplasia) in at least two endocrine glands; tumors can also develop in other organs and tissues. These growths can be noncancerous (benign) or cancerous (malignant). If the tumors become cancerous, the condition can be life-threatening. The major forms of multiple endocrine neoplasia are called type 1, type 2, and type 4. These types are distinguished by the genes involved, the types of hormones made, and the characteristic signs and symptoms. [medquad_ans_0000688-1]

**Supported citations**: `medquad_ans_0000688-1`
**Unsupported citations**: _none_

### Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000200-1__dup2` (score=0.7192) - The mucopolysaccharidoses are a group of inherited metabolic diseases in which a defective or missing enzyme causes large amounts of complex sugar molecules to accumulate in harmful amounts in the body's cells and tis...
2. `medquad_ans_0004160-1` (score=0.6898) - Mucopolysaccharidosis type IIIC (MPS IIIC) is an genetic disorder that makes the body unable to break down large sugar molecules called glycosaminoglycans (GAGs, formerly called mucopolysaccharides). Specifically, peo...
3. `medquad_ans_0004155-1` (score=0.6759) - Mucopolysaccharidosis I (MPS I) is a condition that affects many parts of the body. It is a progressively debilitating disorder; however, the rate of progression varies among affected individuals. MPS I is caused by m...
4. `medquad_ans_0004155-3` (score=0.6661) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycos...
5. `medquad_ans_0004161-1` (score=0.6618) - Mucopolysaccharidosis type IIID (MPS IIID) is an genetic disorder that makes the body unable to break down large sugar molecules called glycosaminoglycans (GAGs, formerly called mucopolysaccharides). Specifically, peo...

**Answer**:

Mucopolysaccharidosis type I (MPS I) is caused by mutations in the IDUA gene. These mutations lead to reduced levels or the complete lack of the IDUA enzyme. Without the proper amount of this enzyme, large sugar molecules called glycosaminoglycans (GAGs) accumulate within cells called lysosomes [medquad_ans_0004155-1].

**Supported citations**: `medquad_ans_0004155-1`
**Unsupported citations**: _none_

### Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Scores**: context_relevance=0.600, groundedness=0.000, answer_relevance=0.760

**Top docs**:

1. `medquad_ans_0004784-4` (score=0.5117) - How is Pelizaeus-Merzbacher disease inherited?
2. `medquad_ans_0002911-3` (score=0.5112) - What causes Holt-Oram syndrome? Holt-Oram syndrome is caused by changes (mutations) in the TBX5 gene. This gene gives the body instructions for making a protein involved in the development of the heart and upper limbs...
3. `medquad_ans_0000107-3__dup2` (score=0.4805) - Primary ciliary dyskinesia (PCD) is a rare, inherited disease. "Inherited" means the disease is passed from parents to children through genes. With PCD, this process is very complex. Researchers are still learning how...
4. `medquad_ans_0004671-4` (score=0.4796) - How is pachyonychia congenita inherited? Pachyonychia congenita (PC) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of the responsible...
5. `medquad_ans_0006446-4` (score=0.4729) - Is Williams syndrome inherited?

**Answer**:

The provided context does not contain any information about the genetic changes related to primary spontaneous pneumothorax. Therefore, a definitive answer cannot be provided based on the given context. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Scores**: context_relevance=0.750, groundedness=0.943, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000066-13` (score=0.6536) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blo...
2. `medquad_ans_0000261-1__dup2` (score=0.6203) - A stroke occurs when the blood supply to part of the brain is suddenly interrupted or when a blood vessel in the brain bursts, spilling blood into the spaces surrounding brain cells. Brain cells die when they no longe...
3. `medquad_ans_0000066-18` (score=0.6178) - Transient ischemic attacks, or TIAs, occur when the warning signs of stroke last only a few moments and then disappear. These brief episodes are also sometimes called "mini-strokes." Although brief, they identify an u...
4. `medquad_ans_0000857-1__dup2` (score=0.6075) - A stroke is a medical emergency. Strokes happen when blood flow to your brain stops. Within minutes, brain cells begin to die. There are two kinds of stroke. The more common kind, called ischemic stroke, is caused by ...
5. `medquad_ans_0000124-1__dup2` (score=0.5944) - A stroke occurs if the flow of oxygen-rich blood to a portion of the brain is blocked. Without oxygen, brain cells start to die after a few minutes. Sudden bleeding in the brain also can cause a stroke if it damages b...

**Answer**:

A stroke is a medical emergency that occurs when the blood supply to part of the brain is suddenly interrupted or when a blood vessel in the brain bursts, spilling blood into the spaces surrounding brain cells. Brain cells die when they no longer receive oxygen and nutrients from the blood or there is sudden bleeding into or around the brain. There are two main types of stroke: ischemic stroke, which is caused by a blood clot that blocks or plugs a blood vessel in the brain and accounts for approximately 80 percent of all strokes, and hemorrhagic stroke, which is caused by a blood vessel that breaks and bleeds into the brain [medquad_ans_0000261-1__dup2]. Additionally, transient ischemic attacks (TIAs), also known as "mini-strokes," occur when the blood supply to the brain is briefly interrupted, resulting in temporary symptoms that resolve within a few moments [medquad_ans_0000

**Supported citations**: `medquad_ans_0000261-1__dup2`
**Unsupported citations**: _none_

### Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.903, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006276-2` (score=0.7243) - What are the signs and symptoms of Uncombable hair syndrome? Uncombable hair syndrome (UHS) may first become apparent any time between the ages of 3 months and 12 years. It only affects the scalp hair. The quantity of...
2. `medquad_ans_0006276-1` (score=0.7080) - Uncombable hair syndrome (UHS) is a rare disorder of the hair shaft of the scalp. It usually is characterized by silvery-blond or straw-colored hair that is disorderly; stands out from the scalp; and cannot be combed ...
3. `medquad_ans_0006276-5` (score=0.7039) - How is uncombable hair syndrome diagnosed? A diagnosis of uncombable hair syndrome (UHS) is made by observing the characteristic symptoms of the condition, as well observing the hair shaft under a special microscope. ...
4. `medquad_ans_0002708-1` (score=0.6790) - HAIR-AN syndrome is a condition that affects women. It is characterized by hyperandrogenism, insulin resistance, and acanthosis nigricans. Insulin resistance is a condition in which the body produces insulin but does ...
5. `medquad_ans_0006276-6` (score=0.6723) - How might uncombable hair syndrome be treated? There is no definitive treatment for uncombable hair syndrome, but the condition usually improves or resolves on its own with the onset of puberty. Gentle hair care is ge...

**Answer**:

The cause of Uncombable hair syndrome (UHS) is not known. A responsible gene has not yet been identified, and while the condition is likely inherited in an autosomal dominant manner with reduced penetrance, no specific genetic cause has been established [medquad_ans_0006276-1].

**Supported citations**: `medquad_ans_0006276-1`
**Unsupported citations**: _none_

### Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.028, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000261-4` (score=0.5075) - In most cases of cyclic vomiting syndrome, affected people have no known history of the disorder in their family. However, many affected individuals have a family history of related conditions, such as migraines, irri...
2. `medquad_ans_0003743-4` (score=0.4794) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition w...
3. `medquad_ans_0000557-4` (score=0.4790) - This condition is generally not inherited but arises from mutations in the body's cells that occur after conception. This alteration is called a somatic mutation and is present only in certain cells. Rarely, this cond...
4. `medquad_ans_0000661-4` (score=0.4728) - Mitochondrial complex III deficiency is usually inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive co...
5. `medquad_ans_0000139-2__dup2` (score=0.4720) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...

**Answer**:

The provided context does not contain any information about Russell-Silver syndrome or its inheritance pattern. Therefore, based on the given context, it is not possible to determine whether Russell-Silver syndrome is inherited.

Answer: Insufficient context [no_doc_id].

**Supported citations**: _none_
**Unsupported citations**: `no_doc_id`

### Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Scores**: context_relevance=1.000, groundedness=0.959, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.7476) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0000718-1__dup2` (score=0.4953) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts...
3. `medquad_ans_0000094-5__dup1` (score=0.4619) - Obesity hypoventilation syndrome (OHS) is diagnosed based on your medical history, signs and symptoms, and test results. Specialists Involved A critical care specialist, pulmonologist (lung specialist), and/or sleep s...
4. `medquad_ans_0000018-1__dup2` (score=0.4542) - After any operation, you'll have some side effects. There is usually some pain with surgery. There may also be swelling and soreness around the area that the surgeon cut. Your surgeon can tell you which side effects t...
5. `medquad_ans_0000022-5` (score=0.4189) - These resources address the diagnosis or management of adiposis dolorosa: - Genetic Testing Registry: Lipomatosis dolorosa - Merck Manual Consumer Version: Lipomas These resources from MedlinePlus offer information ab...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. It often limits the amount of food you can take in, and some types also affect how you digest and absorb nutrients. All types of weight loss surgery have risks and complications, such as infections, hernias, and blood clots. Many people lose weight quickly, but some regain weight later; however, following diet and exercise recommendations can help keep most of the weight off. Ongoing medical follow-up is required for the rest of your life [medquad_ans_0000969-1__dup2].

**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

### Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.753, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000889-2` (score=0.7001) - Saethre-Chotzen syndrome has an estimated prevalence of 1 in 25,000 to 50,000 people.
2. `medquad_ans_0000889-5` (score=0.6934) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Sae...
3. `medquad_ans_0000889-4` (score=0.6456) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
4. `medquad_ans_0001186-4` (score=0.5792) - How is chorea-acanthocytosis treated? There are currently no treatments to prevent or slow the progression of chorea-acanthocytosis; treatment is symptomatic and supportive. Management may include: botulinum toxin for...
5. `medquad_ans_0000889-1` (score=0.5733) - Saethre-Chotzen syndrome is a genetic condition characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the ...

**Answer**:

The provided context does not mention specific treatments for Saethre-Chotzen syndrome. While it details the diagnosis, inheritance pattern, and clinical features, no information is given about therapeutic interventions or management strategies. Therefore, based on the available context, the treatments for Saethre-Chotzen syndrome cannot be determined.

[medquad_ans_0000889-5] mentions resources related to diagnosis and management, such as genetic counseling, surgery and rehabilitation, and drug therapy, but does not specify treatments for Saethre-Chotzen syndrome. Thus, the exact treatments are not explicitly stated in the provided context.

**Supported citations**: `medquad_ans_0000889-5`
**Unsupported citations**: _none_

### Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=1.000, groundedness=0.897, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000512-5` (score=0.7007) - These resources address the diagnosis or management of idiopathic pulmonary fibrosis: - Gene Review: Gene Review: Pulmonary Fibrosis, Familial - Genetic Testing Registry: Idiopathic fibrosing alveolitis, chronic form ...
2. `medquad_ans_0000512-1` (score=0.6993) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effect...
3. `medquad_ans_0000078-1__dup2` (score=0.6991) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...
4. `medquad_ans_0000758-1__dup2` (score=0.6789) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
5. `medquad_ans_0000512-2` (score=0.6641) - Idiopathic pulmonary fibrosis has an estimated prevalence of 13 to 20 per 100,000 people worldwide. About 100,000 people are affected in the United States, and 30,000 to 40,000 new cases are diagnosed each year. Famil...

**Answer**:

Idiopathic pulmonary fibrosis is a chronic, progressive lung disease in which scar tissue (fibrosis) builds up in the lungs, impairing their ability to transport oxygen into the bloodstream effectively [medquad_ans_0000512-1]. It typically affects individuals between the ages of 50 and 70 and is characterized by symptoms such as shortness of breath, a persistent dry, hacking cough, loss of appetite, and gradual weight loss. Some individuals may also develop clubbing of the fingers and toes due to oxygen shortage. The disease progresses over time, leading to reduced lung function and potentially serious complications like lung cancer, pulmonary emboli, pneumonia, or pulmonary hypertension. Most people with idiopathic pulmonary fibrosis survive 3 to 5 years after diagnosis, although the disease course is highly variable, with some living with it for a decade or longer. While most cases are sporadic and occur in only one person in

**Supported citations**: `medquad_ans_0000512-1`
**Unsupported citations**: _none_

### Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.862, answer_relevance=0.617

**Top docs**:

1. `medquad_ans_0000213-3__dup1` (score=0.7319) - Early identification of and treatment for individuals with neuroleptic malignant syndrome improves outcome. If clinically indicated, a low potency neuroleptic can be reintroduced very slowly when the individual recove...
2. `medquad_ans_0004379-1` (score=0.7205) - Neuroleptic malignant syndrome is a rare neurological condition that is caused by an adverse reaction to neuroleptic (tranquilizer) or antipsychotic drugs. These drugs are commonly prescribed for the treatment of schi...
3. `medquad_ans_0000213-4__dup1` (score=0.7068) - The NINDS supports research on neurological disorders such as neuroleptic malignant syndrome. Much of this research focuses on finding ways to prevent and treat the disorder.
4. `medquad_ans_0000273-4__dup1` (score=0.5753) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts TSC research in its laboratories at the National Institutes of Health (NIH) and also supports TSC research through grants to major medical i...
5. `medquad_ans_0000022-4__dup1` (score=0.5692) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system, and to use that knowledge to reduce the burden of neurological di...

**Answer**:

The National Institute of Neurological Disorders and Stroke (NINDS) supports research on neurological disorders such as neuroleptic malignant syndrome, focusing on finding ways to prevent and treat the disorder [medquad_ans_0000213-4__dup1]. However, specific details about ongoing clinical trials or particular research studies on neuroleptic malignant syndrome are not provided in the context.

**Supported citations**: `medquad_ans_0000213-4__dup1`
**Unsupported citations**: _none_

### Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Scores**: context_relevance=0.714, groundedness=0.846, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000203-11` (score=0.7344) - - A kidney stone is a solid piece of material that forms in a kidney when substances that are normally found in the urine become highly concentrated. - Kidney stones are one of the most common disorders of the urinary...
2. `medquad_ans_0000540-1__dup2` (score=0.7315) - A kidney stone is a solid piece of material that forms in the kidney from substances in the urine. It may be as small as a grain of sand or as large as a pearl. Most kidney stones pass out of the body without help fro...
3. `medquad_ans_0000221-8` (score=0.7192) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other he...
4. `medquad_ans_0000221-12` (score=0.7179) - - A kidney stone is a solid piece of material that forms in a kidney when there are high levels of certain substances in the urine. These substances are normally found in the urine and do not cause problems at lower l...
5. `medquad_ans_0000204-11` (score=0.6678) - - A kidney stone is a solid piece of material that forms in a kidney when some substances that are normally found in the urine become highly concentrated. - Kidney stones occur in infants, children, and teenagers from...

**Answer**:

To diagnose kidney stones in adults, a healthcare provider will perform a physical exam and take a medical history, including questions about family history, diet, digestive problems, and other health issues. The provider may also perform urine, blood, and imaging tests to complete the diagnosis. Urine tests can identify infections or substances in the urine that contribute to stone formation, blood tests can detect underlying conditions that may lead to stones, and imaging tests are used to visualize the kidneys and urinary tract [medquad_ans_0000203-11][medquad_ans_0000221-8][medquad_ans_0000540-1__dup2].

**Supported citations**: `medquad_ans_0000203-11`, `medquad_ans_0000221-8`, `medquad_ans_0000540-1__dup2`
**Unsupported citations**: _none_

### Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.541, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000041_1-1` (score=0.7461) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of ...
2. `medquad_ans_0000099-1__dup2` (score=0.6706) - The bladder is a hollow organ in your lower abdomen that stores urine. Bladder cancer occurs in the lining of the bladder. It is the sixth most common type of cancer in the United States. Symptoms include - Blood in y...
3. `medquad_ans_0006289-1` (score=0.6568) - Urachal cancer is a rare type of bladder cancer, making up less than 1% of all bladder cancers. Only about 350 cases have been described in the medical literature to date. The urachus is a primitive structure which be...
4. `medquad_ans_0000041_1-2` (score=0.6340) - Signs of urethral cancer include bleeding or trouble with urination. These and other signs and symptoms may be caused by urethral cancer or by other conditions. There may be no signs or symptoms in the early stages. C...
5. `medquad_ans_0000744-1__dup1` (score=0.6318) - Bladder cancer is a form of cancer that occurs due to abnormal and uncontrolled cell growth in the bladder. Signs and symptoms of the condition may include abdominal pain, blood in the urine, fatigue, painful urinatio...

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. The urethra is the tube that carries urine from the bladder to outside the body. In women, the urethra is about 1 inch long and is just above the vagina. In men, the urethra is about 8 inches long, and goes through the prostate gland and the penis to the outside of the body. In men, the urethra also carries semen. Urethral cancer is a rare cancer that occurs more often in men than in women. There are different types of urethral cancer that begin in cells that line the urethra, named for the types of cells that become malignant: squamous cell carcinoma, transitional cell carcinoma, and adenocarcinoma. Squamous cell carcinoma is the most common type and forms in cells near the bladder in women and in the lining

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Scores**: context_relevance=0.750, groundedness=0.006, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0004671-4` (score=0.5425) - How is pachyonychia congenita inherited? Pachyonychia congenita (PC) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of the responsible...
2. `medquad_ans_0006446-4` (score=0.5210) - Is Williams syndrome inherited?
3. `medquad_ans_0004750-4` (score=0.5069) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is pass...
4. `medquad_ans_0004784-4` (score=0.4957) - How is Pelizaeus-Merzbacher disease inherited?
5. `medquad_ans_0005548-4` (score=0.4893) - Is septo-optic dysplasia inherited?

**Answer**:

The provided context does not contain information about the inheritance of paramyotonia congenita. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.530, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000735-2__dup1` (score=0.7594) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound bio...
2. `medquad_ans_0000735-1__dup1` (score=0.6940) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. The disorder may become apparent in the first few months of life, or later in childhood. The more severe form ...
3. `medquad_ans_0000125-1` (score=0.6807) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. If this condition is not recognized and treated, its signs and symptoms typically appear within the first few ...
4. `medquad_ans_0000125-2` (score=0.6394) - Profound or partial biotinidase deficiency occurs in approximately 1 in 60,000 newborns
5. `medquad_ans_0000484-1` (score=0.6090) - Holocarboxylase synthetase deficiency is an inherited disorder in which the body is unable to use the vitamin biotin effectively. This disorder is classified as a multiple carboxylase deficiency, a group of disorders ...

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), reduced consciousness/confusion (7.5%), respiratory insufficiency (7.5%), skin ulcer (7.5%), and visual field defect (7.5%). Other associated symptoms may include feeding difficulties in infancy, recurrent skin infections, seborrheic dermatitis, sensorineural hearing impairment, hepatomegaly, hyperammonemia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Scores**: context_relevance=0.714, groundedness=0.886, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000966-1__dup2` (score=0.5300) - Summary : We all need clean water. People need it to grow crops and to operate factories, and for drinking and recreation. Fish and wildlife depend on it to survive. Many different pollutants can harm our rivers, stre...
2. `medquad_ans_0000298-1__dup1` (score=0.5004) - Summary : We all need to drink water. How much you need depends on your size, activity level, and the weather where you live. The water you drink is a combination of surface water and groundwater. Surface water includ...
3. `medquad_ans_0000331-1__dup1` (score=0.4887) - Summary : Our environment affects our health. If parts of the environment, like the air, water, or soil become polluted, it can lead to health problems. For example, asthma pollutants and chemicals in the air or in th...
4. `medquad_ans_0000019-1__dup3` (score=0.4104) - Summary : Air pollution is a mixture of solid particles and gases in the air. Car emissions, chemicals from factories, dust, pollen and mold spores may be suspended as particles. Ozone, a gas, is a major part of air p...
5. `medquad_ans_0000661-1__dup1` (score=0.3767) - Summary : Oil spills often happen because of accidents, when people make mistakes or equipment breaks down. Other causes include natural disasters or deliberate acts. Oil spills have major environmental and economic e...

**Answer**:

Yes, there is information about water pollution. The three most common pollutants affecting rivers, streams, lakes, and oceans are soil, nutrients, and bacteria. Rain washes soil into water bodies, which can kill fish eggs, clog fish gills, and block light, causing plant death. Nutrients from fertilizers cause algae growth, turning water green, especially in lakes, ponds, and reservoirs. Bacteria from sewage spills can pollute both fresh and salt water. To help protect water supplies, one should not pour household products like cleansers, medicines, or grease down the drain, and instead dispose of them at hazardous waste sites. Excess grease, diapers, condoms, and pet waste should be thrown in the garbage. Pet waste contains nutrients and germs that can contribute to water pollution [medquad_ans_0000966-1__dup2].

**Supported citations**: `medquad_ans_0000966-1__dup2`
**Unsupported citations**: _none_

### Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Scores**: context_relevance=0.857, groundedness=0.987, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000132-2__dup1` (score=0.6301) - Treatment depends on the specific cause of foot drop. The most common treatment is to support the foot with light-weight leg braces and shoe inserts, called ankle-foot orthotics. Exercise therapy to strengthen the mus...
2. `medquad_ans_0000132-3__dup1` (score=0.6248) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is...
3. `medquad_ans_0000132-1__dup1` (score=0.5850) - Foot drop describes the inability to raise the front part of the foot due to weakness or paralysis of the muscles that lift the foot. As a result, individuals with foot drop scuff their toes along the ground or bend t...
4. `medquad_ans_0000132-4__dup1` (score=0.5673) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to the neurological conditions that cause foot drop in its laboratories at the National Institutes of Health (NIH), and als...
5. `medquad_ans_0006002-1` (score=0.5058) - Talipes equinovarus is a congenital (present from birth) condition where the foot turns inward and downward. The cause of this condition is not known, although it may be passed down through families in some cases. Thi...

**Answer**:

The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1].

**Supported citations**: `medquad_ans_0000132-3__dup1`
**Unsupported citations**: _none_

### Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Scores**: context_relevance=1.000, groundedness=0.511, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000145-2` (score=0.8364) - The prevalence of Camurati-Engelmann disease is unknown. Approximately 200 cases have been reported worldwide.
2. `medquad_ans_0000916-5__dup1` (score=0.6897) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene...
3. `medquad_ans_0000145-5` (score=0.6421) - These resources address the diagnosis or management of Camurati-Engelmann disease: - Gene Review: Gene Review: Camurati-Engelmann Disease - Genetic Testing Registry: Diaphyseal dysplasia These resources from MedlinePl...
4. `medquad_ans_0000145-1` (score=0.6353) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and h...
5. `medquad_ans_0000916-2__dup1` (score=0.6044) - What are the signs and symptoms of Camurati-Engelmann disease? People with Camurati-Engelmann disease have increased bone density, particularly affecting the long bones of the arms and legs (tibia, femur, humerus, uln...

**Answer**:

Camurati-Engelmann disease is a condition that mainly affects the bones, characterized by increased bone density, particularly in the long bones of the arms and legs. It can also affect the skull and hip bones. The thickened bones may lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. Increased skull density can cause increased pressure on the brain, resulting in neurological problems such as headaches, hearing loss, vision problems, dizziness (vertigo), ringing in the ears (tinnitus), and facial paralysis. Additional features include abnormal curvature of the spine (scoliosis), joint deformities (contractures), knock knees, flat feet (pes planus), abnormally long limbs in proportion to height, decreased muscle mass and body fat, and delayed puberty. The age of symptom onset varies, with most individuals developing pain or weakness by adolescence. In some cases, individuals may have the gene mutation

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Scores**: context_relevance=1.000, groundedness=0.974, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0000316-2` (score=0.7433) - X-linked Emery-Dreifuss muscular dystrophy is the most common form of this condition, affecting an estimated 1 in 100,000 people. The autosomal recessive type of this disorder appears to be very rare; only a few cases...
2. `medquad_ans_0002080-2` (score=0.7389) - What are the signs and symptoms of Emery-Dreifuss muscular dystrophy, X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Emery-Dreifuss muscular dystrophy, X-linked. If the in...
3. `medquad_ans_0000316-1` (score=0.7322) - Emery-Dreifuss muscular dystrophy is a condition that chiefly affects muscles used for movement (skeletal muscles) and heart (cardiac) muscle. Among the earliest features of this disorder are joint deformities called ...
4. `medquad_ans_0000316-5` (score=0.6893) - These resources address the diagnosis or management of Emery-Dreifuss muscular dystrophy: - Gene Review: Gene Review: Emery-Dreifuss Muscular Dystrophy - Genetic Testing Registry: Emery-Dreifuss muscular dystrophy - G...
5. `medquad_ans_0002213-2` (score=0.6279) - What are the signs and symptoms of Facioscapulohumeral muscular dystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Facioscapulohumeral muscular dystrophy. If the information ...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked include abnormality of the neck, Achilles tendon contracture, atrioventricular block, childhood onset, decreased cervical spine flexion due to contractures of posterior cervical muscles, elbow flexion contracture, elevated serum creatine phosphokinase, juvenile onset, pectus excavatum, primary atrial arrhythmia, slow progression, sudden cardiac death, type 1 muscle fiber atrophy, waddling gait, and X-linked recessive inheritance [medquad_ans_0002080-2]. Among the earliest features are joint deformities called contractures, which restrict movement and become noticeable in early childhood, most often involving the elbows, ankles, and neck [medquad_ans_0000316-1]. Most affected individuals also experience slowly progressive muscle weakness and wasting, beginning in the muscles of the upper arms and lower legs

**Supported citations**: `medquad_ans_0002080-2`, `medquad_ans_0000316-1`
**Unsupported citations**: _none_

### Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.824, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000072-2` (score=0.7196) - Arterial tortuosity syndrome is a rare disorder; its prevalence is unknown. About 100 cases have been reported in the medical literature.
2. `medquad_ans_0000072-5` (score=0.7015) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins Mc...
3. `medquad_ans_0000072-1` (score=0.6526) - Arterial tortuosity syndrome is a disorder that affects connective tissue. Connective tissue provides strength and flexibility to structures throughout the body, including blood vessels, skin, joints, and the gastroin...
4. `medquad_ans_0003631-5` (score=0.5497) - How might lipodermatosclerosis be treated? Lipodermatosclerosis is primarily treated with compression therapy to improve venous insufficiency. Other strategies for managing venous insufficiency include leg elevation; ...
5. `medquad_ans_0000089-6` (score=0.5398) - Marfan syndrome has no cure. However, treatments can help delay or prevent complications, especially when started early. Marfan syndrome can affect many parts of your body, including your heart, bones and joints, eyes...

**Answer**:

The provided context does not specify treatments for arterial tortuosity syndrome. While it mentions resources that address the diagnosis or management of the disorder, such as genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, and palliative care, it does not detail specific treatments [medquad_ans_0000072-5]. Therefore, based on the given information, the exact treatments cannot be determined.

**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

### Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001093-1__dup1` (score=0.7972) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is ava...
2. `medquad_ans_0001092-1__dup1` (score=0.7956) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
3. `medquad_ans_0001094-2__dup1` (score=0.7527) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1E? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1E. If the information is ava...
4. `medquad_ans_0001088-2__dup1` (score=0.7482) - What are the signs and symptoms of Charcot-Marie-Tooth disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease. If the information is available, the tabl...
5. `medquad_ans_0006492-1` (score=0.7445) - What are the signs and symptoms of X-linked Charcot-Marie-Tooth disease type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked Charcot-Marie-Tooth disease type 1. If the in...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 1C include decreased motor nerve conduction velocity, distal amyotrophy, distal muscle weakness, distal sensory impairment, hypertrophic nerve changes, hyporeflexia, juvenile onset, onion bulb formation, pes cavus, and segmental peripheral demyelination/remyelination [medquad_ans_0001092-1__dup1].

**Supported citations**: `medquad_ans_0001092-1__dup1`
**Unsupported citations**: _none_

### Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.890, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005321-1` (score=0.7883) - What are the signs and symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal arterial macroaneurysm...
2. `medquad_ans_0000870-1` (score=0.7097) - Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) is a disorder that affects blood vessels in the eyes and heart. The condition generally becomes apparent in infancy or childhood. RAMSVPS d...
3. `medquad_ans_0000870-5` (score=0.6129) - These resources address the diagnosis or management of RAMSVPS: - Calgary Retina Consultants: Retinal Arterial Macroaneurysm - Genetic Testing Registry: Retinal arterial macroaneurysm with supravalvular pulmonic steno...
4. `medquad_ans_0005330-2` (score=0.5928) - What are the signs and symptoms of Retinal vasculopathy with cerebral leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal vasculopathy with cerebral leukodystroph...
5. `medquad_ans_0003142-2` (score=0.5918) - What are the signs and symptoms of idiopathic juxtafoveal retinal telangiectasia? Signs and symptoms of idiopathic juxtafoveal retinal telangiectasia may include slow loss of vision, distorted vision, trouble reading,...

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) include exudative retinal detachment and pulmonic stenosis. Retinal arterial macroaneurysms can rupture, leading to bleeding within the eye that may cause vision loss. Additionally, supravalvular pulmonic stenosis results in narrowing of the pulmonary artery above the pulmonic valve, impairing blood flow to the lungs and reducing oxygenation of the blood. This leads to symptoms such as shortness of breath. [medquad_ans_0005321-1][medquad_ans_0000870-1]

**Supported citations**: `medquad_ans_0005321-1`, `medquad_ans_0000870-1`
**Unsupported citations**: _none_

### Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Scores**: context_relevance=1.000, groundedness=0.914, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001838-2` (score=0.6276) - What are the signs and symptoms of Dicarboxylic aminoaciduria? There are no common signs or symptoms of dicarboxylic aminoaciduria. Hypoglycemia, developmental and neurological abnormalities, and obsessive compulsive ...
2. `medquad_ans_0000050-1` (score=0.5551) - Aminoacylase 1 deficiency is an inherited disorder that can cause neurological problems; the pattern and severity of signs and symptoms vary widely among affected individuals. Individuals with this condition typically...
3. `medquad_ans_0000067-3` (score=0.5373) - Mutations in the ASL gene cause argininosuccinic aciduria. Argininosuccinic aciduria belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occur in liver ce...
4. `medquad_ans_0002894-1` (score=0.5121) - What are the signs and symptoms of Histidinuria renal tubular defect? The Human Phenotype Ontology provides the following list of signs and symptoms for Histidinuria renal tubular defect. If the information is availab...
5. `medquad_ans_0000168-7` (score=0.5044) - - Proteinuria is a condition in which urine contains a detectable amount of protein. - Proteinuria is a sign of chronic kidney disease (CKD). - Groups at risk for proteinuria include African Americans, American Indian...

**Answer**:

Dicarboxylic aminoaciduria is a condition characterized by the presence of dicarboxylic amino acids in the urine. It is associated with no common signs or symptoms; however, in individual cases, hypoglycemia, developmental and neurological abnormalities, and obsessive compulsive tendencies have been reported. Some individuals diagnosed with this condition have virtually no signs or symptoms. Aminoaciduria, with autosomal recessive inheritance, and fasting hypoglycemia, intellectual disability, are among the reported signs and symptoms, with frequencies estimated based on studies [medquad_ans_0001838-2]. The Human Phenotype Ontology (HPO) compiles information on the occurrence of these features in patients with the condition [medquad_ans_0001838-2].

**Supported citations**: `medquad_ans_0001838-2`
**Unsupported citations**: _none_

### Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Scores**: context_relevance=1.000, groundedness=0.684, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0005859-1` (score=0.7895) - Spondylospinal thoracic dysostosis is an extremely rare skeletal disorder characterized by a short, curved spine and fusion of the spinous processes, short thorax with 'crab-like' configuration of the ribs, underdevel...
2. `medquad_ans_0000949-5` (score=0.7817) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from Medli...
3. `medquad_ans_0000944-2` (score=0.7226) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.
4. `medquad_ans_0005819-1` (score=0.7221) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
5. `medquad_ans_0005822-1` (score=0.7180) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...

**Answer**:

Spondylothoracic dysostosis is believed to be inherited in an autosomal recessive manner [medquad_ans_0005859-1]. However, it is important to note that spondylothoracic dysostosis may be a misstatement or confusion with spondylocostal dysostosis, as the provided context primarily discusses spondylocostal dysostosis, which is inherited in an autosomal recessive manner [medquad_ans_0005819-1]. Given the lack of direct evidence linking spondylothoracic dysostosis to a specific inheritance pattern in the provided context, the answer is based on the assumption that the two conditions are being conflated. Therefore, based on the available information, spondylothoracic dysostosis is believed to be inherited in an autosomal recessive manner [medquad_ans_0005

**Supported citations**: `medquad_ans_0005859-1`, `medquad_ans_0005819-1`
**Unsupported citations**: _none_

### Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.313, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003296-1` (score=0.7763) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. I...
2. `medquad_ans_0000542-1` (score=0.7734) - Jervell and Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). This disorder is a form of long QT syndrome, which is a heart ...
3. `medquad_ans_0000542-2` (score=0.7628) - Jervell and Lange-Nielsen syndrome is uncommon; it affects an estimated 1.6 to 6 per 1 million people worldwide. This condition has a higher prevalence in Denmark, where it affects at least 1 in 200,000 people.
4. `medquad_ans_0000542-5` (score=0.6801) - These resources address the diagnosis or management of Jervell and Lange-Nielsen syndrome: - Gene Review: Gene Review: Jervell and Lange-Nielsen Syndrome - Genetic Testing Registry: Jervell and Lange-Nielsen syndrome ...
5. `medquad_ans_0000241-1` (score=0.5969) - Cornelia de Lange syndrome is a developmental disorder that affects many parts of the body. The features of this disorder vary widely among affected individuals and range from relatively mild to severe. Cornelia de La...

**Answer**:

Jervell Lange-Nielsen syndrome is a form of long QT syndrome that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. The condition is inherited in an autosomal recessive fashion and has two different types: type 1 and type 2. It is uncommon, affecting an estimated 1.6 to 6 per 1 million people worldwide, with a higher prevalence in Denmark, where it affects at least 1 in 200,000 people [medquad_ans_0003296-1, medquad_ans_0000542-1, medquad_ans_0000542-2].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0003296-1, medquad_ans_0000542-1, medquad_ans_0000542-2`

### Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Scores**: context_relevance=1.000, groundedness=0.768, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000218-5` (score=0.6748) - These resources address the diagnosis or management of CCFDN: - Gene Review: Gene Review: Congenital Cataracts, Facial Dysmorphism, and Neuropathy - Genetic Testing Registry: Congenital Cataracts, Facial Dysmorphism, ...
2. `medquad_ans_0000218-1` (score=0.6691) - Congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is a rare disorder that affects several parts of the body. It is characterized by a clouding of the lens of the eyes at birth (congenital cataracts) and...
3. `medquad_ans_0000507-2` (score=0.6430) - The prevalence of hypomyelination and congenital cataract is unknown.
4. `medquad_ans_0001004-1__dup1` (score=0.6172) - What are the signs and symptoms of Cataract, total congenital? The Human Phenotype Ontology provides the following list of signs and symptoms for Cataract, total congenital. If the information is available, the table ...
5. `medquad_ans_0000998-1__dup1` (score=0.6078) - What are the signs and symptoms of Cataract, autosomal recessive congenital 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Cataract, autosomal recessive congenital 2. If the info...

**Answer**:

The provided context does not specify the number of people affected by congenital cataracts, facial dysmorphism, and neuropathy (CCFDN). [medquad_ans_0000218-1] describes the condition's features and complications but does not include population prevalence or affected numbers. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000218-1`
**Unsupported citations**: _none_

### Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Scores**: context_relevance=0.846, groundedness=0.938, answer_relevance=0.538

**Top docs**:

1. `medquad_ans_0000259-3__dup1` (score=0.6978) - How is alpha-thalassemia x-linked intellectual disability syndrome inherited? Alpha-thalassemia x-linked intellectual disability (ATRX) syndrome is caused by a mutation in the ATRX gene and is inherited in an x-linked...
2. `medquad_ans_0000127-1__dup2` (score=0.6407) - Thalassemias (thal-a-SE-me-ahs) are inherited blood disorders. "Inherited" means that the disorder is passed from parents to children through genes. Thalassemias cause the body to make fewer healthy red blood cells an...
3. `medquad_ans_0003225-3` (score=0.6240) - How is intellectual disability-developmental delay-contractures syndrome inherited? Intellectual disability-developmental delay-contractures syndrome syndrome is inherited in an X-linked recessive manner and is caused...
4. `medquad_ans_0000127-4__dup2` (score=0.5821) - Family history and ancestry are the two risk factors for thalassemias. Family History Thalassemias are inheritedthat is, the genes for the disorders are passed from parents to their children. If your parents have miss...
5. `medquad_ans_0000381-1__dup1` (score=0.5815) - Fragile X syndrome is the most common form of inherited developmental disability. A problem with a specific gene causes the disease. Normally, the gene makes a protein you need for brain development. But the problem c...

**Answer**:

Alpha-thalassemia x-linked intellectual disability (ATRX) syndrome is caused by a mutation in the ATRX gene and is inherited in an x-linked manner [medquad_ans_0000259-3__dup1]. The inheritance pattern depends on whether the mutation was inherited from the mother or occurred de novo. If the mutation is de novo, the risk to relatives is very low. If inherited from the mother, each of her sisters has a 50% chance of being a carrier. A carrier woman has a 25% chance of having a son affected with ATRX syndrome, a 25% chance of having a son without the mutation, a 25% chance of having a daughter who is a carrier, and a 25% chance of having a daughter who is not a carrier [medquad_ans_0000259-3__dup1].

**Supported citations**: `medquad_ans_0000259-3__dup1`
**Unsupported citations**: _none_

### Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0002622-1` (score=0.7559) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepa...
2. `medquad_ans_0002622-2` (score=0.7494) - What are the signs and symptoms of Glycogen storage disease type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 6. If the information is available, ...
3. `medquad_ans_0002615-2` (score=0.7219) - What are the signs and symptoms of Glycogen storage disease type 13? Glycogen storage disease type 13 causes muscle pain (myalgia). Individuals with GSD13 also experience exercise intolerance, which means they have di...
4. `medquad_ans_0002614-1` (score=0.7104) - What are the signs and symptoms of Glycogen storage disease type 12? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 12. If the information is available...
5. `medquad_ans_0002623-2` (score=0.6978) - What are the signs and symptoms of Glycogen storage disease type 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 7. If the information is available, ...

**Answer**:

The symptoms of Glycogen storage disease type 6 include low blood sugar (hypoglycemia), an enlarged liver (hepatomegaly), and increased lactic acid in the blood (lactic acidosis), particularly when an individual does not eat for a long time. Short stature and postnatal growth retardation are also associated with the condition. Hypoglycemia and short stature are reported in approximately 90% of patients [medquad_ans_0002622-2]. The condition is caused by mutations in the PYGL gene and is inherited in an autosomal recessive manner [medquad_ans_0002622-1].

**Supported citations**: `medquad_ans_0002622-2`, `medquad_ans_0002622-1`
**Unsupported citations**: _none_

### Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Scores**: context_relevance=1.000, groundedness=0.903, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000032-1__dup4` (score=0.6871) - Diabetes is a disorder of metabolismthe way our bodies use digested food for energy. Most of the food we eat is broken down into glucose, the form of sugar in the blood. Glucose is the body's main source of fuel. Afte...
2. `medquad_ans_0000035-1__dup4` (score=0.6724) - Diabetes is a complex group of diseases with a variety of causes. People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia. Diabetes is a disorder of metabolismthe way the body uses ...
3. `medquad_ans_0000071-1__dup5` (score=0.6698) - Diabetes is a complex group of diseases with a variety of causes. People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia. Diabetes is a disorder of metabolismthe way the body uses ...
4. `medquad_ans_0000070-1__dup6` (score=0.6698) - Diabetes is a complex group of diseases with a variety of causes. People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia. Diabetes is a disorder of metabolismthe way the body uses ...
5. `medquad_ans_0000037-1__dup4` (score=0.6698) - Diabetes is a complex group of diseases with a variety of causes. People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia. Diabetes is a disorder of metabolismthe way the body uses ...

**Answer**:

Diabetes is caused by the body not making enough insulin or being unable to use insulin effectively, or both [medquad_ans_0000032-1__dup4]. The pancreas, which produces insulin, may fail to generate sufficient amounts of the hormone, or the cells in the muscles, liver, and fat may not respond properly to insulin [medquad_ans_0000035-1__dup4]. This results in high blood glucose levels (hyperglycemia) because glucose remains in the bloodstream instead of being absorbed by cells for energy [medquad_ans_0000035-1__dup4].

**Supported citations**: `medquad_ans_0000032-1__dup4`, `medquad_ans_0000035-1__dup4`
**Unsupported citations**: _none_

### Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Scores**: context_relevance=0.636, groundedness=0.893, answer_relevance=0.782

**Top docs**:

1. `medquad_ans_0000016-21` (score=0.6411) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the lab...
2. `medquad_ans_0000016-8` (score=0.6208) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a com...
3. `medquad_ans_0000016-6` (score=0.6179) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
4. `medquad_ans_0000055-7` (score=0.6158) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need othe...
5. `medquad_ans_0000016-4__dup2` (score=0.6087) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...

**Answer**:

Research being done for diabetic retinopathy includes studies in the laboratory and with patients to find better ways to detect, treat, and prevent vision loss in people with diabetes. Specifically, researchers are studying drugs for the treatment of proliferative retinopathy that may reduce the need for laser surgery [medquad_ans_0000016-21]. Additionally, clinical trials are exploring treatments such as medicines injected into the eye to block proteins that cause abnormal blood vessel growth and fluid leakage, which can help treat macular edema [medquad_ans_0000055-7]. These treatments are administered by ophthalmologists and may include focal or scatter laser treatments depending on the condition [medquad_ans_0000055-7].

**Supported citations**: `medquad_ans_0000016-21`, `medquad_ans_0000055-7`
**Unsupported citations**: _none_

### Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Scores**: context_relevance=0.800, groundedness=0.963, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.6080) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart an...
2. `medquad_ans_0000007-4__dup1` (score=0.5753) - People at risk for ARDS have a condition or illness that can directly or indirectly injure their lungs. Direct Lung Injury Conditions that can directly injure the lungs include: Pneumonia. This is an infection in the ...
3. `medquad_ans_0000007-3__dup1` (score=0.5014) - Many conditions or factors can directly or indirectly injure the lungs and lead to ARDS. Some common ones are: Sepsis. This is a condition in which bacteria infect the bloodstream. Pneumonia. This is an infection in t...
4. `medquad_ans_0000137-1__dup1` (score=0.4652) - Acute respiratory distress syndrome (ARDS) is a life-threatening lung condition that prevents enough oxygen from getting to the lungs and into the blood. People who develop ARDS often are very ill with another disease...
5. `medquad_ans_0000007-1__dup1` (score=0.4622) - ARDS, or acute respiratory distress syndrome, is a lung condition that leads to low oxygen levels in the blood. ARDS can be life threatening because your body's organs need oxygen-rich blood to work well. People who d...

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles in the air from fires or toxic fumes. These substances can damage the eyes and respiratory system and may worsen chronic heart and lung diseases. Symptoms of inhalation injuries include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, the condition may worsen [medquad_ans_0000516-1__dup1].

**Supported citations**: `medquad_ans_0000516-1__dup1`
**Unsupported citations**: _none_

### Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.837, answer_relevance=0.450

**Top docs**:

1. `medquad_ans_0000134-4__dup2` (score=0.7140) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has ...
2. `medquad_ans_0000134-9` (score=0.6898) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing Zollinger-Ellison syndrome.
3. `medquad_ans_0000195-2` (score=0.6369) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0000134-3__dup2` (score=0.6328) - Experts do not know the exact cause of Zollinger-Ellison syndrome. About 25 to 30 percent of gastrinomas are caused by an inherited genetic disorder called multiple endocrine neoplasia type 1 (MEN1).1 MEN1 causes horm...
5. `medquad_ans_0000955-2` (score=0.6199) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.

**Answer**:

The incidence of Zollinger-Ellison syndrome is estimated to affect 1 in 40,000 to 50,000 people [medquad_ans_0000195-2]. Additionally, it is rare and occurs in about one in every 1 million people [medquad_ans_0000134-4__dup2]. However, the prevalence of the condition is unknown [medquad_ans_0000955-2].

**Supported citations**: `medquad_ans_0000195-2`, `medquad_ans_0000134-4__dup2`, `medquad_ans_0000955-2`
**Unsupported citations**: _none_

### Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Scores**: context_relevance=0.857, groundedness=0.736, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000974-1` (score=0.6262) - Tetrahydrobiopterin deficiency is a rare disorder characterized by a shortage (deficiency) of a molecule called tetrahydrobiopterin or BH4. This condition alters the levels of several substances in the body, including...
2. `medquad_ans_0006044-2` (score=0.6066) - What are the signs and symptoms of Tetrahydrobiopterin deficiency? Infants with tetrahydrobiopterin (BH4) deficiency typically appear normal and healthy at birth. Neurological signs and symptoms usually become apparen...
3. `medquad_ans_0006044-1` (score=0.5864) - Tetrahydrobiopterin (BH4) deficiency is a neurological condition caused by an inborn error of metabolism. BH4 is a substance in the body that enhances the action of other enzymes. Deficiency of BH4 leads to abnormally...
4. `medquad_ans_0000974-5` (score=0.5852) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Ge...
5. `medquad_ans_0000974-3` (score=0.5756) - Tetrahydrobiopterin deficiency can be caused by mutations in one of several genes, including GCH1, PCBD1, PTS, and QDPR. These genes provide instructions for making enzymes that help produce and recycle tetrahydrobiop...

**Answer**:

The provided context does not mention any treatments for tetrahydrobiopterin deficiency. Therefore, based on the available information, it is not possible to answer the question about treatments. [medquad_ans_0000974-1] [medquad_ans_0006044-2]

**Supported citations**: `medquad_ans_0000974-1`, `medquad_ans_0006044-2`
**Unsupported citations**: _none_

### Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Scores**: context_relevance=0.750, groundedness=0.887, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000736-2` (score=0.8022) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
2. `medquad_ans_0000150-2__dup1` (score=0.6439) - There is no standard course of treatment for holoprosencephaly. Treatment is symptomatic and supportive.
3. `medquad_ans_0000736-1` (score=0.6321) - Nonsyndromic holoprosencephaly is an abnormality of brain development that also affects the head and face. Normally, the brain divides into two halves (hemispheres) during early development. Holoprosencephaly occurs w...
4. `medquad_ans_0000195-2` (score=0.6255) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
5. `medquad_ans_0000955-2` (score=0.6163) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.

**Answer**:

Holoprosencephaly affects an estimated 1 in 10,000 newborns, and nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases [medquad_ans_0000736-2]. Therefore, the number of people affected by nonsyndromic holoprosencephaly can be estimated as 25 to 50 percent of 1 in 10,000 newborns. However, the exact number of individuals affected is not specified in the provided context. [medquad_ans_0000736-2]

**Supported citations**: `medquad_ans_0000736-2`
**Unsupported citations**: _none_

### Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Scores**: context_relevance=0.727, groundedness=0.796, answer_relevance=0.873

**Top docs**:

1. `medquad_ans_0000525-4` (score=0.7027) - Susceptibility to intrahepatic cholestasis of pregnancy is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to increase the risk of developing the disorde...
2. `medquad_ans_0000118-5` (score=0.6864) - These resources address the diagnosis or management of benign recurrent intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Benign recurrent intrahepatic cholestasis 1 -...
3. `medquad_ans_0000829-5` (score=0.6803) - These resources address the diagnosis or management of progressive familial intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Progressive familial intrahepatic cholest...
4. `medquad_ans_0005084-1` (score=0.6773) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching...
5. `medquad_ans_0003240-1` (score=0.6456) - Intrahepatic cholestasis of pregnancy (ICP) is a disorder of the liver that occurs in women during pregnancy. Cholestasis is a condition that impairs the release of bile (a digestive juice) from liver cells. The bile ...

**Answer**:

The genetic changes related to progressive familial intrahepatic cholestasis type 2 (PFIC2) are mutations in the ABCB11 gene [medquad_ans_0005084-1]. PFIC2 is inherited in an autosomal recessive manner, meaning two copies of the altered gene (one from each parent) are required to develop the condition [medquad_ans_0005084-1]. The provided context does not specify genetic changes for other types of progressive familial intrahepatic cholestasis (e.g., PFIC1 or PFIC3), only that PFIC2 is associated with ABCB11 gene mutations.

Thus, the genetic change specifically identified is mutations in the ABCB11 gene for PFIC2 [medquad_ans_0005084-1].

**Supported citations**: `medquad_ans_0005084-1`
**Unsupported citations**: _none_

### Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.990, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.7538) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below in...
2. `medquad_ans_0003814-2` (score=0.6581) - What are the signs and symptoms of Marshall-Smith syndrome? Marshall-Smith syndrome is characterized by accelerated skeletal maturation, relative failure to thrive, respiratory difficulties, mental retardation, and un...
3. `medquad_ans_0003845-2` (score=0.6490) - What are the signs and symptoms of McCune Albright syndrome? People with McCune Albright syndrome (MAS) may have symptoms related to bones, the endocrine system, and/or skin. The symptoms can range from mild to severe...
4. `medquad_ans_0001983-1` (score=0.5856) - What are the signs and symptoms of Dyssegmental dysplasia Silverman-Handmaker type? The Human Phenotype Ontology provides the following list of signs and symptoms for Dyssegmental dysplasia Silverman-Handmaker type. I...
5. `medquad_ans_0002658-1` (score=0.5828) - What are the signs and symptoms of Gracile bone dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Gracile bone dysplasia. If the information is available, the table below in...

**Answer**:

The signs and symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

### Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Scores**: context_relevance=1.000, groundedness=0.857, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000453-1` (score=0.7424) - Hepatic veno-occlusive disease with immunodeficiency (also called VODI) is a hereditary disorder of the liver and immune system. Its signs and symptoms appear after the first few months of life. Hepatic veno-occlusive...
2. `medquad_ans_0000453-5` (score=0.6314) - These resources address the diagnosis or management of VODI: - Gene Review: Gene Review: Hepatic Veno-Occlusive Disease with Immunodeficiency - Genetic Testing Registry: Hepatic venoocclusive disease with immunodefici...
3. `medquad_ans_0002804-1` (score=0.5597) - What are the signs and symptoms of Hepatic venoocclusive disease with immunodeficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Hepatic venoocclusive disease with immunodefic...
4. `medquad_ans_0000324-4` (score=0.4972) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
5. `medquad_ans_0005558-3` (score=0.4887) - How is severe combined immunodeficiency (SCID) diagnosed? A diagnosis of severe combined immunodeficiency (SCID) may be suspected if a baby shows any of the following persistent symptoms within the first year of life:...

**Answer**:

Yes, hepatic veno-occlusive disease with immunodeficiency is inherited in an autosomal recessive pattern. This is supported by the presence of "autosomal recessive inheritance" in the list of signs and symptoms provided in the context [medquad_ans_0002804-1].

**Supported citations**: `medquad_ans_0002804-1`
**Unsupported citations**: _none_

### Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Scores**: context_relevance=1.000, groundedness=0.914, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005902-1` (score=0.6377) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how o...
2. `medquad_ans_0001768-1` (score=0.6098) - What are the signs and symptoms of Dehydrated hereditary stomatocytosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Dehydrated hereditary stomatocytosis. If the information is a...
3. `medquad_ans_0000013_3-2__dup4` (score=0.5632) - Signs and symptoms of juvenile myelomonocytic leukemia include fever, weight loss, and feeling very tired. These and other signs and symptoms may be caused by JMML or by other conditions. Check with your doctor if you...
4. `medquad_ans_0002784-1` (score=0.5571) - What are the signs and symptoms of Hemophagocytic lymphohistiocytosis, familial, 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Hemophagocytic lymphohistiocytosis, familial, 4. I...
5. `medquad_ans_0002781-2` (score=0.5560) - What are the signs and symptoms of Hemophagocytic lymphohistiocytosis? The signs and symptoms of hemophagocytic lymphohistiocytosis typically develop during the first months or years of life. However, in rare cases, a...

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. These signs and symptoms are based on the Human Phenotype Ontology, which compiles data from sources such as Orphanet [medquad_ans_0005902-1].

**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

### Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Scores**: context_relevance=0.667, groundedness=0.912, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000159-8` (score=0.7001) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed...
2. `medquad_ans_0003688-1` (score=0.6640) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, h...
3. `medquad_ans_0000159-6` (score=0.6589) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
4. `medquad_ans_0000159-7` (score=0.6482) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may n...
5. `medquad_ans_0000159-3__dup2` (score=0.6455) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time...

**Answer**:

For lupus nephritis, treatment involves medications that suppress the immune system to prevent damage to the kidneys. Standard treatment typically includes a corticosteroid, such as prednisone, to reduce inflammation and an immunosuppressive medication, such as cyclophosphamide or mycophenolate mofetil, used in combination with prednisone. If high blood pressure is present, medications to lower blood pressure are also prescribed, as this can slow the progression of kidney disease. In some cases, treatment may involve dialysis or a kidney transplant. Early diagnosis and treatment are crucial to prevent chronic kidney damage and complications such as kidney failure, which affects 10 to 30 percent of people with lupus nephritis [medquad_ans_0000159-8][medquad_ans_0003688-1]. People with lupus nephritis are also at a higher risk for certain

**Supported citations**: `medquad_ans_0000159-8`, `medquad_ans_0003688-1`
**Unsupported citations**: _none_

### Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.510, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000114-6` (score=0.7598) - Restless legs syndrome (RLS) has no cure. If a condition or medicine triggers RLS, it may go away or get better if the trigger is relieved or stopped. RLS can be treated. The goals of treatment are to: Prevent or reli...
2. `medquad_ans_0005317-1` (score=0.7007) - Restless legs syndrome is a neurological condition that causes an irresistible urge to move the legs. The movement is triggered by strange or uncomfortable feelings, which occur mostly while the affected person is sit...
3. `medquad_ans_0000776-1__dup2` (score=0.6675) - Restless legs syndrome (RLS) causes a powerful urge to move your legs. Your legs become uncomfortable when you are lying down or sitting. Some people describe it as a creeping, crawling, tingling, or burning sensation...
4. `medquad_ans_0000114-5__dup1` (score=0.6590) - Your doctor will diagnose restless legs syndrome (RLS) based on your signs and symptoms, your medical and family histories, a physical exam, and test results. Your doctor will use this information to rule out other co...
5. `medquad_ans_0000246-1__dup2` (score=0.6520) - Restless legs syndrome (RLS) is a neurological disorder characterized by unpleasant sensations in the legs and an uncontrollable, and sometimes overwhelming, urge to move them for relief. Individuals affected with the...

**Answer**:

The treatments for Restless Legs Syndrome (RLS) include lifestyle changes and, in more severe cases, daily medicines. For mild cases, lifestyle changes may be the only treatment needed. These include avoiding substances that trigger RLS, such as alcohol and tobacco, and informing your doctor about all medications you are taking, as some can cause or worsen RLS symptoms. Good sleep habits, such as keeping the sleep environment cool, quiet, dark, and comfortable, going to bed and waking up at the same time every day, and avoiding long periods of lying awake, can also help. Engaging in regular, moderate physical activity and doing mentally challenging tasks before bedtime may ease symptoms. Certain activities that relieve symptoms include walking, stretching, taking a hot or cold bath, massaging the affected limb(s), or using heat or ice packs. If lifestyle changes are insufficient, medicines may be prescribed, though no single medicine works for all people, and it may take

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Scores**: context_relevance=0.750, groundedness=0.725, answer_relevance=0.350

**Top docs**:

1. `medquad_ans_0002575-1` (score=0.6970) - Gliomatosis cerebri is a type of brain cancer. It is a variant form of glioblastoma multiforme. It is characterized by scattered and widespread tumor cells that can cause the cerebrum, cerebellum, or brain stem to enl...
2. `medquad_ans_0002574-1` (score=0.6874) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...
3. `medquad_ans_0004525-1` (score=0.6142) - Oligoastrocytoma is a brain tumor that forms when two types of cells in the brain, called oligodendrocytes and astrocytes, rapidly increase in number to form a mass. These brain cells are known as glial cells, which n...
4. `medquad_ans_0004363-1` (score=0.5920) - Neuroblastoma is a tumor that develops from a nerve in a child, usually before the age of 5. It occurs in the abdomen near the adrenal glands, but it can also occur in other parts of the body. It is considered an aggr...
5. `medquad_ans_0000006_5-1` (score=0.5767) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - T...

**Answer**:

The provided context does not contain information about glioblastoma. While it mentions gliomatosis cerebri as a variant form of glioblastoma multiforme, it does not define or describe glioblastoma itself. Therefore, based on the available context, the answer cannot be determined.

[medquad_ans_0002575-1]

**Supported citations**: `medquad_ans_0002575-1`
**Unsupported citations**: _none_

### Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Scores**: context_relevance=0.333, groundedness=0.567, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0005496-1` (score=0.7232) - Schizencephaly is a rare congenital (present from birth) brain malformation in which abnormal slits or clefts form in the cerebral hemispheres of the brain. The signs and symptoms of this condition may include develop...
2. `medquad_ans_0000251-1__dup2` (score=0.6773) - Schizencephaly is an extremely rare developmental birth defect characterized by abnormal slits, or clefts, in the cerebral hemispheres of the brain. Babies with clefts in both hemispheres (called bilateral clefts) com...
3. `medquad_ans_0000251-3__dup1` (score=0.6007) - The prognosis for individuals with schizencephaly varies depending on the size of the clefts and the extent of neurological disabilities.
4. `medquad_ans_0000064-1__dup1` (score=0.5767) - Cephalic disorders are congenital conditions that stem from damage to or abnormal development of the budding nervous system. Most cephalic disorders are caused by a disturbance that occurs very early in the developmen...
5. `medquad_ans_0000251-4__dup1` (score=0.5631) - The NINDS conducts and supports a wide range of studies that explore the mechanisms of normal brain development. The knowledge gained from these fundamental studies provides the foundation for understanding how to pre...

**Answer**:

The exact cause of schizencephaly is unknown, but it has been linked to a variety of genetic and non-genetic factors [medquad_ans_0005496-1]. Additionally, some cephalic disorders, which include conditions affecting the development of the nervous system, may be influenced by hereditary or genetic conditions or by environmental exposures during pregnancy, such as medication taken by the mother, maternal infection, or exposure to radiation [medquad_ans_0000064-1]. However, specific environmental or genetic causes directly tied to schizencephaly are not detailed in the provided context.

**Supported citations**: `medquad_ans_0005496-1`
**Unsupported citations**: `medquad_ans_0000064-1`

### Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Scores**: context_relevance=0.900, groundedness=0.313, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000309-1__dup1` (score=0.6596) - Amyotrophic lateral sclerosis (ALS), also referred to as "Lou Gehrig's disease," is a progressive motor neuron disease which leads to problems with muscle control and movement. There are various types of ALS, which ar...
2. `medquad_ans_0000309-3__dup1` (score=0.6357) - What causes amyotrophic lateral sclerosis? In approximately 90-95% of cases the cause of amyotrophic lateral sclerosis (ALS) is unknown and is sporadic (occurring in individuals with no history of the condition in the...
3. `medquad_ans_0000052-5` (score=0.6331) - These resources address the diagnosis or management of amyotrophic lateral sclerosis: - Gene Review: Gene Review: ALS2-Related Disorders - Gene Review: Gene Review: Amyotrophic Lateral Sclerosis Overview - Gene Review...
4. `medquad_ans_0000052-3` (score=0.6317) - Mutations in several genes can cause familial ALS and contribute to the development of sporadic ALS. Mutations in the C9orf72 gene account for 30 to 40 percent of familial ALS in the United States and Europe. Worldwid...
5. `medquad_ans_0003331-3` (score=0.6311) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...

**Answer**:

Genetic changes related to amyotrophic lateral sclerosis (ALS) include mutations in several genes, such as C9orf72, SOD1, TARDBP, FUS, ANG, ALS2, SETX, and VAPB. These mutations can cause familial ALS and may also contribute to the development of sporadic ALS. Approximately 60% of individuals with familial ALS have an identifiable genetic mutation, while the genetic cause in the remaining cases is unknown. The genes associated with ALS appear to play a role in neuronal function or in regulating the production of various proteins [doc_id: medquad_ans_0000309-3__dup1].

**Supported citations**: _none_
**Unsupported citations**: `doc_id: medquad_ans_0000309-3__dup1`

### Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.917, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0003473-6` (score=0.7675) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injectio...
2. `medquad_ans_0000581-5` (score=0.6855) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Te...
3. `medquad_ans_0003473-1` (score=0.6762) - Laron syndrome is a condition that occurs when the body is unable to utilize growth hormone. It is primarily characterized by short stature. Other signs and symptoms vary but may include reduced muscle strength and en...
4. `medquad_ans_0000581-2` (score=0.6718) - Laron syndrome is a rare disorder. About 350 people have been diagnosed with the condition worldwide. The largest single group of affected individuals (about 100 people) lives in an area of southern Ecuador.
5. `medquad_ans_0003455-6` (score=0.6505) - How might Lafora disease be treated? Unfortunately, there is currently no cure for Lafora disease or way to slow the progression of the condition. Treatment is based on the signs and symptoms present in each person. F...

**Answer**:

The only specific treatment available for Laron syndrome is subcutaneous injections of insulin-like growth factor 1 (IGF-1), which stimulates linear growth, improves brain growth, and addresses metabolic abnormalities caused by long-term IGF-1 deficiency. IGF-1 has also been shown to raise blood glucose levels, reduce cholesterol, and increase muscle growth. Treatment involves closely monitoring IGF-1 and growth hormone (GH) levels to prevent health problems from overdosage [medquad_ans_0003473-6]. General treatment focuses on improving growth and may include IGF-1 injections [medquad_ans_0003473-1].

**Supported citations**: `medquad_ans_0003473-6`, `medquad_ans_0003473-1`
**Unsupported citations**: _none_

### Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Scores**: context_relevance=1.000, groundedness=0.905, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000016-15` (score=0.7048) - During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels of blood s...
2. `medquad_ans_0000016-4__dup2` (score=0.6905) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...
3. `medquad_ans_0000055-7` (score=0.6881) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need othe...
4. `medquad_ans_0000016-8` (score=0.6258) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a com...
5. `medquad_ans_0000016-19` (score=0.6137) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although bo...

**Answer**:

The treatments for diabetic retinopathy include:

- For the first three stages, no treatment is needed unless macular edema is present. To prevent progression, people with diabetes should control their blood sugar, blood pressure, and blood cholesterol levels [medquad_ans_0000016-15].

- For macular edema, prompt treatment with anti-VEGF drugs—either alone or in combination with laser treatment—results in better vision than laser treatment alone or steroid injections. These drugs reduce fluid leakage and inhibit the growth of new blood vessels in the retina. In some cases, focal laser treatment is used alongside injections, where up to several hundred small laser burns are placed around the macula to slow fluid leakage [medquad_ans_0000016-4__dup2].

- Proliferative retinopathy is treated with scatter laser treatment, which involves placing 1,000 to

**Supported citations**: `medquad_ans_0000016-15`, `medquad_ans_0000016-4__dup2`
**Unsupported citations**: _none_

### Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Scores**: context_relevance=1.000, groundedness=0.516, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0001421-3` (score=0.7547) - How might collagenous colitis be treated? Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases the condition may resolve on its own (spontaneous re...
2. `medquad_ans_0001421-1` (score=0.6603) - Collagenous colitis is a type of inflammatory bowel disease that affects the colon. It is a form of microscopic colitis, which means that the inflammation is only visible when a biopsy is examined under a microscope; ...
3. `medquad_ans_0000122-7` (score=0.6125) - A health care provider treats ulcerative colitis with - medications - surgery Which treatment a person needs depends on the severity of the disease and the symptoms. Each person experiences ulcerative colitis differen...
4. `medquad_ans_0000116-8` (score=0.6000) - To help reduce symptoms, a health care provider may recommend the following dietary changes: - avoid foods and drinks that contain caffeine or artificial sugars - drink plenty of liquids to prevent dehydration during ...
5. `medquad_ans_0000116-7` (score=0.5867) - Treatment depends on the severity of symptoms. The gastroenterologist will - review the medications the person is taking - make recommendations to change or stop certain medications - recommend that the person quit sm...

**Answer**:

Treatments for collagenous colitis vary depending on the symptoms and severity in each individual. In some cases, the condition may resolve on its own (spontaneous remission), although most people continue to experience ongoing or occasional diarrhea. Dietary changes are usually tried first to alleviate symptoms, including a reduced-fat diet, eliminating caffeine and lactose, and avoiding over-the-counter pain relievers such as ibuprofen or aspirin. If dietary changes are insufficient, medications may be used to control symptoms. Prescription anti-inflammatory medications such as mesalamine and sulfasalazine may help reduce swelling. Steroids like budesonide and prednisone can reduce inflammation but are typically reserved for sudden attacks of diarrhea due to the risk of unwanted side effects with long-term use. Anti-diarrheal medications such as bismuth subsalicylate, diphenoxylate with atropine, and loperamide can provide short-term relief.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Scores**: context_relevance=0.667, groundedness=0.861, answer_relevance=0.756

**Top docs**:

1. `medquad_ans_0000418-6` (score=0.7158) - Safe and effective prescription drugs are available to treat both Trichinella infection and the symptoms that occur as a result of infection. Treatment should begin as soon as possible; a doctor will make the decision...
2. `medquad_ans_0000418-1__dup2` (score=0.7033) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...
3. `medquad_ans_0000418-2__dup1` (score=0.6699) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
4. `medquad_ans_0000418-7` (score=0.6174) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause ...
5. `medquad_ans_0000418-5__dup1` (score=0.5909) - A diagnosis of trichinellosis is made in patients whose signs and symptoms are compatible with trichinellosis, have a positive laboratory test for Trichinella, and who can recall eating raw or undercooked pork or wild...

**Answer**:

To prevent Trichinellosis (also known as Trichinosis), wash your hands with warm water and soap after handling raw meat [medquad_ans_0000418-7]. Additionally, avoid consuming raw or undercooked meat, particularly from wild game such as bear or cougar, or from domestic pigs and wild boar, as these can harbor the Trichinella parasite [medquad_ans_0000418-1\_\_dup2][medquad_ans_0000418-2__dup1]. Curing (salting), drying, smoking, or microwaving meat alone may not be sufficient for prevention; proper cooking is essential to kill the parasite [medquad_ans_0000418-7]. Public awareness of the risks associated with eating raw or undercooked meat products also plays a key role in reducing the incidence of infection [medquad_ans

**Supported citations**: `medquad_ans_0000418-7`, `medquad_ans_0000418-1__dup2`, `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

### Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Scores**: context_relevance=0.750, groundedness=0.961, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000182-1__dup3` (score=0.8172) - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the normal components, or cells, of the liver and causes inflammation and liver damage. The immune system normally prot...
2. `medquad_ans_0000182-3__dup1` (score=0.7918) - A combination of autoimmunity, environmental triggers, and a genetic predisposition can lead to autoimmune hepatitis.
3. `medquad_ans_0000182-5__dup1` (score=0.7303) - Autoimmune hepatitis is classified into several types. Type 1 autoimmune hepatitis is the most common form in North America. Type 1 can occur at any age; however, it most often starts in adolescence or young adulthood...
4. `medquad_ans_0000182-10` (score=0.6877) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing autoimmune hepatitis.
5. `medquad_ans_0000182-11` (score=0.6869) - - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the liver and causes inflammation and damage. - Autoimmune hepatitis is a serious condition that may worsen over time...

**Answer**:

Autoimmune hepatitis is caused by a combination of autoimmunity, environmental triggers, and a genetic predisposition [medquad_ans_0000182-3__dup1].

**Supported citations**: `medquad_ans_0000182-3__dup1`
**Unsupported citations**: _none_

### Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Scores**: context_relevance=0.800, groundedness=0.948, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000006_3-1` (score=0.7281) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
2. `medquad_ans_0000006_3-4` (score=0.6305) - The signs and symptoms of astrocytomas are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain or spinal cord. - The size of the tumor. - How fast the tumor gr...
3. `medquad_ans_0000324-1__dup1` (score=0.5770) - Anaplastic astrocytoma is a rare, cancerous (malignant) type of brain tumor that arises from star-shaped brain cells called astrocytes. These cells surround and protect nerve cells in the brain and spinal cord. An ana...
4. `medquad_ans_0004883-4` (score=0.5621) - Are pilocytic astrocytomas inherited? Pilocytic astrocytomas are typically sporadic, occurring by chance in individuals with no history of the condition in the family. Sporadic abnormalities are not inherited from a p...
5. `medquad_ans_0004883-1` (score=0.5455) - Pilocytic astrocytoma is an often benign, slow-growing tumor of the brain or spinal cord. The tumor may be in the form of a cyst and usually does not spread to nearby tissues. Symptoms vary depending upon the size and...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They are tumors that start in star-shaped brain cells called astrocytes, which are a type of glial cell that holds nerve cells in place, brings food and oxygen to them, and helps protect them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

### Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Scores**: context_relevance=0.857, groundedness=0.919, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000013_3-2__dup3` (score=0.7489) - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising or bleeding and feeling tired and weak. These and other signs and symptoms may be caused by atypical CML or by other conditions. Check ...
2. `medquad_ans_0000013_3-1__dup3` (score=0.7184) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia in...
3. `medquad_ans_0000001_4-1` (score=0.6947) - Key Points - Chronic myelogenous leukemia is a disease in which the bone marrow makes too many white blood cells. - Leukemia may affect red blood cells, white blood cells, and platelets. - Signs and symptoms of chroni...
4. `medquad_ans_0000013_2-1__dup3` (score=0.6857) - Chronic myelogenous leukemia is a disease in which too many white blood cells are made in the bone marrow. See the PDQ summary on Chronic Myelogenous Leukemia Treatment for information on diagnosis, staging, and treat...
5. `medquad_ans_0000013_2-2__dup3` (score=0.6787) - See the PDQ summary about Chronic Myelogenous Leukemia Treatment for information.

**Answer**:

Atypical chronic myelogenous leukemia (CML) is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. In this condition, the body tells too many blood stem cells to become granulocytes, and some of these stem cells never become mature white blood cells; instead, they remain as immature cells called blasts. Over time, these granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. Atypical CML is similar to typical CML in that the leukemia cells look alike under a microscope, but it does not have the "Philadelphia chromosome" chromosome change that is typically present in typical CML [medquad_ans_0000013_3-1__dup3]. Signs and symptoms may include easy bruising or bleeding, feeling tired and weak, shortness of breath, pale skin, petechiae, and pain or a

**Supported citations**: `medquad_ans_0000013_3-1__dup3`
**Unsupported citations**: _none_

### Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Scores**: context_relevance=0.125, groundedness=0.000, answer_relevance=0.625

**Top docs**:

1. `medquad_ans_0000405-4` (score=0.4899) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000891-4` (score=0.4899) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0001090-4` (score=0.4899) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000741-4` (score=0.4899) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000801-4` (score=0.4899) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain any information about genetic changes related to trisomy 13. Therefore, the question cannot be answered based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Scores**: context_relevance=0.714, groundedness=0.955, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0004415-3` (score=0.8333) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respirato...
2. `medquad_ans_0000218-2__dup1` (score=0.8150) - There is currently no cure for Niemann-Pick disease. Treatment is supportive. Children usually die from infection or progressive neurological loss. There is currently no effective treatment for persons with type A. Bo...
3. `medquad_ans_0004415-1` (score=0.6995) - Niemann-Pick disease is an inherited condition involving lipid metabolism, which is the breakdown, transport, and use of fats and cholesterol in the body. In people with this condition, abnormal lipid metabolism cause...
4. `medquad_ans_0000218-1__dup1` (score=0.6676) - Niemann-Pick disease (NP) refers to a group of inherited metabolic disorders known as lipid storage diseases. Lipids (fatty materials such as waxes, fatty acids, oils, and cholesterol) and proteins are usually broken ...
5. `medquad_ans_0000731-1` (score=0.6573) - Niemann-Pick disease is a condition that affects many body systems. It has a wide range of symptoms that vary in severity. Niemann-Pick disease is divided into four main types: type A, type B, type C1, and type C2. Th...

**Answer**:

There is no specific treatment for Niemann-Pick disease type A. Treatment is supportive and may include care from specialists such as a pulmonologist for respiratory problems, a cardiologist for heart problems, liver and spleen specialists, nutritionists, physical therapists, and gastroenterologists. Learning specialists may also be involved. There is currently no effective treatment for persons with type A, and bone marrow transplantation has been attempted only in a few individuals with type B. Enzyme replacement and gene therapies might be helpful for type B, but these are not applicable to type A. Restricting one's diet does not prevent the buildup of lipids in cells and tissues [medquad_ans_0004415-3][medquad_ans_0000218-2__dup1][medquad_ans_0004415-1].

**Supported citations**: `medquad_ans_0004415-3`, `medquad_ans_0000218-2__dup1`, `medquad_ans_0004415-1`
**Unsupported citations**: _none_

### Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001714-1` (score=0.7732) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial poly...
2. `medquad_ans_0001708-1` (score=0.6520) - What are the signs and symptoms of Dandy-Walker like malformation with atrioventricular septal defect? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker like malformation ...
3. `medquad_ans_0001710-1` (score=0.6025) - What are the signs and symptoms of Dandy-Walker malformation with facial hemangioma? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with facial hemangioma....
4. `medquad_ans_0000270-5` (score=0.5829) - These resources address the diagnosis or management of Dandy-Walker malformation: - Genetic Testing Registry: Dandy-Walker syndrome - National Hydrocephalus Foundation: Treatment of Hydrocephalus These resources from ...
5. `medquad_ans_0003881-2` (score=0.5825) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speec...

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, autosomal recessive inheritance, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, and wide nasal bridge [

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.725, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003789-1` (score=0.7514) - Mantle cell lymphoma (MCL) belongs to a group of diseases known as non-Hodgkins lymphomas (NHL), which are cancers that affect the the lymphatic system (part of the immune system). MCL accounts for 6% of all non-Hodgk...
2. `medquad_ans_0003789-5` (score=0.6509) - How might Mantle cell lympoma be treated? Various treatmentsare currently available for Mantle cell lymphomas. Rare cases of early stage mantle cell lymphomas may be treated with radiation therapy. For more advance st...
3. `medquad_ans_0003789-4` (score=0.6262) - How is Mantle cell lymphoma diagnosed? Mantle cell lymphoma is diagnosed by a biopsy (surgical removal) of the lymph nodes. If lymph nodes are not easily accessible to be biopsied, a fine needle aspiration may be perf...
4. `medquad_ans_0003789-2` (score=0.6169) - What are the signs and symptoms of Mantle cell lymphoma? Common symptoms of Mantle cell lymphoma include fatigue, loss of appetite, and enlarged lymph nodes, spleen, and/or liver. Other symptoms may include night swea...
5. `medquad_ans_0003471-1` (score=0.5269) - Large granular lymphocyte (LGL) leukemia is a rare cancer of a type of white blood cells called lymphocytes. LGL leukemia causes a slow increase in white blood cells called T lymphocytes, or T cells, which originate i...

**Answer**:

The provided context does not specify the cause of Mantle cell lymphoma. [medquad_ans_0003789-1], [medquad_ans_0003789-5], [medquad_ans_0003789-4], and [medquad_ans_0003789-2] discuss diagnosis, treatment, symptoms, and classification but do not address the underlying cause of the disease. Therefore, based on the given information, the cause of Mantle cell lymphoma cannot be determined.

**Supported citations**: `medquad_ans_0003789-1`, `medquad_ans_0003789-5`, `medquad_ans_0003789-4`, `medquad_ans_0003789-2`
**Unsupported citations**: _none_

### Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.893, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000589-5` (score=0.7203) - These resources address the diagnosis or management of Legg-Calv-Perthes disease: - National Osteonecrosis Foundation - Seattle Children's Hospital These resources from MedlinePlus offer information about the diagnosi...
2. `medquad_ans_0000589-1` (score=0.7050) - Legg-Calv-Perthes disease is a bone disorder that affects the hips. Usually, only one hip is involved, but in about 10 percent of cases, both hips are affected. Legg-Calv-Perthes disease begins in childhood, typically...
3. `medquad_ans_0000589-3` (score=0.6925) - Legg-Calv-Perthes disease is usually not caused by genetic factors. The cause in these cases is unknown. In a small percentage of cases, mutations in the COL2A1 gene cause the bone abnormalities characteristic of Legg...
4. `medquad_ans_0006597-4` (score=0.5861) - Can 48,XXYY syndrome be inherited?
5. `medquad_ans_0003078-4` (score=0.5406) - How is hypophosphatemic rickets inherited? Hypophosphatemic rickets is most often inherited in an X-linked dominant manner. This means that the gene responsible for the condition is located on the X chromosome, and ha...

**Answer**:

Legg-Calv-Perthes disease is usually not caused by genetic factors, and the cause in most cases is unknown. However, in a small percentage of cases, mutations in the COL2A1 gene are associated with the bone abnormalities characteristic of the disease [medquad_ans_0000589-3]. Therefore, while it is not typically inherited, there can be a genetic component in rare instances. [medquad_ans_0000589-3]

**Supported citations**: `medquad_ans_0000589-3`
**Unsupported citations**: _none_

### Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Scores**: context_relevance=0.800, groundedness=0.526, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000647-2` (score=0.6840) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigra...
2. `medquad_ans_0000647-5` (score=0.6609) - These resources address the diagnosis or management of metachromatic leukodystrophy: - Gene Review: Gene Review: Arylsulfatase A Deficiency - Genetic Testing Registry: Metachromatic leukodystrophy - Genetic Testing Re...
3. `medquad_ans_0000647-1` (score=0.6586) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the sub...
4. `medquad_ans_0003566-1` (score=0.6408) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to def...
5. `medquad_ans_0000192-1__dup1` (score=0.6372) - Metachromatic leukodystrophy (MLD) is one of a group of genetic disorders called the leukodystrophies, which are characterized by the toxic buildup of lipids (fatty materials such as oils and waxes) and other storage ...

**Answer**:

Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells, particularly affecting cells in the nervous system that produce myelin, the substance that insulates and protects nerves. This accumulation leads to progressive destruction of white matter (leukodystrophy) throughout the nervous system, including the brain, spinal cord, and peripheral nerves. The damage results in progressive deterioration of intellectual functions and motor skills, such as the ability to walk, loss of sensation in the extremities (peripheral neuropathy), incontinence, seizures, paralysis, inability to speak, blindness, and hearing loss. Eventually, individuals may lose awareness of their surroundings and become unresponsive. While neurological problems are the primary feature, sulfatide accumulation can also affect other organs, most commonly the gallbladder. The most common form, affecting about 50 to 60 percent of individuals, is the late infantile form,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.927, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0005135-1` (score=0.6915) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
2. `medquad_ans_0001983-1` (score=0.6565) - What are the signs and symptoms of Dyssegmental dysplasia Silverman-Handmaker type? The Human Phenotype Ontology provides the following list of signs and symptoms for Dyssegmental dysplasia Silverman-Handmaker type. I...
3. `medquad_ans_0005835-1` (score=0.6452) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia with multiple dislocations? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia with ...
4. `medquad_ans_0000094-1__dup1` (score=0.6440) - What are the signs and symptoms of Acromesomelic dysplasia Maroteaux type? The Human Phenotype Ontology provides the following list of signs and symptoms for Acromesomelic dysplasia Maroteaux type. If the information ...
5. `medquad_ans_0000077-1__dup1` (score=0.6390) - What are the signs and symptoms of Acrofacial dysostosis Catania type? The Human Phenotype Ontology provides the following list of signs and symptoms for Acrofacial dysostosis Catania type. If the information is avail...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies (90%), elbow dislocation (90%), hypoplasia of the zygomatic bone (90%), scoliosis (90%), omphalocele (7.5%), hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, tongue-like lumbar vertebral deformities, and fever. These frequencies are based on the Human Phenotype Ontology, which compiles data from sources such as Orphanet [medquad_ans_0005135-1].

**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

### Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.940, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001664-3` (score=0.6278) - What causes cri du chat syndrome? Cri du chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected...
2. `medquad_ans_0000220-1__dup3` (score=0.4419) - Interstitial cystitis\*painful bladder syndrome (IC/PBS) is one of several conditions that causes bladder pain and a need to urinate frequently and urgently. Some doctors have started using the term bladder pain syndro...
3. `medquad_ans_0006007-1` (score=0.4090) - TAR syndrome is characterized by the absence of a bone called the radius in each forearm, short stature, and thrombocytopenia. The thrombocytopenia often appears first in infancy but becomes less severe or returns to ...
4. `medquad_ans_0004775-1` (score=0.3963) - Pearson syndrome is a mitochondrial disorder characterized by transfusion-dependent sideroblastic anemia and pancreatic dysfunction resulting in in malabsorption and chronic diarrhea. The features of this progressive ...
5. `medquad_ans_0004143-1` (score=0.3949) - Mowat-Wilson syndrome (MWS) is a rare genetic disorder that affects many systems of the body. The main features include moderate to severe intellectual disability, distinctive facial features, and epilepsy. Other feat...

**Answer**:

Cri-du-chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5, written as 5p-. The size of the deletion varies among affected individuals, with larger deletions generally leading to more severe intellectual disability and developmental delay than smaller deletions. The signs and symptoms are likely due to the loss of multiple genes on the short arm of chromosome 5. Researchers believe that the loss of a specific gene, CTNND2, is associated with severe intellectual disability in some individuals with this condition. [medquad_ans_0001664-3]

**Supported citations**: `medquad_ans_0001664-3`
**Unsupported citations**: _none_

### Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Scores**: context_relevance=1.000, groundedness=0.989, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000719-2__dup1` (score=0.7538) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmen...
2. `medquad_ans_0000718-1__dup1` (score=0.7019) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
3. `medquad_ans_0003881-2` (score=0.6477) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speec...
4. `medquad_ans_0000719-1__dup1` (score=0.6444) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides ...
5. `medquad_ans_0004960-2` (score=0.6129) - What are the symptoms of polymyositis? Polymyositis is characterized by chronic muscle inflammation and weakness involving the skeletal muscles (those involved with making movement) on both sides of the body. Weakness...

**Answer**:

The signs and symptoms of bilateral frontoparietal polymicrogyria may include moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze, ataxia, strabismus, increased muscle tone, finger dysmetria, ankle clonus, Babinski sign, broad-based gait, cerebellar hypoplasia, esotropia, exotropia, hyperreflexia, hypertonia, hypoplasia of the brainstem, nystagmus, polymicrogyria with an anterior to posterior gradient, and truncal ataxia [medquad_ans_0000719-2__dup1].

**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

### Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Scores**: context_relevance=0.167, groundedness=0.000, answer_relevance=0.367

**Top docs**:

1. `medquad_ans_0005818-1` (score=0.5483) - Spondylocarpotarsal synostosis (SCT) syndrome is an inherited syndrome characterized by disproportionate short stature, abnormalities of the vertebrae in the spine, scoliosis and lordosis, carpal and tarsal fusion (fu...
2. `medquad_ans_0006375-1` (score=0.5264) - Vohwinkel syndrome is an inherited condition that affects the skin. People with the "classic form" generally have honeycomb-patterned calluses on the palms of the hands and the soles of the feet (palmoplantar keratose...
3. `medquad_ans_0001143-1` (score=0.5065) - CHILD syndrome, also known as congenital hemidysplasia with ichthyosiform erythroderma and limb defects, is a genetic condition that is typically characterized by large patches of skin that are red and inflamed (eryth...
4. `medquad_ans_0000840-1__dup1` (score=0.4942) - Branchiootorenal syndrome is characterized by birth defects or anomalies of tissues in the neck, malformations of the external ear, hearing loss, and kidney malformations. Symptom and symptom severity can vary greatly...
5. `medquad_ans_0003384-1` (score=0.4937) - Keutel syndrome is an inherited condition characterized by cartilage calcification in the ears, nose, larnyx, trachea (voice box), and ribs; pulmonary artery stenoses; brachytelephalangism (short fingers and nails tha...

**Answer**:

The provided context does not mention "Konigsmark Knox Hussels syndrome" or any information related to this condition. Therefore, the cause cannot be determined from the given text.

Answer: Insufficient context [no_doc_id].

**Supported citations**: _none_
**Unsupported citations**: `no_doc_id`

### Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.990, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0004590-1` (score=0.7782) - What are the signs and symptoms of Orofaciodigital syndrome 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 3. If the information is available, the table ...
2. `medquad_ans_0004584-2` (score=0.7674) - What are the signs and symptoms of Orofaciodigital syndrome 1? Oral features of OFD1 may include a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormali...
3. `medquad_ans_0004593-1` (score=0.7589) - What are the signs and symptoms of Orofaciodigital syndrome 8? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 8. If the information is available, the table ...
4. `medquad_ans_0004589-2` (score=0.7158) - What are the signs and symptoms of Orofaciodigital syndrome 2? Although the signs and symptoms that occur in people with orofaciodigital syndrome type 2 may vary, the following findings may be present:Facial findings ...
5. `medquad_ans_0004586-1` (score=0.7082) - What are the signs and symptoms of Orofaciodigital syndrome 11? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 11. If the information is available, the tabl...

**Answer**:

Oral features of Orofaciodigital syndrome 1 may include a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormalities. Facial features may include hypertelorism (increased width between the eyes), a small nose, micrognathia (small jaw), and other features. The fingers and toes may be short (brachydactyly), webbed or joined together (syndactyly), abnormally curved (clinodactyly), or have other abnormalities. There may be brain abnormalities (such as cysts) and kidney problems (such as polycystic kidney disease) [medquad_ans_0004584-2].

**Supported citations**: `medquad_ans_0004584-2`
**Unsupported citations**: _none_

### Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Scores**: context_relevance=0.667, groundedness=0.894, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0000863-1__dup1` (score=0.5431) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For exa...
2. `medquad_ans_0000718-1__dup2` (score=0.5341) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts...
3. `medquad_ans_0000969-1__dup2` (score=0.5161) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
4. `medquad_ans_0000010-23` (score=0.5034) - Several types of surgery are available for someone with colorectal cancer. If the cancer is found at a very early stage, the doctor may remove it without cutting through the abdominal wall. Instead, the doctor may put...
5. `medquad_ans_0000029-18` (score=0.4977) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.

**Answer**:

Yes, there is information about surgery. Surgery can relieve or prevent pain, reduce symptoms, improve body function, or find a problem such as through a biopsy. Some surgeries, like heart surgery, can save a life. Laparoscopic surgery allows for smaller incisions using a thin tube with a camera and small tools. After surgery, there are risks such as infection, bleeding, reaction to anesthesia, or accidental injury, and pain is almost always present [medquad_ans_0000863-1__dup1]. Cosmetic surgery can reshape body parts, including breasts, ears, eyes, face, hair, nose, and tummy [medquad_ans_0000718-1__dup2]. Weight loss surgery helps people with extreme obesity, especially when diet and exercise are ineffective or when obesity causes serious health problems; it often limits food intake and affects digestion and nutrient absorption, with risks including infections, hernias

**Supported citations**: `medquad_ans_0000863-1__dup1`, `medquad_ans_0000718-1__dup2`
**Unsupported citations**: _none_

### Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Scores**: context_relevance=0.750, groundedness=0.985, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0002935-1` (score=0.7994) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affe...
2. `medquad_ans_0002935-4` (score=0.7503) - How is human T-cell leukemia virus, type 2 diagnosed? Human T-cell leukemia virus, type 2 (HTLV-2) is usually diagnosed based on blood tests that detect antibodies to the virus. However, HTLV-2 is often never suspecte...
3. `medquad_ans_0002935-5` (score=0.7084) - How might human T-cell leukemia virus, type 2 be treated? No cure or treatment exists for human T-cell leukemia virus, type 2 (HTLV-2). Management is focused on early detection and preventing the spread of HTLV-2 to o...
4. `medquad_ans_0002934-1` (score=0.6801) - Human T-cell leukemia virus, type 1 (HTLV-1) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, some affected people may later de...
5. `medquad_ans_0002934-4` (score=0.6655) - How is human T-cell leukemia virus, type 1 diagnosed? Human T-cell leukemia virus, type 1 (HTLV-1) is usually diagnosed based on blood tests that detect antibodies to the virus. However, HTLV-1 is often never suspecte...

**Answer**:

Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affects the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affected people may later develop neurological problems and/or chronic lung infections. HTLV-2 is spread by blood transfusions, sexual contact, and sharing needles. It can also be spread from mother to child during birth or breastfeeding. There is no cure or treatment for HTLV-2, and it is considered a lifelong condition; however, most infected people remain asymptomatic throughout life [medquad_ans_0002935-1].

**Supported citations**: `medquad_ans_0002935-1`
**Unsupported citations**: _none_

### Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Scores**: context_relevance=0.833, groundedness=0.977, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000522-1__dup1` (score=0.7612) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...
2. `medquad_ans_0000038-1__dup2` (score=0.7184) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chroni...
3. `medquad_ans_0000038-5__dup1` (score=0.6368) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
4. `medquad_ans_0000038-6` (score=0.5654) - Childhood interstitial lung disease (chILD) has many signs and symptoms because the disease has many forms. Signs and symptoms may include: Fast breathing, which also is called tachypnea (tak-ip-NE-ah) Labored breathi...
5. `medquad_ans_0000758-1__dup2` (score=0.5629) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...

**Answer**:

Interstitial lung diseases (ILD) are a large group of diseases that inflame or scar the lungs, making it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Causes include breathing in dust or other particles, such as coal dust (black lung disease), farm dust (farmer's lung), asbestos fibers (asbestosis), iron from mines or welding fumes (siderosis), and silica dust (silicosis). Other causes include autoimmune diseases or occupational exposures to molds, gases, or fumes. Some types have no known cause. Treatment depends on the type of exposure and disease stage and may involve medicines, oxygen therapy, or lung transplant in severe cases [medquad_ans_0000522-1__dup1]. In children, childhood interstitial lung disease (chILD) is a broad term for rare lung diseases affecting babies, children, and teens, with similar symptoms like chronic cough, rapid

**Supported citations**: `medquad_ans_0000522-1__dup1`
**Unsupported citations**: _none_

### Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0003743-4` (score=0.5293) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition w...
2. `medquad_ans_0002735-4` (score=0.5152) - Is Hashimoto's encephalitis inherited? We are aware of only one instance when more than one person in the same family had Hashimoto's encephalitis (HE). To our knowledge, no other cases of familial HE have been report...
3. `medquad_ans_0005695-4` (score=0.5080) - Is spastic diplegia cerebral palsy inherited? Scientists have found that family members of people with cerebral palsy, including spastic diplegia cerebral palsy, have an increased risk of developing the condition. The...
4. `medquad_ans_0005004-4` (score=0.5065) - Is postural orthostatic tachycardia syndrome inherited? Most cases of postural orthostatic tachycardia syndrome (POTS) are not thought to be inherited. Although the condition generally occurs sporadically, some people...
5. `medquad_ans_0000261-4` (score=0.5049) - In most cases of cyclic vomiting syndrome, affected people have no known history of the disorder in their family. However, many affected individuals have a family history of related conditions, such as migraines, irri...

**Answer**:

The provided context does not contain information about whether Melnick-Needles syndrome is inherited. Therefore, based on the given sources, it is not possible to determine if Melnick-Needles syndrome is inherited.

Answer: Insufficient context [no_source].

**Supported citations**: _none_
**Unsupported citations**: `no_source`

### Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006140-1` (score=0.6693) - Tracheobronchopathia osteoplastica (TO) is a rare condition of the large airways. It is characterized by the presence of multiple growths (nodules) made of bone and cartilage tissue, in the submucosa of the tracheobro...
2. `medquad_ans_0006138-5` (score=0.6447) - How is tracheobronchomalacia diagnosed? A diagnosis of tracheobronchomalacia (TBM) may be suspected based on the presence of characteristic signs and symptoms or abnormal pulmonary function tests. Additional testing s...
3. `medquad_ans_0006138-1` (score=0.6106) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of ...
4. `medquad_ans_0006140-3` (score=0.5918) - What causes tracheobronchopathia osteoplastica? The underlying cause of tracheobronchopathia osteoplastica (TO) remains unknown. Several theories have been proposed, including chronic airway inflammation, exostosis (f...
5. `medquad_ans_0006138-3` (score=0.5772) - What causes tracheobronchomalacia? The underlying cause of tracheobronchomalacia (TBM) varies by subtype. Most cases of primary TBM (also called congenital TBM) are caused by genetic conditions that weaken the walls o...

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse [medquad_ans_0006138-1]. There are two forms: a congenital form (primary TBM), which typically develops during infancy or early childhood, and an acquired form (secondary TBM), which is usually seen in adults and often results from trauma, chronic inflammation, or prolonged compression of the airways [medquad_ans_0006138-1]. The condition is typically progressive and may lead to symptoms such as shortness of breath, cough, sputum retention, and wheezing or stridor with breathing [medquad_ans_0006138-1]. A diagnosis may be suspected based on symptoms or abnormal pulmonary function tests

**Supported citations**: `medquad_ans_0006138-1`
**Unsupported citations**: _none_

### Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Scores**: context_relevance=0.800, groundedness=0.993, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0006030-1` (score=0.7920) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people m...
2. `medquad_ans_0000880-1__dup2` (score=0.5171) - The temporomandibular joint (TMJ) connects your jaw to the side of your head. When it works well, it enables you to talk, chew, and yawn. For people with TMJ dysfunction, problems with the joint and muscles around it ...
3. `medquad_ans_0004514-1` (score=0.4633) - Oculopharyngeal muscular dystrophy (OPMD) is a genetic muscle disorder with onset during adulthood, most often between 40 and 60 years of age. This condition is characterized by slowly progressive muscle disease (myop...
4. `medquad_ans_0000389-1` (score=0.4617) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which al...
5. `medquad_ans_0000810-1` (score=0.4591) - Popliteal pterygium syndrome is a condition that affects the development of the face, skin, and genitals. Most people with this disorder are born with a cleft lip, a cleft palate (an opening in the roof of the mouth),...

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene, and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

### Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.054, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0003845-4` (score=0.6468) - Is McCune Albright syndrome inherited? McCune Albright syndrome (MAS) is not inherited. It is caused by a random change (mutation) in the GNAS gene that occurs very early in development. As a result, some of the body'...
2. `medquad_ans_0000631-4` (score=0.6034) - McCune-Albright syndrome is not inherited. Instead, it is caused by a random mutation in the GNAS gene that occurs very early in development. As a result, some of the body's cells have a normal version of the GNAS gen...
3. `medquad_ans_0004143-4` (score=0.6027) - How is Mowat-Wilson inherited? Mowat-Wilson syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of Mowat-Wilson...
4. `medquad_ans_0000615-4` (score=0.5819) - Maffucci syndrome is not inherited. The mutations that cause this disorder are somatic, which means they occur during a person's lifetime. A somatic mutation occurs in a single cell. As that cell continues to grow and...
5. `medquad_ans_0000181-5__dup1` (score=0.5769) - Alagille syndrome occurs in about one of every 30,000 live births.3 The disorder affects both sexes equally and shows no geographical, racial, or ethnic preferences. JAG1 and NOTCH2 gene mutations are inherited in an ...

**Answer**:

The provided context does not contain information about the inheritance of Kallmann syndrome. Therefore, it is not possible to determine based on the given text whether Kallmann syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Scores**: context_relevance=1.000, groundedness=0.895, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005967-2` (score=0.7260) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how...
2. `medquad_ans_0005970-1` (score=0.6527) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how...
3. `medquad_ans_0000908-1__dup1` (score=0.6459) - What are the signs and symptoms of Camptodactyly syndrome Guadalajara type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Camptodactyly syndrome Guadalajara type 3. If the inform...
4. `medquad_ans_0006385-1` (score=0.6418) - What are the signs and symptoms of Waardenburg syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Waardenburg syndrome type 3. If the information is available, the tabl...
5. `medquad_ans_0005022-1` (score=0.6267) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the tabl...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and autosomal dominant inheritance [medquad_ans_0005967-2].

**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

### Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Scores**: context_relevance=0.800, groundedness=0.442, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000197-1__dup3` (score=0.7058) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...
2. `medquad_ans_0000853-1__dup1` (score=0.6583) - Bronchiolitis obliterans is an inflammatory obstruction of the lung's tiniest airways, the bronchioles. The bronchioles may become damaged and inflamed after inhalation of toxic fumes, as a result of respiratory infec...
3. `medquad_ans_0000133-1__dup2` (score=0.6523) - The bronchi are two tubes that branch off the trachea, or windpipe. The bronchi carry air to your lungs. The most common problem with the bronchi is bronchitis, an inflammation of the tubes. Bronchitis can be acute or...
4. `medquad_ans_0000021-1__dup3` (score=0.6417) - Espaol Bronchitis (bron-KI-tis) is a condition in which the bronchial tubes become inflamed. These tubes carry air to your lungs. (For more information about the bronchial tubes and airways, go to the Diseases and Con...
5. `medquad_ans_0000008-1__dup3` (score=0.6080) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...

**Answer**:

Chronic bronchitis is an inflammation of the bronchial tubes, the airways that carry air to the lungs. It causes a cough that often brings up mucus and can also lead to shortness of breath, wheezing, a low fever, and chest tightness. It is one type of COPD (chronic obstructive pulmonary disease) and is characterized by the production of large amounts of mucus due to inflamed bronchial tubes, resulting in persistent coughing and difficulty breathing. Cigarette smoking is the most common cause, along with long-term exposure to air pollution, fumes, or dust. It is a long-term condition that tends to recur or never fully resolves. Diagnosis involves evaluating symptoms, listening to breathing, and possibly other tests. Treatment includes medicines to open airways and clear mucus, oxygen therapy, and pulmonary rehabilitation to help manage daily life [medquad_ans_0000197

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Scores**: context_relevance=0.875, groundedness=0.945, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000840-1__dup2` (score=0.6757) - Spinal muscular atrophy (SMA) is a genetic disease that attacks nerve cells, called motor neurons, in the spinal cord. These cells communicate with your voluntary muscles - the ones you can control, like in your arms ...
2. `medquad_ans_0000934-5` (score=0.6187) - These resources address the diagnosis or management of spinal and bulbar muscular atrophy: - Gene Review: Gene Review: Spinal and Bulbar Muscular Atrophy - Genetic Testing Registry: Bulbo-spinal atrophy X-linked - Med...
3. `medquad_ans_0000935-2` (score=0.6119) - Spinal muscular atrophy affects 1 in 6,000 to 1 in 10,000 people.
4. `medquad_ans_0006267-3` (score=0.5993) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
5. `medquad_ans_0000233-2__dup1` (score=0.5903) - Generally, treatment for the disorder begins with stretching exercises and massage. Anti-inflammatory drugs may be prescribed. Cessation of running, bicycling, or similar activities may be advised. A corticosteroid in...

**Answer**:

Treatments for Spinal Muscular Atrophy (SMA) include machines to help with breathing, nutritional support, physical therapy, and medicines. There is no cure, but these treatments help with symptoms and prevent complications [medquad_ans_0000840-1__dup2]. Genetic counseling is also important, especially if the disease runs in the family [medquad_ans_0000934-5].

**Supported citations**: `medquad_ans_0000840-1__dup2`, `medquad_ans_0000934-5`
**Unsupported citations**: _none_

### Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Scores**: context_relevance=0.889, groundedness=0.893, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000040-19` (score=0.6850) - Standard treatments for leukemia include chemotherapy, biological therapy, radiation therapy, and immunotherapy. Some patients receive a combination of treatments. Learn more about treatments for acute myeloid leukemi...
2. `medquad_ans_0000001_6-7` (score=0.6593) - Key Points - There are different types of treatment for childhood acute lymphoblastic leukemia (ALL). - Children with ALL should have their treatment planned by a team of doctors who are experts in treating childhood ...
3. `medquad_ans_0000001_1-7` (score=0.6294) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - C...
4. `medquad_ans_0000040-4__dup2` (score=0.6230) - There are many treatment options for people with leukemia. The choice of treatment depends on your age and general health, the type of leukemia you have, whether or not it has spread outside the bone marrow, and other...
5. `medquad_ans_0000001_1-1` (score=0.6150) - Key Points - Adult acute lymphoblastic leukemia (ALL) is a type of cancer in which the bone marrow makes too many lymphocytes (a type of white blood cell). - Leukemia may affect red blood cells, white blood cells, and...

**Answer**:

The provided context does not contain information about treatments for Adult Acute Lymphoblastic Leukemia (ALL). While the context details treatments for childhood acute lymphoblastic leukemia, including chemotherapy, radiation therapy, chemotherapy with stem cell transplant, targeted therapy, and clinical trials, there is no mention of adult ALL treatments. [medquad_ans_0000001_6-7]

**Supported citations**: `medquad_ans_0000001_6-7`
**Unsupported citations**: _none_

### Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Scores**: context_relevance=0.750, groundedness=0.026, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000388-2` (score=0.7626) - Friedreich ataxia is estimated to affect 1 in 40,000 people. This condition is found in people with European, Middle Eastern, or North African ancestry. It is rarely identified in other ethnic groups.
2. `medquad_ans_0000382-1__dup2` (score=0.7595) - Friedreich's ataxia is an inherited disease that damages your nervous system. The damage affects your spinal cord and the nerves that control muscle movement in your arms and legs. Symptoms usually begin between the a...
3. `medquad_ans_0000388-1` (score=0.7543) - Friedreich ataxia is a genetic condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features of...
4. `medquad_ans_0000133-1__dup1` (score=0.7459) - Friedreich's ataxia is a rare inherited disease that causes progressive damage to the nervous system and movement problems. Neurological symptoms include awkward, unsteady movements, impaired sensory function, speech ...
5. `medquad_ans_0000388-5` (score=0.7062) - These resources address the diagnosis or management of Friedreich ataxia: - Friedreich's Ataxia Research Alliance: Clinical Care Guidelines - Gene Review: Gene Review: Friedreich Ataxia - Genetic Testing Registry: Fri...

**Answer**:

The provided context does not specify the cause of Friedreich ataxia. While it describes the symptoms, inheritance, and affected systems, it does not explain the underlying genetic or biological cause. Therefore, based on the given information, the cause cannot be determined. [doc_id] is not available.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Scores**: context_relevance=0.833, groundedness=0.776, answer_relevance=0.633

**Top docs**:

1. `medquad_ans_0000013-4__dup2` (score=0.5793) - Atrial fibrillation (AF) occurs if the heart's electrical signals don't travel through the heart in a normal way. Instead, they become very rapid and disorganized. Damage to the heart's electrical system causes AF. Th...
2. `medquad_ans_0000013-7` (score=0.5644) - Atrial fibrillation (AF) is diagnosed based on your medical and family histories, a physical exam, and the results from tests and procedures. Sometimes AF doesn't cause signs or symptoms. Thus, it may be found during ...
3. `medquad_ans_0000070-1__dup4` (score=0.5575) - An arrhythmia is a problem with the speed or rhythm of the heartbeat. Atrial fibrillation (AF) is the most common type of arrhythmia. The cause is a disorder in the heart's electrical system. Often, people who have AF...
4. `medquad_ans_0000013-8` (score=0.5400) - Treatment for atrial fibrillation (AF) depends on how often you have symptoms, how severe they are, and whether you already have heart disease. General treatment options include medicines, medical procedures, and life...
5. `medquad_ans_0000013-6` (score=0.5246) - Atrial fibrillation (AF) usually causes the heart's lower chambers, the ventricles, to contract faster than normal. When this happens, the ventricles can't completely fill with blood. Thus, they may not be able to pum...

**Answer**:

The provided context does not define or describe familial atrial fibrillation. While it mentions family medical history as part of the diagnostic process for atrial fibrillation [medquad_ans_0000013-7], it does not specify whether familial atrial fibrillation exists as a condition or provide details about its causes, inheritance, or prevalence. Therefore, based on the given information, familial atrial fibrillation cannot be answered.

Answer: Insufficient context to determine what familial atrial fibrillation is.

**Supported citations**: `medquad_ans_0000013-7`
**Unsupported citations**: _none_

### Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.8227) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the assoc...
2. `medquad_ans_0004627-2` (score=0.6388) - What are the signs and symptoms of Osteopathia striata cranial sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopathia striata cranial sclerosis. If the information is...
3. `medquad_ans_0004625-1` (score=0.5656) - Osteomesopyknosis is a bone disorder characterized by abnormal hardening of bone (osteosclerosis). It is generally limited to the axial spine, pelvis, and proximal part of the long bones, which is what distinguishes t...
4. `medquad_ans_0000260-1__dup1` (score=0.5615) - Striatonigral degeneration is a neurological disorder caused by a disruption in the connection between two areas of the brain-the striatum and the substantia nigra. These two areas work together to enable balance and ...
5. `medquad_ans_0000238-1__dup1` (score=0.5423) - Primary lateral sclerosis (PLS) is a rare neuromuscular disease with slowly progressive weakness in voluntary muscle movement. PLS belongs to a group of disorders known as motor neuron diseases. PLS affects the upper ...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary significantly from person to person, even within a single family. Features of the condition are generally present at birth and may include skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected people may also have developmental delay, hearing loss, heart defects, and/or ophthalmoplegia (paralysis of the muscles surrounding the eyes). Osteopathia striata cranial sclerosis is caused by changes (mutations) in the WTX gene and is inherited in an X-linked dominant manner. Treatment is based on the signs and symptoms present in each person [medquad_ans_0004

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0004097-5` (score=0.6926) - How is mixed connective tissue disease diagnosed? Mixed connective tissue disease (MCTD) is often suspected after a physical examination reveals signs and symptoms associated with the condition. The diagnosis is suppo...
2. `medquad_ans_0004097-2` (score=0.6884) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus eryt...
3. `medquad_ans_0004097-1` (score=0.6248) - Mixed connective tissue disease (MCTD) is a rare autoimmune disorder that is characterized by features commonly seen in three different connective tissue disorders: systemic lupus erythematosus, scleroderma, and polym...
4. `medquad_ans_0000230-1__dup2` (score=0.6152) - Connective tissue is the material inside your body that supports many of its parts. It is the "cellular glue" that gives your tissues their shape and helps keep them strong. It also helps some of your tissues do their...
5. `medquad_ans_0004172-2` (score=0.5516) - What are the signs and symptoms of multicentric Castleman disease? The signs and symptoms of multicentric Castleman disease (MCD) are often nonspecific and blamed on other, more common conditions. They can vary but ma...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. A condition called Raynaud's phenomenon sometimes occurs months or years before other symptoms of MCTD develop. Most people with MCTD have pain in multiple joints, and/or inflammation of joints (arthritis). Muscle weakness, fevers, and fatigue are also common. Other signs and symptoms may include: accumulation of fluid in the tissue of the hands that causes puffiness and swelling (edema), skin findings including lupus-like rashes (including reddish brown patches), reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia), dysfunction of the esophagus

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Scores**: context_relevance=1.000, groundedness=0.879, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000004-5__dup2` (score=0.6768) - Medications Can Treat Symptoms There is no known cure for Alzheimer's disease, but there are medicines that can treat symptoms of the disease. Most Alzheimers medicines work best for people in the mild or moderate sta...
2. `medquad_ans_0000024-1__dup2` (score=0.6639) - Caring for someone who has Alzheimer's disease (AD) can be stressful and overwhelming. It's important to take care of yourself. Ask for and accept help. Talk to the doctor. Find out what treatments might help control ...
3. `medquad_ans_0000003-6__dup1` (score=0.6632) - Most people with Alzheimers disease are cared for at home by family members. Within families, caregiving is provided most often by wives and husbands, followed by daughters. As Alzheimers disease gets worse, the perso...
4. `medquad_ans_0000098-2__dup1` (score=0.6575) - Drugs to specifically treat Alzheimers disease and some other progressive dementias are now available. Although these drugs do not halt the disease or reverse existing brain damage, they can improve symptoms and slow ...
5. `medquad_ans_0000003-30` (score=0.5910) - As Alzheimers disease gets worse, you will need more help to care for the person. It's okay to seek help whenever you need it. Several kinds of help are available. - Home health care agencies send a home health aide o...

**Answer**:

The provided context does not contain specific information about treatments for Alzheimer's caregiving. It discusses medications for treating symptoms, memory aids, and managing behavioral symptoms, but does not address treatments or strategies specifically for caregivers. Therefore, based on the given context, it is not possible to provide a definitive answer to the question about treatments for Alzheimer's caregiving. [medquad_ans_0000004-5__dup2]

**Supported citations**: `medquad_ans_0000004-5__dup2`
**Unsupported citations**: _none_

### Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.740

**Top docs**:

1. `medquad_ans_0000293-4` (score=0.6302) - Distal myopathy 2 is inherited in an autosomal dominant pattern, which means one copy of the altered MATR3 gene in each cell is sufficient to cause the disorder.
2. `medquad_ans_0003179-1` (score=0.6002) - Inclusion body myopathy 2, also known as hereditary inclusion body myopathy (HIBM), GNE-related myopathy, distal myopathy with rimmed vacuoles, and Nonaka myopathy, is an inherited condition that primarily affects the...
3. `medquad_ans_0000707-1__dup1` (score=0.5755) - Bethlem myopathy is an inherited movement disorder characterized by progressive muscle weakness and joint stiffness (contractures) in the fingers, wrists, elbows, and ankles. Due to a progressive course, up to two-thi...
4. `medquad_ans_0006236-4` (score=0.5731) - Is tubular aggregate myopathy genetic? It is evident from family history studies that the condition can be passed through families in either an autosomal dominant or autosomal recessive fashion. Some cases appear to b...
5. `medquad_ans_0006500-1` (score=0.5640) - X-linked scapuloperoneal myopathy is an inherited muscular dystrophy characterized by weakness and wasting of the muscles in the lower legs and the area of the shoulder blades. In some individuals, facial muscles may ...

**Answer**:

The provided context does not contain any information about the genetic changes related to idiopathic inflammatory myopathy. Therefore, a definitive answer cannot be provided based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.845, answer_relevance=0.567

**Top docs**:

1. `medquad_ans_0000855-2` (score=0.8100) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
2. `medquad_ans_0000855-5` (score=0.6950) - These resources address the diagnosis or management of pyruvate dehydrogenase deficiency: - Genetic Testing Registry: Pyruvate dehydrogenase E1-beta deficiency - Genetic Testing Registry: Pyruvate dehydrogenase E2 def...
3. `medquad_ans_0000855-1` (score=0.6831) - Pyruvate dehydrogenase deficiency is characterized by the buildup of a chemical called lactic acid in the body and a variety of neurological problems. Signs and symptoms of this condition usually first appear shortly ...
4. `medquad_ans_0005223-2` (score=0.6812) - What are the signs and symptoms of Pyruvate dehydrogenase deficiency? Pyruvate dehydrogenase (PDH) deficiency can have a significant effect on fetal development, which may become apparent during late pregnancy with po...
5. `medquad_ans_0005223-5` (score=0.6809) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for...

**Answer**:

Diagnosis of pyruvate dehydrogenase deficiency can be approached through genetic testing, which identifies specific deficiencies in components of the pyruvate dehydrogenase complex, such as E1-beta, E2, E3-binding protein, or phosphatase deficiencies [medquad_ans_0000855-5]. Additionally, diagnostic resources such as those provided by MedlinePlus, including Diagnostic Tests, Genetic Counseling, and Drug Therapy, may support the evaluation and management of the condition [medquad_ans_0000855-5]. Signs and symptoms, including lactic acidosis, neurological issues, and developmental delays, may also prompt further diagnostic evaluation [medquad_ans_0000855-1]. However, specific diagnostic procedures or tests are not detailed in the provided context.

**Supported citations**: `medquad_ans_0000855-5`, `medquad_ans_0000855-1`
**Unsupported citations**: _none_

### Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006562-2` (score=0.7983) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb a...
2. `medquad_ans_0006559-2` (score=0.7302) - What are the signs and symptoms of 16q24.3 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 16q24.3 microdeletion syndrome. If the information is available, th...
3. `medquad_ans_0006554-2` (score=0.7301) - What are the signs and symptoms of 15q11.2 microdeletion? The signs and symptoms in people with a 15q11.2 microdeletion can vary widely. Some people with the microdeletion don't have any apparent features, while other...
4. `medquad_ans_0006565-2` (score=0.6987) - What are the signs and symptoms of 1q21.1 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 1q21.1 microdeletion syndrome. If the information is available, the ...
5. `medquad_ans_0006566-1` (score=0.6830) - What are the signs and symptoms of 1q44 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 1q44 microdeletion syndrome. If the information is available, the tabl...

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

**Scores**: context_relevance=1.000, groundedness=0.923, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003860-1` (score=0.8542) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...
2. `medquad_ans_0003860-3` (score=0.8426) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
3. `medquad_ans_0000635-3` (score=0.8289) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
4. `medquad_ans_0000635-1` (score=0.8121) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infan...
5. `medquad_ans_0000635-2` (score=0.7997) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of...

**Answer**:

MECP2 duplication syndrome is caused by an extra copy (duplication) of the MECP2 gene in each cell, which is generally due to a duplication of genetic material located on the long (q) arm of the X chromosome [medquad_ans_0003860-1]. The duplication can vary in size, ranging from 100,000 to 900,000 base pairs (100 to 900 kilobases), and the MECP2 gene is always included in this duplication [medquad_ans_0000635-3]. This results in the production of excess MeCP2 protein, which disrupts the normal regulation of other genes in the brain, leading to irregular brain activity and the associated signs and symptoms [medquad_ans_0003860-3].

**Supported citations**: `medquad_ans_0003860-1`, `medquad_ans_0000635-3`, `medquad_ans_0003860-3`
**Unsupported citations**: _none_

### Example 187 (`0004633-2`)

**Question**: What are the symptoms of Osteopetrosis autosomal dominant type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004633-2` (score=0.8057) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 2. If the informatio...
2. `medquad_ans_0004637-2` (score=0.7341) - What are the signs and symptoms of Osteopetrosis autosomal recessive 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 4. If the information is ava...
3. `medquad_ans_0004632-2` (score=0.7298) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 1. If the informatio...
4. `medquad_ans_0004640-2` (score=0.7123) - What are the signs and symptoms of Osteopetrosis autosomal recessive 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 7. If the information is ava...
5. `medquad_ans_0000759-1` (score=0.7020) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of ...

**Answer**:

The symptoms of Osteopetrosis autosomal dominant type 2 include abnormality of epiphysis morphology, abnormality of the metacarpal bones, abnormality of the metaphyses, aseptic necrosis, bone pain, facial palsy, frontal bossing, joint dislocation, macrocephaly, osteoarthritis, osteomyelitis, recurrent fractures, short distal phalanx of finger, anemia, genu valgum, optic atrophy, short stature, visual impairment, abnormality of leukocytes, carious teeth, hearing impairment, hydrocephalus, hypocalcemia, bone marrow hypocellularity, abnormality of pelvic girdle bone morphology, abnormality of the vertebral endplates, generalized osteosclerosis, hip osteoarthritis, juvenile onset, mandibular osteomyelitis, and fractures of the long bones [medquad_ans_0004633

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 188 (`0003459-4`)

**Question**: What are the treatments for Lambert Eaton myasthenic syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.309, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0003459-4` (score=0.7084) - How might Lambert-Eaton myasthenic syndrome be treated? Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, ...
2. `medquad_ans_0006133-6` (score=0.6179) - Is there treatment for Townes-Brocks syndrome? Treatment is directed towards the specific symptoms, including immediate surgical intervention for imperforate anus; surgery for severe malformations of the hands; routin...
3. `medquad_ans_0003459-1` (score=0.6139) - Lambert Eaton myasthenic syndrome (LEMS) is a disorder of the neuromuscular junction. The neuromuscular junction is the site where nerve cells meet muscle cells and help activate the muscles. This syndrome occurs when...
4. `medquad_ans_0003459-2` (score=0.5952) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficu...
5. `medquad_ans_0000200-2__dup2` (score=0.5846) - Currently there is no cure for these disease syndromes.Medical care is directed at treating systemic conditions and improving the person's quality of life. Physical therapy and daily exercise may delay joint problems ...

**Answer**:

Treatments for Lambert-Eaton myasthenic syndrome (LEMS) may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, plasmapheresis, IV immunoglobulins, steroids (e.g., prednisone), azathioprine or cyclosporine, and/or 3,4-diaminopyridine. While some evidence suggests that 3,4-diaminopyridine or IV immunoglobulin can improve muscle strength and nerve to muscle cell communication, the degree of benefit still needs to be determined. Treatment of the underlying disorder or cancer is the first priority [medquad_ans_0003459-4, medquad_ans_0003459-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0003459-4, medquad_ans_0003459-1`

### Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.923, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000445-1__dup1` (score=0.7404) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...
2. `medquad_ans_0000445-2__dup1` (score=0.7150) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes...
3. `medquad_ans_0000065-2` (score=0.6736) - Arginase deficiency is a very rare disorder; it has been estimated to occur once in every 300,000 to 1,000,000 individuals.
4. `medquad_ans_0000065-1` (score=0.6658) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down ...
5. `medquad_ans_0003472-2` (score=0.5984) - What are the signs and symptoms of L-arginine:glycine amidinotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for L-arginine:glycine amidinotransferase deficiency....

**Answer**:

The symptoms of Arginase deficiency include growth deficiency, spasticity (abnormal tensing of the muscles), developmental delay, loss of developmental milestones, intellectual disability, seizures, microcephaly, hyperammonemia, aminoaciduria, behavioral abnormality, cognitive impairment, neurological speech impairment, EEG abnormality, hemiplegia/hemiparesis, hyperactivity, irritability, oroticaciduria, postnatal growth retardation, progressive spastic quadriplegia, and vomiting [medquad_ans_0000445-1__dup1]. Additionally, approximately 90% of patients exhibit aminoaciduria, behavioral abnormality, cognitive impairment, and neurological speech impairment, while 50% show EEG abnormality, hyperammonemia, seizures, hemiplegia/hemiparesis, and other listed symptoms [medquad_ans_0000445-2\_\_dup

**Supported citations**: `medquad_ans_0000445-1__dup1`
**Unsupported citations**: _none_

### Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Scores**: context_relevance=0.778, groundedness=0.947, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0006376-6` (score=0.8656) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but ar...
2. `medquad_ans_0000961-1__dup1` (score=0.8018) - Von Hippel-Lindau disease (VHL) is a rare, genetic disease that causes tumors and cysts to grow in your body. The tumors can be either cancerous or benign. They can grow in your brain and spinal cord, kidneys, pancrea...
3. `medquad_ans_0006376-1` (score=0.7523) - Von Hippel-Lindau (VHL) disease is an inherited disorder characterized by the abnormal growth of both benign and cancerous tumors and cysts in many parts of the body. Tumors usually first appear in young adulthood. Th...
4. `medquad_ans_0006376-5` (score=0.7416) - How is von Hippel-Lindau (VHL) disease diagnosed? The diagnosis of von Hippel-Lindau (VHL) disease can be made based on specific clinical criteria (signs and symptoms), or when molecular genetic testing reveals a muta...
5. `medquad_ans_0000022-1__dup1` (score=0.7308) - von Hippel-Lindau disease (VHL) is a rare, genetic multi-system disorder in which non-cancerous tumors grow in certain parts of the body. Slow-growing hemgioblastomas -- benign tumors with many blood vessels -- may de...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. The goal is to treat growths when they cause symptoms but are still small to prevent permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All people with VHL disease should be carefully followed by a physician or medical team familiar with the disorder [medquad_ans_0006376-6]. Surgery or radiation therapy are commonly used, with the aim of addressing tumors early [medquad_ans_0000961-1__dup1].

**Supported citations**: `medquad_ans_0006376-6`, `medquad_ans_0000961-1__dup1`
**Unsupported citations**: _none_

### Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Scores**: context_relevance=0.800, groundedness=0.052, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000738-2` (score=0.7072) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
2. `medquad_ans_0000738-1` (score=0.6686) - Noonan syndrome is a condition that affects many areas of the body. It is characterized by mildly unusual facial characteristics, short stature, heart defects, bleeding problems, skeletal malformations, and many other...
3. `medquad_ans_0004454-3` (score=0.6213) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004451-3` (score=0.6213) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004450-4` (score=0.6213) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not mention "Noonan syndrome 3" or any such classification. Therefore, based on the available information, it is not possible to answer the question. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005865-2` (score=0.7346) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may...
2. `medquad_ans_0005865-1` (score=0.6182) - Sprengel deformity is a congenital condition characterized by abnormal development and elevation of the shoulder blade (scapula). Severity can range considerably from being almost invisible when covered with clothes, ...
3. `medquad_ans_0005865-3` (score=0.5875) - How might Sprengel deformity be treated? Treatment of Sprengel deformity depends on the degree of functional impairment and/or cosmetic disfigurement. Many people with Sprengel deformity do not need surgery and may ha...
4. `medquad_ans_0001315-2` (score=0.5451) - What are the signs and symptoms of Chronic atypical neutrophilic dermatosis with lipodystrophy and elevated temperature? The Human Phenotype Ontology provides the following list of signs and symptoms for Chronic atypi...
5. `medquad_ans_0000003-2__dup3` (score=0.5443) - Signs and symptoms of Cushing's syndrome vary, but most people with the disorder have upper body obesity, a rounded face, increased fat around the neck, and relatively slender arms and legs. Children tend to be obese ...

**Answer**:

The symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additional features that may be present include abnormality of the shoulder (seen in approximately 90% of patients), cleft palate (in about 7.5% of patients), scoliosis, hemivertebrae, neck muscle hypoplasia, rib segmentation abnormalities, cervical segmentation defects, spina bifida occulta, and abnormalities of the collarbone or chest organs [medquad_ans_0005865-2]. The deformity more commonly occurs on the left side but can be bilateral [medquad_ans_0005865-2].

**Supported citations**: `medquad_ans_0005865-2`
**Unsupported citations**: _none_

### Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.908, answer_relevance=0.844

**Top docs**:

1. `medquad_ans_0000882-4` (score=0.7837) - As discussed above, Robinow syndrome can have either an autosomal recessive or an autosomal dominant pattern of inheritance. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. T...
2. `medquad_ans_0000882-5` (score=0.6972) - These resources address the diagnosis or management of Robinow syndrome: - Gene Review: Gene Review: Autosomal Dominant Robinow Syndrome - Gene Review: Gene Review: ROR2-Related Robinow Syndrome - Genetic Testing Regi...
3. `medquad_ans_0005415-4` (score=0.6887) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...
4. `medquad_ans_0005415-1` (score=0.6846) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
5. `medquad_ans_0000882-2` (score=0.6737) - Both the autosomal recessive and autosomal dominant forms of Robinow syndrome are rare. Fewer than 200 people with autosomal recessive Robinow syndrome have been described in the medical literature. This form of the c...

**Answer**:

Genetic changes related to Robinow syndrome include mutations in the genes that cause either autosomal recessive or autosomal dominant inheritance patterns. In autosomal recessive Robinow syndrome, both copies of the gene in each cell have mutations, while in autosomal dominant Robinow syndrome, one copy of the altered gene in each cell is sufficient to cause the disorder. Specific gene-related changes are associated with conditions such as ROR2-related Robinow syndrome [medquad_ans_0000882-4]. However, not all individuals with Robinow syndrome have identifiable mutations in known genes, and in some cases, the cause remains unknown [medquad_ans_0005415-4]. Genetic testing is available for both forms of the syndrome, and the Genetic Testing Registry provides information on these tests [medquad_ans_0005415-4].

**Supported citations**: `medquad_ans_0000882-4`, `medquad_ans_0005415-4`
**Unsupported citations**: _none_

### Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Scores**: context_relevance=0.909, groundedness=0.910, answer_relevance=0.836

**Top docs**:

1. `medquad_ans_0000180-1__dup3` (score=0.6676) - Kidney failure means your kidneys no longer work well enough to do their job. You need treatment to replace the work your damaged kidneys have stopped doing. The treatments for kidney failure are - hemodialysis - peri...
2. `medquad_ans_0000179-12` (score=0.6289) - Eating healthy foods can help you keep up your energy and strength. All dialysis and transplant centers have a dietitian. The dietitian helps people with kidney failure learn about healthy food choices. You should tal...
3. `medquad_ans_0000158-14` (score=0.6258) - - Kidney failure can affect a persons health in several ways. - When the kidneys stop working, waste products build up in the blood, a condition known as uremia. - People with kidney failure can avoid most of the prob...
4. `medquad_ans_0000180-8` (score=0.6254) - Eating the right foods can help you feel better when you are on dialysis or have a kidney transplant. Staying healthy with kidney failure requires watching how much of these elements are included in your diet: - Prote...
5. `medquad_ans_0000158-13` (score=0.6217) - For people who are on dialysis or approaching total kidney failure, adequate nutrition is important for maintaining energy, strength, healthy sleep patterns, bone health, heart health, and good mental health. A person...

**Answer**:

The provided context does not contain information about "Kidney Failure: Eat Right to Feel Right on Hemodialysis." While it mentions that a dietitian helps people with kidney failure learn about healthy food choices and that following a special diet can help keep wastes from building up in the blood, there is no specific content related to the topic "Eat Right to Feel Right on Hemodialysis." [medquad_ans_0000180-1__dup3]

**Supported citations**: `medquad_ans_0000180-1__dup3`
**Unsupported citations**: _none_

### Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Scores**: context_relevance=0.778, groundedness=0.945, answer_relevance=0.511

**Top docs**:

1. `medquad_ans_0000677-2` (score=0.8220) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
2. `medquad_ans_0000675-2` (score=0.7473) - Mucolipidosis III alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
3. `medquad_ans_0000674-2` (score=0.7451) - Mucolipidosis II alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
4. `medquad_ans_0000676-2` (score=0.7198) - Mucolipidosis III gamma is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
5. `medquad_ans_0000677-1` (score=0.6917) - Mucolipidosis type IV is an inherited disorder characterized by delayed development and vision impairment that worsens over time. The severe form of the disorder is called typical mucolipidosis type IV, and the mild f...

**Answer**:

Mucolipidosis type IV is estimated to occur in 1 in 40,000 people [medquad_ans_0000677-2].

**Supported citations**: `medquad_ans_0000677-2`
**Unsupported citations**: _none_

### Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Scores**: context_relevance=0.750, groundedness=0.994, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.5793) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chi...
2. `medquad_ans_0000193-1__dup4` (score=0.4297) - Cirrhosis\* is scarring of the liver. Scar tissue forms because of injury or long-term disease. Scar tissue replaces healthy liver tissue and blocks the normal flow of blood through the liver. A healthy liver - makes p...
3. `medquad_ans_0005201-1` (score=0.3955) - Punctate palmoplantar keratoderma type I, also known as keratosis palmoplantaris papulosa (KPPP) or Brauer-Buschke-Fisher Syndrome is is a rare condition that affects the skin. It is a type of punctate palmoplantar ke...
4. `medquad_ans_0003715-1` (score=0.3901) - Lymphomatoid papulosis is a skin disorder that is characterized by crops of self healing skin lesions that look cancerous under the microscope but are actually benign (non-cancerous). Lesions contain unusual cells tha...
5. `medquad_ans_0000063-12` (score=0.3889) - Skin cancer occurs when cancer cells form in the tissues of the skin. The skin is mainly made up of two layers: the inner layer, called the dermis, and the outer layer, called the epidermis. Within the epidermis, ther...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams. [medquad_ans_0000795-1__dup1]

**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

### Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.786, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0001045-1` (score=0.7630) - X-linked adrenoleukodystrophy is a genetic disorder that occurs primarily in males. It mainly affects the nervous system and the adrenal glands, which are small glands located on top of each kidney. In this disorder, ...
2. `medquad_ans_0006488-4` (score=0.6826) - How is X-linked adrenal hypoplasia congenita inherited? X-linked adrenal hypoplasia congenita is inherited in an X-linked recessive pattern. A condition is considered X-linked if the mutated gene that causes the disor...
3. `medquad_ans_0006488-1` (score=0.6741) - X-linked adrenal hypoplasia congenita is an inherited disorder that mainly affects males. It involves many hormone-producing (endocrine) tissues in the body, particularly a pair of small glands on top of each kidney c...
4. `medquad_ans_0006502-4` (score=0.5995) - How is X-linked hypophosphatemia inherited? X-linked hypophosphatemia (XLH) is caused by mutations in the PHEX gene, and is inherited in an X-linked dominant manner. This means that the gene responsible for the condit...
5. `medquad_ans_0006553-1` (score=0.5727) - Congenital adrenal hyperplasia (CAH) due to 11-beta-hydroxylase deficiency is one of a group of disorders (collectively called congenital adrenal hyperplasia) that affect the adrenal glands. In this condition, the adr...

**Answer**:

The provided context does not specify the exact genetic changes related to X-linked adrenoleukodystrophy. It describes the disorder's effects on the nervous system and adrenal glands, as well as its types and symptoms, but does not detail the underlying genetic mutations or molecular mechanisms. [medquad_ans_0001045-1]

**Supported citations**: `medquad_ans_0001045-1`
**Unsupported citations**: _none_

### Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Scores**: context_relevance=1.000, groundedness=0.984, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000988-2` (score=0.7483) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of Europe...
2. `medquad_ans_0000988-1` (score=0.6117) - Transthyretin amyloidosis is a slowly progressive condition characterized by the buildup of abnormal deposits of a protein called amyloid (amyloidosis) in the body's organs and tissues. These protein deposits most fre...
3. `medquad_ans_0000195-2` (score=0.5684) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0001101-2` (score=0.5605) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
5. `medquad_ans_0000303-1__dup1` (score=0.5553) - What are the signs and symptoms of Amyloidosis familial visceral? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyloidosis familial visceral. If the information is available, the ...

**Answer**:

The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence is estimated to be one in 538 people. Among Americans of European descent, it affects one in 100,000 people. The cardiac form affects between 3 percent and 3.9 percent of African Americans and approximately 5 percent of people in some areas of West Africa [medquad_ans_0000988-2].

**Supported citations**: `medquad_ans_0000988-2`
**Unsupported citations**: _none_

### Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Scores**: context_relevance=1.000, groundedness=0.880, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.7249) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the informatio...
2. `medquad_ans_0001787-1` (score=0.5729) - What are the signs and symptoms of Dentin dysplasia, type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Dentin dysplasia, type 1. If the information is available, the table belo...
3. `medquad_ans_0005856-1` (score=0.5546) - What are the signs and symptoms of Spondylometaphyseal dysplasia with dentinogenesis imperfecta? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia with de...
4. `medquad_ans_0003176-1` (score=0.5503) - What are the signs and symptoms of Imperforate oropharynx-costo vetebral anomalies? The Human Phenotype Ontology provides the following list of signs and symptoms for Imperforate oropharynx-costo vetebral anomalies. I...
5. `medquad_ans_0003065-1` (score=0.5494) - What are the signs and symptoms of Hypomandibular faciocranial dysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypomandibular faciocranial dysostosis. If the information ...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include microdontia, taurodontia, dentinogenesis imperfecta limited to primary teeth, obliteration of the pulp chamber, and periapical radiolucency. These symptoms are reported in the Human Phenotype Ontology (HPO) database, which compiles information from sources such as Orphanet. The frequency of these symptoms is typically listed as a rough estimate of the percentage of patients who have the feature, or as a fraction indicating the number of patients with the symptom out of the total number examined in a study [medquad_ans_0001780-1]. However, specific symptom frequencies are not detailed in the provided context.

**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

### Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Scores**: context_relevance=0.750, groundedness=0.931, answer_relevance=0.350

**Top docs**:

1. `medquad_ans_0000782-2` (score=0.8056) - Periventricular heterotopia is a rare condition. Its incidence is unknown.
2. `medquad_ans_0004818-2` (score=0.6211) - What are the signs and symptoms of periventricular nodular heterotopia? The condition is first noticed when seizures appear, often during the teenage years. The nodules around the ventricles are then typically discove...
3. `medquad_ans_0000782-1` (score=0.5726) - Periventricular heterotopia is a condition in which nerve cells (neurons) do not migrate properly during the early development of the fetal brain, from about the 6th week to the 24th week of pregnancy. Heterotopia mea...
4. `medquad_ans_0004818-1` (score=0.5703) - Periventricular heterotopia is a condition in which the nerve cells (neurons) do not migrate properly during the early development of the fetal brain from about the 6th week to the 24th week of pregnancy. Affected peo...
5. `medquad_ans_0004818-3` (score=0.5677) - What are the recommended evaluations for patients diagnosed with periventricular nodular heterotopia? The following evaluations are recommended:[1823] Imaging exams of the brain to establish the diagnosis Evaluation b...

**Answer**:

The incidence of periventricular heterotopia is unknown [medquad_ans_0000782-2].

**Supported citations**: `medquad_ans_0000782-2`
**Unsupported citations**: _none_

### Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Scores**: context_relevance=0.636, groundedness=0.085, answer_relevance=0.709

**Top docs**:

1. `medquad_ans_0003462-3` (score=0.6891) - How is Landau-Kleffner syndrome (LKS) diagnosed? LKS is diagnosed based on clinical features and the results of an electroencephalogram (EEG), a recording of the electric activity of the brain. All LKS children have a...
2. `medquad_ans_0003462-1` (score=0.6325) - Landau-Kleffner syndrome (LKS) is a rare, childhood neurological disorder characterized by the sudden or gradual development of aphasia (the inability to understand or express language) and an abnormal electro-encepha...
3. `medquad_ans_0003462-2` (score=0.6263) - What are the signs and symptoms of Landau-Kleffner syndrome? Landau-Kleffner syndrome is characterized by the sudden or gradual development of aphasia (the inability to understand or express language) in previously no...
4. `medquad_ans_0000004-1__dup2` (score=0.6093) - Landau-Kleffner syndrome (LKS) is a rare, childhood neurological disorder characterized by the sudden or gradual development of aphasia (the inability to understand or express language) and an abnormal electro-encepha...
5. `medquad_ans_0006423-3` (score=0.5398) - How might Wernicke-Korsakoff syndrome be treated?

**Answer**:

The provided context does not mention any ongoing research or clinical trials for Landau-Kleffner Syndrome. Therefore, based on the given information, it is not possible to identify specific research or clinical trials. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004514-4` (score=0.7459) - How is oculopharyngeal muscular dystrophy inherited?
2. `medquad_ans_0003601-1` (score=0.7327) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
3. `medquad_ans_0000631-1__dup1` (score=0.7256) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...
4. `medquad_ans_0001537-1` (score=0.7122) - Congenital muscular dystrophy (CMD) refers to a group of inherited conditions that affect the muscles and are present at birth or in early infancy. The severity of the condition, the associated signs and symptoms and ...
5. `medquad_ans_0001935-1` (score=0.6988) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystroph...

**Answer**:

Muscular dystrophy (MD) is a group of more than 30 inherited diseases that cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood, while others may not appear until middle age or later. The different types can vary in whom they affect, which muscles they affect, and what the symptoms are. All forms of MD grow worse as the person's muscles get weaker, and most people with MD eventually lose the ability to walk. There is no cure for muscular dystrophy; treatments can help with symptoms and prevent complications, including physical and speech therapy, orthopedic devices, surgery, and medications [medquad_ans_0000631-1__dup1].

**Supported citations**: `medquad_ans_0000631-1__dup1`
**Unsupported citations**: _none_

### Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.350, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.5364) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that t...
2. `medquad_ans_0004956-1` (score=0.4265) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. Specifically, the surface of the brain develops too many folds which are unusually small. The signs and symptoms associate...
3. `medquad_ans_0005019-1` (score=0.3820) - What are the signs and symptoms of Preaxial polydactyly type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 1. If the information is available, the tabl...
4. `medquad_ans_0000754-5` (score=0.3810) - These resources address the diagnosis or management of oral-facial-digital syndrome: - Gene Review: Gene Review: Oral-Facial-Digital Syndrome Type I - Genetic Testing Registry: Mohr syndrome - Genetic Testing Registry...
5. `medquad_ans_0003881-2` (score=0.3775) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speec...

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding with no other physical anomalies or intellectual impairment, or it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits, and surgical management depends greatly on the complexity of the deformity [1] [2].

**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

### Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Scores**: context_relevance=1.000, groundedness=0.840, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002519-2` (score=0.6156) - What are the signs and symptoms of Geniospasm? The Human Phenotype Ontology provides the following list of signs and symptoms for Geniospasm. If the information is available, the table below includes how often the sym...
2. `medquad_ans_0004534-2` (score=0.5566) - What are the signs and symptoms of Omenn syndrome? Infants with Omenn syndrome typically present shortly after birth, usually by 3 months of age. This is similar to other types of severe combined immunodeficiency (SCI...
3. `medquad_ans_0002628-2` (score=0.5502) - What are the signs and symptoms of GM1 gangliosidosis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for GM1 gangliosidosis type 2. If the information is available, the table be...
4. `medquad_ans_0003786-1` (score=0.5394) - What are the signs and symptoms of Mannosidosis, beta A, lysosomal? The Human Phenotype Ontology provides the following list of signs and symptoms for Mannosidosis, beta A, lysosomal. If the information is available, ...
5. `medquad_ans_0002114-2` (score=0.5300) - What are the signs and symptoms of Eosinophilic fasciitis? The Human Phenotype Ontology provides the following list of signs and symptoms for Eosinophilic fasciitis. If the information is available, the table below in...

**Answer**:

The signs and symptoms of Geniospasm include chin myoclonus. This information is derived from the Human Phenotype Ontology, which documents the occurrence of symptoms in individuals with Geniospasm [medquad_ans_0002519-2].

**Supported citations**: `medquad_ans_0002519-2`
**Unsupported citations**: _none_

### Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=0.800, groundedness=0.806, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000078-1__dup2` (score=0.6041) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...
2. `medquad_ans_0000253-1__dup2` (score=0.5648) - Cystic fibrosis (CF) is an inherited disease of the mucus and sweat glands. It affects mostly your lungs, pancreas, liver, intestines, sinuses, and sex organs. CF causes your mucus to be thick and sticky. The mucus cl...
3. `medquad_ans_0000758-1__dup2` (score=0.5619) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
4. `medquad_ans_0000049-3__dup2` (score=0.5503) - A defect in the CFTR gene causes cystic fibrosis (CF). This gene makes a protein that controls the movement of salt and water in and out of your body's cells. In people who have CF, the gene makes a protein that doesn...
5. `medquad_ans_0001551-1` (score=0.5401) - Congenital pulmonary alveolar proteinosis is a rare form of respiratory failure that is present from birth. In this condition, a type of protein builds up in the air sacs (alveoli) of the lungs, making breathing diffi...

**Answer**:

The provided context does not specify the exact genetic changes related to idiopathic pulmonary fibrosis (IPF). It mentions that genetics may play a role in causing IPF and that familial IPF occurs when more than one member of a family has the disease [medquad_ans_0000078-1__dup2]. However, no details about specific genetic mutations or changes are given. Therefore, based on the available information, the genetic changes related to IPF cannot be determined.

**Supported citations**: `medquad_ans_0000078-1__dup2`
**Unsupported citations**: _none_

### Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.039, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0003593-4` (score=0.6335) - How is Liddle syndrome inherited? Liddle syndrome is inherited in an autosomal dominant manner. This means that only one mutated copy of the disease-causing gene in each cell is sufficient to cause the condition. The ...
2. `medquad_ans_0004750-4` (score=0.5721) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is pass...
3. `medquad_ans_0000673-4` (score=0.5702) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
4. `medquad_ans_0006446-4` (score=0.5558) - Is Williams syndrome inherited?
5. `medquad_ans_0004784-4` (score=0.5532) - How is Pelizaeus-Merzbacher disease inherited?

**Answer**:

The provided context does not contain information about the inheritance of Potocki-Shaffer syndrome. Therefore, it is not possible to determine based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Scores**: context_relevance=0.857, groundedness=0.935, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000177-1__dup2` (score=0.5631) - Summary : Children's healthy development depends on safe and positive experiences when they are very young. If you work or go to school, you want to know that your child is in good hands while you are away. You may ch...
2. `medquad_ans_0000189-1__dup2` (score=0.4538) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...
3. `medquad_ans_0000744-1__dup2` (score=0.4173) - Summary : Prenatal care is the health care you get while you are pregnant. It includes your checkups and prenatal testing. Prenatal care can help keep you and your baby healthy. It lets your health care provider spot ...
4. `medquad_ans_0000175-1__dup2` (score=0.4120) - Child abuse is doing something or failing to do something that results in harm to a child or puts a child at risk of harm. Child abuse can be physical, sexual or emotional. Neglect, or not providing for a child's need...
5. `medquad_ans_0000227-5` (score=0.4040) - These resources address the diagnosis or management of CHILD syndrome: - Gene Review: Gene Review: NSDHL-Related Disorders - Genetic Testing Registry: Child syndrome These resources from MedlinePlus offer information ...

**Answer**:

Yes, there is information about child care. Child care options include in-home care, where the caregiver comes to your home, or the child going to the caregiver's home, and also child care centers. It is important to get to know your child's caregivers, as they will be a big part of your child's life. The caregiver's training should include knowledge of how young children learn and grow, positive and consistent discipline, signs that a child is sick, cleanliness and safety practices to prevent illness or injury, and basic first aid [medquad_ans_0000177-1__dup2].

**Supported citations**: `medquad_ans_0000177-1__dup2`
**Unsupported citations**: _none_

### Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Scores**: context_relevance=0.800, groundedness=0.954, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000046-2` (score=0.8776) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
2. `medquad_ans_0000046-5` (score=0.7063) - These resources address the diagnosis or management of alternating hemiplegia of childhood: - The Great Ormond Street Hospital - University of Utah School of Medicine These resources from MedlinePlus offer information...
3. `medquad_ans_0000016-3__dup1` (score=0.7041) - Children with the benign form of alternating hemiplegia have a good prognosis. Those who experience the more severe form have a poor prognosis because intellectual and mental capacities do not respond to drug therapy,...
4. `medquad_ans_0000016-1__dup1` (score=0.6404) - Alternating hemiplegia is a rare neurological disorder that develops in childhood, most often before the child is 18 months old. The disorder is characterized by recurrent episodes of paralysis that involve one or bot...
5. `medquad_ans_0000046-1` (score=0.6309) - Alternating hemiplegia of childhood is a neurological condition characterized by recurrent episodes of temporary paralysis, often affecting one side of the body (hemiplegia). During some episodes, the paralysis altern...

**Answer**:

Approximately 1 in 1 million people are affected by alternating hemiplegia of childhood [medquad_ans_0000046-2].

**Supported citations**: `medquad_ans_0000046-2`
**Unsupported citations**: _none_

### Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003687-2` (score=0.7172) - What are the signs and symptoms of Lupus? You can read about the signs and symptoms of lupus from MedlinePlus and the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS). The Human Phenotype ...
2. `medquad_ans_0000159-3__dup2` (score=0.7148) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time...
3. `medquad_ans_0000186-1__dup1` (score=0.6530) - Lupus (also called systemic lupus erythematosus) is a disorder of the immune system. Normally, the immune system protects the body against invading infections and cancers. In lupus, the immune system is over-active an...
4. `medquad_ans_0000569-1__dup2` (score=0.6071) - If you have lupus, your immune system attacks healthy cells and tissues by mistake. This can damage your joints, skin, blood vessels and organs. There are many kinds of lupus. The most common type, systemic lupus eryt...
5. `medquad_ans_0003688-1` (score=0.5831) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, h...

**Answer**:

The symptoms of Lupus include abdominal pain, abnormality of temperature regulation, abnormality of the heart valves, abnormality of the pericardium, alopecia, arthralgia, arthritis, autoimmunity, chest pain, cutaneous photosensitivity, skin rash, thrombocytopenia, thrombophlebitis, abnormal pyramidal signs, abnormal tendon morphology, abnormality of the autonomic nervous system, abnormality of the endocardium, abnormality of the pleura, anorexia, arterial thrombosis, aseptic leukocyturia, bone marrow hypocellularity, conjunctival telangiectasia, cranial nerve paralysis, cutis marmorata, dry skin, eczema, edema of the lower limbs, glomerulopathy, hallucinations, hematuria, hepatomegaly, hyperkeratosis, hypoproteinemia, increased antibody level in blood

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.517, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000194-2` (score=0.7443) - Cockayne syndrome occurs in about 2 per million newborns in the United States and Europe.
2. `medquad_ans_0000194-1` (score=0.7367) - Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features of this disorder include a failure to gain weight and grow at the expected rate (failure to thrive), a...
3. `medquad_ans_0001398-1` (score=0.7021) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
4. `medquad_ans_0001395-1` (score=0.7021) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
5. `medquad_ans_0001397-1` (score=0.7021) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...

**Answer**:

Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging (progeria). It is marked by failure to gain weight and grow at the expected rate (failure to thrive), abnormally small head size (microcephaly), and impaired development of the nervous system. Affected individuals have extreme sensitivity to sunlight (photosensitivity), where even small amounts of sun exposure can cause sunburn. Other possible signs and symptoms include hearing loss, eye abnormalities, severe tooth decay, bone abnormalities, and changes in the brain visible on brain scans. The syndrome is divided into three subtypes: classical (type I) with onset in early childhood (usually after age 1 year), type II (congenital) with severe symptoms apparent at birth and often referred to as cerebro-oculo-facio-skeletal (COFS) syndrome or Pena-Shokeir syndrome type II, and type III, which has the mildest symptoms

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Scores**: context_relevance=0.833, groundedness=0.961, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000029-8__dup1` (score=0.7476) - Gingivitis is inflammation of the gums. In gingivitis, the gums become red, swollen and can bleed easily. Gingivitis is a mild form of gum disease. It can usually be reversed with daily brushing and flossing, and regu...
2. `medquad_ans_0000029-6__dup1` (score=0.7436) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead...
3. `medquad_ans_0000413-1__dup1` (score=0.7135) - If you have gum disease, you're not alone. Many U.S. adults currently have some form of the disease. It ranges from simple gum inflammation, called gingivitis, to serious damage to the tissue and bone supporting the t...
4. `medquad_ans_0000029-1__dup5` (score=0.7067) - An Infection of the Gums and Surrounding Tissues Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. The two forms of gum disease are gingivitis, a mild form that is...
5. `medquad_ans_0000029-12` (score=0.6974) - In some studies, researchers have observed that people with periodontal disease (when compared to people without periodontal disease) were more likely to develop heart disease or have difficulty controlling their bloo...

**Answer**:

Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. It ranges from a mild form called gingivitis, which is reversible with good oral hygiene such as daily brushing and flossing and regular cleanings by a dentist or dental hygienist, to a more severe form called periodontitis, which can damage the soft tissues and bone supporting the teeth. If left untreated, periodontitis can lead to tooth loss. In its early stages, gum disease is usually painless, and many people are not aware they have it; in more advanced stages, it can cause sore gums and pain when chewing [medquad_ans_0000029-1__dup5]. Gingivitis is characterized by red, swollen gums that bleed easily, and it does not involve loss of bone or tissue that hold teeth in place [medquad_ans_0000029

**Supported citations**: `medquad_ans_0000029-1__dup5`
**Unsupported citations**: _none_

### Example 212 (`0000494-2`)

**Question**: How many people are affected by hyperkalemic periodic paralysis ?

**Scores**: context_relevance=0.889, groundedness=0.939, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.9274) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0000494-5` (score=0.7453) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis ...
3. `medquad_ans_0000505-2` (score=0.7251) - Although its exact prevalence is unknown, hypokalemic periodic paralysis is estimated to affect 1 in 100,000 people. Men tend to experience symptoms of this condition more often than women.
4. `medquad_ans_0003000-2` (score=0.7247) - What are the signs and symptoms of Hyperkalemic periodic paralysis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hyperkalemic periodic paralysis. If the information is available, ...
5. `medquad_ans_0000494-1` (score=0.7004) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to...

**Answer**:

Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people [medquad_ans_0000494-2].

**Supported citations**: `medquad_ans_0000494-2`
**Unsupported citations**: _none_

### Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Scores**: context_relevance=0.800, groundedness=0.989, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0005212-1` (score=0.7220) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end o...
2. `medquad_ans_0004813-1` (score=0.5359) - Periodic fever, aphthous stomatitis, pharyngitis, cervical adenitis (PFAPA) is a periodic disease, which is a heterogeneous group of disorders characterized by short episodes of illness that regularly recur for severa...
3. `medquad_ans_0005016-1` (score=0.5109) - Prader-Willi syndrome (PWS) is a genetic condition that affects many parts of the body. Infants with PWS have severe hypotonia (low muscle tone), feeding difficulties, and slow growth. In later infancy or early childh...
4. `medquad_ans_0000227-1__dup1` (score=0.5083) - Pelizaeus-Merzbacher disease (PMD) is a rare, progressive, degenerative central nervous system disorder in which coordination, motor abilities, and intellectual function deteriorate. The disease is one of a group of g...
5. `medquad_ans_0000054-7` (score=0.5080) - Peripheral arterial disease (P.A.D.) is a disease in which plaque (plak) builds up in the arteries that carry blood to your head, organs, and limbs. Plaque is made up of fat, cholesterol, calcium, fibrous tissue, and ...

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (relative constriction of the diaphysis and flaring of the metaphysis), widening of the ribs and clavicles, platyspondyly (flattening of the spine bones), and cortical thinning. Cranial involvement is minimal, with some cases showing mild hyperostosis of the skull base and thickening of the frontal and occipital bones. Pyle disease is inherited in an autosomal recessive manner [medquad_ans_0005212-1].

**Supported citations**: `medquad_ans_0005212-1`
**Unsupported citations**: _none_

### Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Scores**: context_relevance=0.600, groundedness=0.912, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000051-1__dup2` (score=0.6904) - Espaol Deep vein thrombosis (throm-BO-sis), or DVT, is a blood clot that forms in a vein deep in the body. Blood clots occur when blood thickens and clumps together. Most deep vein blood clots occur in the lower leg o...
2. `medquad_ans_0000256-1__dup3` (score=0.6757) - Deep vein thrombosis, or DVT, is a blood clot that forms in a vein deep in the body. Most deep vein clots occur in the lower leg or thigh. If the vein swells, the condition is called thrombophlebitis. A deep vein thro...
3. `medquad_ans_0000104-1__dup4` (score=0.6754) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent b...
4. `medquad_ans_0000051-4__dup2` (score=0.6653) - The risk factors for deep vein thrombosis (DVT) include: A history of DVT. Conditions or factors that make your blood thicker or more likely to clot than normal. Some inherited blood disorders (such as factor V Leiden...
5. `medquad_ans_0000051-5__dup1` (score=0.6563) - The signs and symptoms of deep vein thrombosis (DVT) might be related to DVT itself or pulmonary embolism (PE). See your doctor right away if you have signs or symptoms of either condition. Both DVT and PE can cause s...

**Answer**:

Deep vein thrombosis (DVT) is caused by blood thickening and clumping together, leading to the formation of a blood clot in a deep vein in the body. Most clots occur in the lower leg or thigh. Risk factors include certain genetic disorders, atherosclerosis, diabetes, atrial fibrillation, overweight or obesity, metabolic syndrome, some medicines, and smoking [medquad_ans_0000104-1__dup4]. Sitting still for a long time can also increase the likelihood of developing DVT [medquad_ans_0000256-1__dup3].

**Supported citations**: `medquad_ans_0000104-1__dup4`, `medquad_ans_0000256-1__dup3`
**Unsupported citations**: _none_

### Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.108, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004079-1` (score=0.7502) - Complex II deficiency is a mitochondrial disease. Mitochondria are specialized compartments in cells that create more than 90% of the energy needed by the body. In mitochondrial diseases, the mitochondria don't work c...
2. `medquad_ans_0000661-1` (score=0.7321) - Mitochondrial complex III deficiency is a genetic condition that can affect several parts of the body, including the brain, kidneys, liver, heart, and the muscles used for movement (skeletal muscles). Signs and sympto...
3. `medquad_ans_0004079-4` (score=0.7311) - How might mitochondrial complex II deficiency be treated? Treatment options for complex II deficiency may be similar to those for other mitochondrial disorders in general.[8677] The United Mitochondrial Disease Founda...
4. `medquad_ans_0004081-2` (score=0.7151) - What are the signs and symptoms of Mitochondrial complex IV deficiency? There are currently 4 known forms of COX deficiency. The range and severity of signs and symptoms can vary widely from case to case. In one form,...
5. `medquad_ans_0004079-2` (score=0.6666) - What are the signs and symptoms of Mitochondrial complex II deficiency? The signs and symptoms of mitochondrial complex II deficiency can vary greatly from severe life-threatening symptoms in infancy to muscle disease...

**Answer**:

The provided context does not contain any information about mitochondrial complex IV deficiency. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Scores**: context_relevance=1.000, groundedness=0.516, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000078-2` (score=0.7961) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.
2. `medquad_ans_0000078-1` (score=0.7686) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...
3. `medquad_ans_0000078-5` (score=0.7450) - These resources address the diagnosis or management of ataxia with oculomotor apraxia: - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type 1 - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type ...
4. `medquad_ans_0000496-1__dup1` (score=0.6977) - What are the signs and symptoms of Ataxia with Oculomotor Apraxia Type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Ataxia with Oculomotor Apraxia Type 2. If the information is...
5. `medquad_ans_0000495-2__dup1` (score=0.6567) - What are the signs and symptoms of Ataxia telangiectasia? Ataxia-telangiectasia affects the nervous system, immune system, and other body systems. This disorder is characterized by progressive difficulty with coordina...

**Answer**:

Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement, with the hallmark being difficulty coordinating movements (ataxia), which is often the first symptom. Most affected individuals also have oculomotor apraxia, making it difficult to move their eyes side-to-side, requiring them to turn their head to see peripheral vision. There are multiple types of this condition, which are very similar but caused by mutations in different genes. The two most common types (types 1 and 2) share features including involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, ataxia begins around age 4; in type 2, it begins around age 15. Chorea and myoclonus tend to disappear gradually in type 1 but persist throughout life in type 2. Type 1

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Scores**: context_relevance=1.000, groundedness=0.937, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000221-1` (score=0.8394) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that fo...
2. `medquad_ans_0001744-1` (score=0.7650) - What are the signs and symptoms of Deafness with labyrinthine aplasia microtia and microdontia (LAMM)? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness with labyrinthine apla...
3. `medquad_ans_0000221-5` (score=0.6839) - These resources address the diagnosis or management of LAMM syndrome: - Gene Review: Gene Review: Congenital Deafness with Labyrinthine Aplasia, Microtia, and Microdontia - Genetic Testing Registry: Deafness with laby...
4. `medquad_ans_0000623-1` (score=0.6040) - Mandibulofacial dysostosis with microcephaly (MFDM) is a disorder that causes abnormalities of the head and face. People with this disorder often have an unusually small head at birth, and the head does not grow at th...
5. `medquad_ans_0001496-1` (score=0.5966) - Congenital deafness with vitiligo and achalasia is a syndrome characterized by deafness present from birth (congenital), associated with short stature, vitiligo, muscle wasting and achalasia (swallowing difficulties)....

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects the development of the ears and teeth. It is characterized by the complete absence of inner ear structures (labyrinthine aplasia), leading to congenital sensorineural hearing loss. Affected individuals often have abnormally small outer ears (microtia) with narrow ear canals and unusually small, widely spaced teeth (microdontia). The condition may also cause delayed motor skill development due to balance issues from inner ear abnormalities. Additional signs include abnormalities of cranial nerves (seen in 90% of patients), microdontia (90%), and other features such as pointed chin, wide nasal bridge, and preauricular skin tags [medquad_ans_0000221-1][medquad_ans_0001744-1]. The syndrome

**Supported citations**: `medquad_ans_0000221-1`, `medquad_ans_0001744-1`
**Unsupported citations**: _none_

### Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.631, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000064-4` (score=0.8794) - Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases of Apert syndrome result from new mutations in ...
2. `medquad_ans_0000064-2` (score=0.7406) - Apert syndrome affects an estimated 1 in 65,000 to 88,000 newborns.
3. `medquad_ans_0000420-1__dup1` (score=0.6962) - Apert syndrome is a disorder mainly characterized by craniosynostosis (premature fusion of skull bones, causing abnormalities in the shape of the head and face) and syndactyly (fusion or webbing or fingers and/or toes...
4. `medquad_ans_0006597-4` (score=0.6893) - Can 48,XXYY syndrome be inherited?
5. `medquad_ans_0000064-1` (score=0.6751) - Apert syndrome is a genetic disorder characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the head and fa...

**Answer**:

Yes, Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. However, almost all cases result from new mutations and occur in individuals with no family history of the disorder. Individuals with Apert syndrome can pass the condition to their offspring [medquad_ans_0000064-4]. The disorder is caused by mutations in the FGFR2 gene and is inherited in an autosomal dominant manner [medquad_ans_000420-1__dup1].

**Supported citations**: `medquad_ans_0000064-4`
**Unsupported citations**: `medquad_ans_000420-1__dup1`

### Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Scores**: context_relevance=0.750, groundedness=0.538, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.7305) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organi...
2. `medquad_ans_0000341-1__dup3` (score=0.6347) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella bu...
3. `medquad_ans_0000341-4__dup2` (score=0.6104) - In the United States, Q fever outbreaks have resulted mainly from occupational exposure involving veterinarians, meat processing plant workers, sheep and dairy workers, livestock farmers, and researchers at facilities...
4. `medquad_ans_0000341-3__dup2` (score=0.6088) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact ...
5. `medquad_ans_0000092-4__dup3` (score=0.5439) - CHHF virus has been successfully isolated from both blood and serum during the acute febrile phase of illness. Although not undertaken at the time of the initial cluster, virus can certainly be isolated from tissue ob...

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air contaminated by dried placental material, birth fluids, and excreta of infected animals. Other rare modes of transmission include tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, individuals with acute Q fever develop a chronic form of the disease. Treatment of the acute form involves antibiotics, while treatment for the chronic form depends on the symptoms

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000038-1__dup2` (score=0.7218) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chroni...
2. `medquad_ans_0000038-5__dup1` (score=0.6716) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
3. `medquad_ans_0000038-9` (score=0.6529) - At this time, most types of childhood interstitial lung disease (chILD) can't be prevented. People who have a family history of inherited (genetic) interstitial lung disease may want to consider genetic counseling. A ...
4. `medquad_ans_0000038-4__dup2` (score=0.6315) - Researchers don't yet know all of the causes of childhood interstitial lung disease (chILD). Many times, these diseases have no clear cause. Some conditions and factors that may cause or lead to chILD include: Inherit...
5. `medquad_ans_0000038-7` (score=0.6112) - Doctors diagnose childhood interstitial lung disease (chILD) based on a child's medical and family histories and the results from tests and procedures. To diagnose chILD, doctors may first need to rule out other disea...

**Answer**:

Babies, children, and teens are at risk for Childhood Interstitial Lung Disease (chILD) [medquad_ans_0000038-1__dup2].

**Supported citations**: `medquad_ans_0000038-1__dup2`
**Unsupported citations**: _none_

### Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Scores**: context_relevance=1.000, groundedness=0.975, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.8903) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in lat...
2. `medquad_ans_0000547-1` (score=0.6063) - Juvenile idiopathic arthritis refers to a group of conditions involving joint inflammation (arthritis) that first appears before the age of 16. This condition is an autoimmune disorder, which means that the immune sys...
3. `medquad_ans_0000087-1__dup1` (score=0.5425) - Vasculitis is an inflammation of blood vessels, which includes the veins, arteries, and capillaries. Inflammation occurs with infection or is thought to be due to a faulty immune system response. It also can be caused...
4. `medquad_ans_0002107-1` (score=0.5186) - Enthesitis-related juvenile idiopathic arthritis is a subtype of juvenile idiopathic arthritis that is characterized by both arthritis and inflammation of an enthesitis site (the point at which a ligament, tendon, or ...
5. `medquad_ans_0003322-1` (score=0.5176) - What are the signs and symptoms of Juvenile idiopathic arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Juvenile idiopathic arthritis. If the information is available, the ...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels. It is generally diagnosed in late childhood or early adulthood and specifically affects the temporal arteries, which are located at the lower sides of the skull, directly underneath the temple. Affected individuals often have no signs or symptoms aside from a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically with no family history. It is generally treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

### Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.512, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0006577-1` (score=0.8187) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particul...
2. `medquad_ans_0001082-1` (score=0.7646) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as...
3. `medquad_ans_0001082-2` (score=0.7445) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyry...
4. `medquad_ans_0001090-2` (score=0.6972) - The exact incidence of 3-hydroxyacyl-CoA dehydrogenase deficiency is unknown; it has been reported in only a small number of people worldwide.
5. `medquad_ans_0001082-5` (score=0.6918) - These resources address the diagnosis or management of 2-methylbutyryl-CoA dehydrogenase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of 2-methylbutyryl-CoA dehydrogenase These resources from...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of the enzyme 2-methylbutyryl-CoA dehydrogenase, which is necessary for the proper processing of the amino acid isoleucine. This deficiency leads to the buildup of isoleucine and organic acids in the body, which can be toxic and cause a range of health problems. Symptoms vary widely, from severe and life-threatening to mild or absent, and may include poor feeding, lack of energy (lethargy), vomiting, irritability, difficulty breathing, seizures, coma, poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills such as standing and walking. Symptoms can be triggered by fasting, infections, or high-protein diets. The disorder is more common among individuals of Hmong ancestry, particularly in Hmong populations in southeast Asia and Hmong Americans,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Scores**: context_relevance=1.000, groundedness=0.537, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005840-2` (score=0.7658) - What are the signs and symptoms of Spondyloepiphyseal dysplasia congenita? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia congenita. If the information ...
2. `medquad_ans_0005841-2` (score=0.7334) - What are the signs and symptoms of Spondyloepiphyseal dysplasia Maroteaux type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia Maroteaux type. If the in...
3. `medquad_ans_0005827-1` (score=0.7323) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia joint laxity? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia joint laxity. If th...
4. `medquad_ans_0005835-1` (score=0.7289) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia with multiple dislocations? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia with ...
5. `medquad_ans_0005839-1` (score=0.7267) - What are the signs and symptoms of Spondyloepiphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia. If the information is available, the ta...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include abnormal form of the vertebral bodies (90%), abnormality of epiphysis morphology (90%), micromelia (90%), narrow chest (90%), short neck (90%), short stature (90%), short thorax (90%), skeletal dysplasia (90%), broad forehead (50%), cleft palate (50%), hyperlordosis (50%), hypertelorism (50%), malar flattening (50%), osteoarthritis (50%), talipes (50%), cataract (7.5%), glaucoma (7.5%), hearing impairment (7.5%), kyphosis (7.5%), myopia (7.5%), nystagmus (7.5%), retinal detachment (7.5%), and scoliosis (7.5%)

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Scores**: context_relevance=1.000, groundedness=0.935, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0005219-1` (score=0.8206) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepti...
2. `medquad_ans_0000853-2` (score=0.8141) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
3. `medquad_ans_0000853-1` (score=0.7797) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepti...
4. `medquad_ans_0005219-2` (score=0.7350) - What are the signs and symptoms of Pyridoxine-dependent epilepsy? Those affected by pyridoxine-dependent epilepsy typically experience prolonged seizures lasting several minutes (status epilepticus). These seizures in...
5. `medquad_ans_0005219-3` (score=0.7325) - What causes pyridoxine-dependent epilepsy? Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each ...

**Answer**:

Pyridoxine-dependent epilepsy is caused by mutations in the ALDH7A1 gene, which is inherited in an autosomal recessive fashion [medquad_ans_0005219-1].

**Supported citations**: `medquad_ans_0005219-1`
**Unsupported citations**: _none_

### Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Scores**: context_relevance=0.875, groundedness=0.938, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000004_6-2` (score=0.7258) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
2. `medquad_ans_0000004_1-2` (score=0.6648) - Age, gender, and Epstein-Barr infection can affect the risk of adult Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will g...
3. `medquad_ans_0000004_7-2` (score=0.6497) - Signs of childhood non-Hodgkin lymphoma include breathing problems and swollen lymph nodes. These and other signs may be caused by childhood non-Hodgkin lymphoma or by other conditions. Check with a doctor if your chi...
4. `medquad_ans_0000004_6-5` (score=0.6330) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whet...
5. `medquad_ans_0000004_2-3` (score=0.6236) - Age, gender, and a weakened immune system can affect the risk of adult non-Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you ...

**Answer**:

Children who are infected with the Epstein-Barr virus, infected with the human immunodeficiency virus (HIV), have certain diseases of the immune system, have a personal history of mononucleosis ("mono"), or have a parent or sibling with a personal history of Hodgkin lymphoma are at risk for childhood Hodgkin lymphoma [medquad_ans_0000004_6-2]. Being exposed to common infections in early childhood may decrease the risk of Hodgkin lymphoma in children due to its effect on the immune system [medquad_ans_0000004_6-2].

**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

### Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Scores**: context_relevance=1.000, groundedness=0.916, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0001795-1` (score=0.7940) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow...
2. `medquad_ans_0001795-2` (score=0.6524) - What are the signs and symptoms of Dermatitis herpetiformis ? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatitis herpetiformis . If the information is available, the table be...
3. `medquad_ans_0000095-2__dup3` (score=0.6518) - Dermatitis herpetiformis is caused by the deposit of immunoglobulin A (IgA) in the skin, which triggers further immunologic reactions resulting in lesion formation. DH is an external manifestation of an abnormal immun...
4. `medquad_ans_0001795-3` (score=0.6334) - How might dermatitis herpetiformis be treated? The antibiotic dapsone is extremely effective in treating this condition. Symptomatic improvement may occur in as little as several hours after the first dose. However, d...
5. `medquad_ans_0000095-1__dup4` (score=0.6012) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation ...

**Answer**:

Dermatitis herpetiformis is a rare, chronic skin disorder characterized by groups of severely itchy blisters and raised skin lesions, commonly found on the knees, elbows, buttocks, and shoulder blades. It is associated with the inability to digest gluten and is caused by the deposit of immunoglobulin A (IgA) in the skin, triggering an immune response to gluten, specifically targeting the skin antigen epidermal transglutaminase. The condition is frequently linked to autoimmunity and malabsorption, with symptoms including abnormal blistering, pruritus, red lesions resembling hives, and fluid-filled sores. It typically presents in adulthood but can also affect children. The disease is often treated with dapsone [medquad_ans_0001795-1]. Family studies indicate that approximately 5% of first-degree relatives will also have the condition [medquad_ans_00017

**Supported citations**: `medquad_ans_0001795-1`
**Unsupported citations**: _none_

### Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Scores**: context_relevance=0.857, groundedness=0.927, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000709-5` (score=0.6848) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Regi...
2. `medquad_ans_0000210-2__dup2` (score=0.6840) - Treatment for myotonia may include mexiletine, quinine, phenytoin, and other anticonvulsant drugs. Physical therapy and other rehabilitative measures may help muscle function.
3. `medquad_ans_0000040-2__dup1` (score=0.6830) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors ...
4. `medquad_ans_0000210-3__dup1` (score=0.6816) - Myotonia is a chronic disorder. Symptoms may improve later in life.
5. `medquad_ans_0000040-4__dup1` (score=0.6452) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to myotonia congenita and also supports additional research through grants to major research institutions across the countr...

**Answer**:

Treatments for myotonia congenita may include mexiletine, quinine, phenytoin, and other anticonvulsant drugs. Physical therapy and other rehabilitative measures may help improve muscle function. Most people with myotonia congenita do not require special treatments, as stiff muscles usually resolve with exercise or light movement, especially after resting. For individuals with more limiting symptoms, medications such as quinine or anticonvulsant drugs like phenytoin have been used successfully. [medquad_ans_0000210-2__dup2] [medquad_ans_0000040-2__dup1]

**Supported citations**: `medquad_ans_0000210-2__dup2`, `medquad_ans_0000040-2__dup1`
**Unsupported citations**: _none_

### Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Scores**: context_relevance=0.667, groundedness=0.864, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0006449-5` (score=0.7819) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lif...
2. `medquad_ans_0001037-2` (score=0.7217) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
3. `medquad_ans_0001037-5` (score=0.7211) - These resources address the diagnosis or management of Wilson disease: - Gene Review: Gene Review: Wilson Disease - Genetic Testing Registry: Wilson's disease - MedlinePlus Encyclopedia: Wilson's disease - National Hu...
4. `medquad_ans_0000133-12` (score=0.7208) - A person cannot prevent Wilson disease; however, people with a family history of Wilson disease, especially those with an affected sibling or parent, should talk with a health care provider about testing. A health car...
5. `medquad_ans_0006449-1` (score=0.7008) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver,...

**Answer**:

Treatment for Wilson disease involves lifelong management aimed at reducing or controlling copper accumulation in the body. This may include specific medications and dietary modifications. In cases where treatment is ineffective or liver failure develops, a liver transplant may be necessary [medquad_ans_0006449-5]. Early diagnosis, especially in individuals with a family history of the disease, can lead to timely intervention that reduces or prevents organ damage [medquad_ans_0000133-12]. People with a family history should consult a healthcare provider about testing and may benefit from genetic testing to identify gene mutations, with referral to a geneticist if needed [medquad_ans_0000133-12]. Additional resources for diagnosis and management include GeneReviews, the Genetic Testing Registry, MedlinePlus Encyclopedia, and the National Institute of Diabetes and Digestive and Kidney Disease (NIDDK) [medquad_ans_00

**Supported citations**: `medquad_ans_0006449-5`, `medquad_ans_0000133-12`
**Unsupported citations**: _none_

### Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.969, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000592-3` (score=0.7001) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are re...
2. `medquad_ans_0000592-2` (score=0.6735) - Lennox-Gastaut syndrome affects an estimated 1 in 50,000 to 1 in 100,000 children. This condition accounts for about 4 percent of all cases of childhood epilepsy. For unknown reasons, it appears to be more common in m...
3. `medquad_ans_0000592-4` (score=0.6443) - Most cases of Lennox-Gastaut syndrome are sporadic, which means they occur in people with no history of the disorder in their family. However, 3 to 30 percent of people with this condition have a family history of som...
4. `medquad_ans_0000592-5` (score=0.6442) - These resources address the diagnosis or management of Lennox-Gastaut syndrome: - Cleveland Clinic - Genetic Testing Registry: Epileptic encephalopathy Lennox-Gastaut type - National Institute of Neurological Disorder...
5. `medquad_ans_0000179-3__dup1` (score=0.6415) - The prognosis for individuals with Lennox-Gastaut syndrome varies. There is no cure for the disorder. Complete recovery, including freedom from seizures and normal development, is very unusual.

**Answer**:

Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component [medquad_ans_0000592-3].

**Supported citations**: `medquad_ans_0000592-3`
**Unsupported citations**: _none_

### Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.890, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000066-2` (score=0.9046) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
2. `medquad_ans_0000066-5` (score=0.7512) - These resources address the diagnosis or management of arginine:glycine amidinotransferase deficiency: - Gene Review: Gene Review: Creatine Deficiency Syndromes - Genetic Testing Registry: Arginine:glycine amidinotran...
3. `medquad_ans_0003472-1` (score=0.6912) - L-arginine:glycine amidinotransferase (AGAT) deficiency is a rare condition that primarily affects the brain. People with AGAT deficiency generally have mild to moderate intellectual disability. Other signs and sympto...
4. `medquad_ans_0003472-2` (score=0.6870) - What are the signs and symptoms of L-arginine:glycine amidinotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for L-arginine:glycine amidinotransferase deficiency....
5. `medquad_ans_0000445-1__dup1` (score=0.6201) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...

**Answer**:

The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families [medquad_ans_0000066-2]. Therefore, the exact number of people affected is not known.

**Supported citations**: `medquad_ans_0000066-2`
**Unsupported citations**: _none_

### Example 231 (`0002399-2`)

**Question**: What are the symptoms of Focal dermal hypoplasia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0002399-2` (score=0.7663) - What are the signs and symptoms of Focal dermal hypoplasia? Focal dermal hypoplasia is usually evident from birth and primarily affects the skin, skeleton, eyes, and face. The signs and symptoms of vary widely, althou...
2. `medquad_ans_0002399-1` (score=0.6996) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. The skin abnormalities are present from birth and can include streaks of very thin skin (dermal hypoplasia), cut...
3. `medquad_ans_0000379-2` (score=0.6723) - Focal dermal hypoplasia appears to be a rare condition, although its exact prevalence is unknown.
4. `medquad_ans_0000379-1` (score=0.6443) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. About 90 percent of affected individuals are female. Males usually have milder signs and symptoms than females. ...
5. `medquad_ans_0004054-1` (score=0.6405) - What are the signs and symptoms of Microsomia hemifacial radial defects? The Human Phenotype Ontology provides the following list of signs and symptoms for Microsomia hemifacial radial defects. If the information is a...

**Answer**:

The symptoms of Focal dermal hypoplasia include skin abnormalities such as streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), telangiectases, and streaks of slightly darker or lighter skin. These can cause pain, itching, irritation, or lead to skin infections. With age, most develop wart-like growths (papillomas) around the nostrils, lips, anus, and female genitalia, and may also be present in the throat, causing problems with swallowing, breathing, or sleeping. Other features include small, ridged fingernails and toenails, sparse or brittle scalp hair, hand and foot abnormalities such as missing fingers or toes (oligodactyly), webbed or fused fingers or toes (syndactyly), and a deep split in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Scores**: context_relevance=0.818, groundedness=0.860, answer_relevance=0.891

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=0.7251) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
2. `medquad_ans_0000418-1__dup2` (score=0.6902) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...
3. `medquad_ans_0000418-6` (score=0.6304) - Safe and effective prescription drugs are available to treat both Trichinella infection and the symptoms that occur as a result of infection. Treatment should begin as soon as possible; a doctor will make the decision...
4. `medquad_ans_0000418-5__dup1` (score=0.5908) - A diagnosis of trichinellosis is made in patients whose signs and symptoms are compatible with trichinellosis, have a positive laboratory test for Trichinella, and who can recall eating raw or undercooked pork or wild...
5. `medquad_ans_0000091-2__dup3` (score=0.5581) - Chagas disease, or American trypanosomiasis, is caused by the parasite Trypanosoma cruzi. Infection is most commonly acquired through contact with the feces of an infected triatomine bug (or "kissing bug"), a blood-su...

**Answer**:

People at risk for trichinellosis (also known as trichinosis) include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating undercooked wild game, especially bear meat, puts individuals at risk. Additionally, individuals who consume raw or undercooked meat of animals infected with Trichinella larvae, such as domestic pigs or wild boar, are also at risk [medquad_ans_0000418-1__dup2]. The risk from commercially raised and properly prepared pork is very low, but it remains a concern when meat is not cooked adequately [medquad_ans_0000418-2__dup1].

**Supported citations**: `medquad_ans_0000418-1__dup2`, `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

### Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001924-2` (score=0.7096) - What are the signs and symptoms of Dravet syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Dravet syndrome. If the information is available, the table below includes how oft...
2. `medquad_ans_0005016-2` (score=0.6171) - What are the signs and symptoms of Prader-Willi syndrome? In infancy, Prader-Willi syndrome (PWS) is characterized by weak muscle tone (hypotonia), feeding difficulties, poor growth, and delayed development. In later ...
3. `medquad_ans_0002063-2` (score=0.6080) - What are the signs and symptoms of Ehlers-Danlos syndrome, vascular type? The signs and symptoms of Ehlers-Danlos syndrome (EDS), vascular type vary but may include: Fragile tissues (including arteries, muscles and in...
4. `medquad_ans_0003700-2` (score=0.6048) - What are the signs and symptoms of Lymphedema-distichiasis syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lymphedema-distichiasis syndrome. If the information is available...
5. `medquad_ans_0002055-2` (score=0.5828) - What are the signs and symptoms of Ehlers-Danlos syndrome, dermatosparaxis type? The signs and symptoms of Ehlers-Danlos syndrome (EDS), dermatosparaxis type vary but may include: Soft, doughy skin that is extremely f...

**Answer**:

The symptoms of Dravet syndrome include absence seizures, ataxia, autosomal dominant inheritance, cerebral atrophy, cortical visual impairment, epileptic encephalopathy, focal seizures with impairment of consciousness or awareness, generalized myoclonic seizures, hemiclonic seizures, infantile onset, mental deterioration, motor delay, postnatal microcephaly, and status epilepticus [medquad_ans_0001924-2].

**Supported citations**: `medquad_ans_0001924-2`
**Unsupported citations**: _none_

### Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Scores**: context_relevance=0.889, groundedness=0.908, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0002533-3` (score=0.6081) - What causes Gerstmann-Straussler-Scheinker disease? Gerstmann-Straussler-Scheinker disease (GSS) is usually caused by certain changes (mutations) in the PRNP gene. PRNP encodes a protein called prion protein. Although...
2. `medquad_ans_0004784-4` (score=0.5919) - How is Pelizaeus-Merzbacher disease inherited?
3. `medquad_ans_0000011-13` (score=0.5828) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...
4. `medquad_ans_0000007-2__dup1` (score=0.5488) - Health Alert: Adrenal Crisis Causes Death in Some People Who Were Treated with Human Growth Hormone National Hormone and Pituitary Program (NHPP): Information for People Treated with Pituitary Human Growth Hormone (Su...
5. `medquad_ans_0006446-4` (score=0.5413) - Is Williams syndrome inherited?

**Answer**:

Genetic changes related to prion disease are mutations in the PRNP gene. These mutations result in the production of an abnormally shaped prion protein, which accumulates in the brain and forms clumps that damage or destroy neurons. This process leads to the signs and symptoms of prion diseases such as Gerstmann-Straussler-Scheinker disease (GSS) [medquad_ans_0002533-3]. In inherited cases of Creutzfeldt-Jakob disease (CJD), an abnormal gene leads to changes in the prion protein, making it more likely to adopt the abnormal shape [medquad_ans_0000011-13].

**Supported citations**: `medquad_ans_0002533-3`, `medquad_ans_0000011-13`
**Unsupported citations**: _none_

### Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.984, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0006430-1` (score=0.7064) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information ...
2. `medquad_ans_0003459-2` (score=0.6369) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficu...
3. `medquad_ans_0003474-2` (score=0.6130) - What are the signs and symptoms of Larsen syndrome? The signs and symptoms of Larsen syndrome vary from person to person, but may include the following: Joint dislocation (especially of the hips, knees, and elbows) Hy...
4. `medquad_ans_0002388-1` (score=0.6101) - What are the signs and symptoms of Fitzsimmons-Guilbert syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fitzsimmons-Guilbert syndrome. If the information is available, the ...
5. `medquad_ans_0006442-2` (score=0.6098) - What are the signs and symptoms of Wiedemann-Steiner syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Wiedemann-Steiner syndrome. If the information is available, the table ...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, hypoplasia of the radius, oligodactyly (hands), proximal placement of thumb, proximal radial head dislocation, cleft palate, cleft upper lip, high palate, hydronephrosis, hypotelorism, long face, mesomelia, narrow face, and high palate [medquad_ans_0006430-1].

**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

### Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Scores**: context_relevance=1.000, groundedness=0.910, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.7219) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the ta...
2. `medquad_ans_0004524-1` (score=0.6177) - What are the signs and symptoms of Okamoto syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Okamoto syndrome. If the information is available, the table below includes how o...
3. `medquad_ans_0002061-2` (score=0.5611) - What are the signs and symptoms of Ehlers-Danlos syndrome, progeroid type? Ehlers-Danlos syndrome refers to a group of connective tissue disorders characterized by stretchy or kneadable skin, double jointedness, and d...
4. `medquad_ans_0003546-2` (score=0.5548) - What are the signs and symptoms of Leprechaunism? The Human Phenotype Ontology provides the following list of signs and symptoms for Leprechaunism. If the information is available, the table below includes how often t...
5. `medquad_ans_0004961-1` (score=0.5413) - What are the signs and symptoms of Polyneuropathy mental retardation acromicria premature menopause? The Human Phenotype Ontology provides the following list of signs and symptoms for Polyneuropathy mental retardation...

**Answer**:

The symptoms of Premature aging Okamoto type include abnormality of the blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face [medquad_ans_0005029-1]. The frequency of these symptoms is based on data from the Human Phenotype Ontology, which collects information from sources such as Orphanet, with frequencies often listed as percentages or fractions representing the proportion of patients exhibiting each symptom [medquad_ans_0005029-1].

**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

### Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.771, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000329-2` (score=0.6617) - Essential thrombocythemia affects an estimated 1 to 24 per 1 million people worldwide.
2. `medquad_ans_0001065-2` (score=0.6443) - The estimated incidence of X-linked thrombocytopenia is between 1 and 10 per million males worldwide; this condition is rarer among females.
3. `medquad_ans_0000979-1` (score=0.6297) - Thrombocytopenia-absent radius (TAR) syndrome is characterized by the absence of a bone called the radius in each forearm. Affected individuals also have a shortage (deficiency) of blood cells involved in clotting (pl...
4. `medquad_ans_0000484-2` (score=0.5782) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
5. `medquad_ans_0000770-2` (score=0.5754) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.

**Answer**:

The provided context does not specify the exact number of people affected by thrombocytopenia-absent radius (TAR) syndrome. While it mentions that TAR syndrome is characterized by a deficiency of platelets and associated malformations, and that it is unusual among similar conditions, no incidence or estimated prevalence is given for TAR syndrome in the context. Therefore, based on the available information, the number of people affected by thrombocytopenia-absent radius syndrome cannot be determined. [medquad_ans_0000979-1]

**Supported citations**: `medquad_ans_0000979-1`
**Unsupported citations**: _none_

### Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Scores**: context_relevance=0.455, groundedness=0.732, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000972-5` (score=0.7628) - These resources address the diagnosis or management of Tay-Sachs disease: - Gene Review: Gene Review: Hexosaminidase A Deficiency - Genetic Testing Registry: Tay-Sachs disease - MedlinePlus Encyclopedia: Tay-Sachs Dis...
2. `medquad_ans_0006017-5` (score=0.7442) - How might children with Tay-Sachs disease be treated? Although several attempts have been made at purified enzyme replacement therapy for children with Tay-Sachs disease, none has been successful. Cellular infusions a...
3. `medquad_ans_0000871-1__dup2` (score=0.7402) - Tay-Sachs disease is a rare, inherited disorder. It causes too much of a fatty substance to build up in the brain. This buildup destroys nerve cells, causing mental and physical problems. Infants with Tay-Sachs diseas...
4. `medquad_ans_0000265-2__dup1` (score=0.7211) - Presently there is no specific treatment for Tay-Sachs disease. Anticonvulsant medicine may initially control seizures. Other supportive treatment includes proper nutrition and hydration and techniques to keep the air...
5. `medquad_ans_0000265-1__dup1` (score=0.6780) - Tay-Sachs disease is a inherited metabolic disease caused by the harmful buildup of lipids (fatty materials such as oils and acids) in various cells and tissues in the body. It is part of a group of genetic disorders ...

**Answer**:

The provided context does not mention any ongoing research or clinical trials for Tay-Sachs disease. [medquad_ans_0000972-5], [medquad_ans_0006017-5], [medquad_ans_0000871-1__dup2], and [medquad_ans_0000265-2__dup1] discuss diagnosis, management, and supportive care but do not reference specific research or clinical trials. Therefore, based on the given information, no such details are available.

**Supported citations**: `medquad_ans_0000972-5`, `medquad_ans_0006017-5`, `medquad_ans_0000871-1__dup2`, `medquad_ans_0000265-2__dup1`
**Unsupported citations**: _none_

### Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Scores**: context_relevance=0.750, groundedness=0.550, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000008-5` (score=0.6092) - These resources address the diagnosis or management of achromatopsia: - Gene Review: Gene Review: Achromatopsia - Genetic Testing Registry: Achromatopsia - MedlinePlus Encyclopedia: Color Vision Test These resources f...
2. `medquad_ans_0000008-2` (score=0.5335) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on o...
3. `medquad_ans_0000008-1` (score=0.5196) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achro...
4. `medquad_ans_0000025-1__dup1` (score=0.4674) - An acanthoma is a small, reddish bump that usually develops on the skin of an older adult. There are several types of acanthoma, including "acantholytic", "epidermolytic", "clear cell", and "melanoacanthoma". Though m...
5. `medquad_ans_0000008-3` (score=0.4616) - Achromatopsia results from changes in one of several genes: CNGA3, CNGB3, GNAT2, PDE6C, or PDE6H. A particular CNGB3 gene mutation underlies the condition in Pingelapese islanders. Achromatopsia is a disorder of the r...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. The condition also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals may also have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia results from changes in one of several genes: CNGA3, CNGB3, GNAT2, PDE6C, or PDE6H. It is a disorder of the retina,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Scores**: context_relevance=0.857, groundedness=0.904, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000039-11` (score=0.7354) - Treatments your doctor will likely recommend before knee replacement include - exercises to strengthen the muscles around the knee and improve flexibility - weight loss, if needed, to reduce the load the knee must bea...
2. `medquad_ans_0000039-26` (score=0.6625) - You can learn more about knee replacement from the following resources. National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) Information Clearinghouse National Institutes of Health 1 AMS Circl...
3. `medquad_ans_0000544-1__dup1` (score=0.6460) - Knee replacement is surgery for people with severe knee damage. Knee replacement can relieve pain and allow you to be more active. Your doctor may recommend it if you have knee pain and medicine and other treatments a...
4. `medquad_ans_0000061-13` (score=0.6391) - Medication, exercise, and, in some cases, surgery are common treatments for this disease. Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation...
5. `medquad_ans_0000029-18` (score=0.6295) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.

**Answer**:

The treatments recommended before knee replacement include exercises to strengthen the muscles around the knee and improve flexibility, weight loss if needed to reduce the load on the knee, using walking aids such as canes to reduce stress on the joint, wearing shoe inserts to improve knee alignment, and taking medicines to relieve pain [medquad_ans_0000039-11].

Knee replacement surgery itself involves removing damaged cartilage and bone from the knee joint and replacing them with a man-made surface of metal and plastic in a total knee replacement, or replacing only one part of the joint in a partial knee replacement [medquad_ans_0000544-1__dup1]. After surgery, patients may experience scarring, blood clots, and rarely infections, and will generally no longer be able to perform certain activities like jogging or high-impact sports [medquad_ans_0000544-1\_\_dup

**Supported citations**: `medquad_ans_0000039-11`, `medquad_ans_0000544-1__dup1`
**Unsupported citations**: _none_

### Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.943, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000090-5__dup1` (score=0.6994) - Metabolic syndrome is a group of risk factors that raises your risk for heart disease and other health problems, such as diabetes and stroke. These risk factors can increase your risk for health problems even if they'...
2. `medquad_ans_0000044-11` (score=0.6668) - Metabolic syndrome, also called insulin resistance syndrome, is a group of traits and medical conditions linked to overweight and obesity that puts people at risk for both CVD and type 2 diabetes. Metabolic syndrome i...
3. `medquad_ans_0000608-1__dup1` (score=0.6668) - Metabolic syndrome is a group of conditions that put you at risk for heart disease and diabetes. These conditions are - High blood pressure - High blood glucose, or blood sugar, levels - High levels of triglycerides, ...
4. `medquad_ans_0000032-6__dup1` (score=0.6429) - Metabolic syndrome is a grouping of traits and medical conditions that puts people at risk for both heart disease and type 2 diabetes. It is defined by the National Cholesterol Education Program as having any three of...
5. `medquad_ans_0000090-3__dup2` (score=0.6227) - Metabolic syndrome has several causes that act together. You can control some of the causes, such as overweight and obesity, an inactive lifestyle, and insulin resistance. You can't control other factors that may play...

**Answer**:

The symptoms of metabolic syndrome include increased thirst, increased urination (especially at night), fatigue (tiredness), and blurred vision, which are symptoms of high blood sugar when diabetes—especially type 2 diabetes—is present. High blood pressure usually has no signs or symptoms, although some people in the early stages may experience dull headaches, dizziness, or more frequent nosebleeds. A large waistline is a visible sign of metabolic syndrome. However, most of the metabolic risk factors have no signs or symptoms [medquad_ans_0000090-5__dup1]. [medquad_ans_0000044-11]

**Supported citations**: `medquad_ans_0000090-5__dup1`, `medquad_ans_0000044-11`
**Unsupported citations**: _none_

### Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Scores**: context_relevance=1.000, groundedness=0.957, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005211-1` (score=0.6909) - What are the signs and symptoms of Pyknoachondrogenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyknoachondrogenesis. If the information is available, the table below includ...
2. `medquad_ans_0002152-2` (score=0.5888) - What are the signs and symptoms of Erdheim-Chester disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Erdheim-Chester disease. If the information is available, the table below ...
3. `medquad_ans_0004625-2` (score=0.5856) - What are the signs and symptoms of Osteomesopyknosis? Osteomesopyknosis may cause chronic, low-grade back pain in the thoracic (middle) and lumbar (lower) regions. It is considered a mild form of osteosclerosis and is...
4. `medquad_ans_0005210-2` (score=0.5815) - What are the signs and symptoms of Pycnodysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pycnodysostosis. If the information is available, the table below includes how oft...
5. `medquad_ans_0000309-2__dup1` (score=0.5702) - What are the signs and symptoms of Amyotrophic lateral sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyotrophic lateral sclerosis. If the information is available, the ...

**Answer**:

The symptoms of Pyknoachondrogenesis include abnormal vertebral ossification, abnormality of pelvic girdle bone morphology, abnormality of the mouth, abnormality of the sacrum, depressed nasal ridge, enlarged thorax, increased bone mineral density, low-set, posteriorly rotated ears, macrocephaly, micromelia, palpebral edema, premature birth, short stature, short thorax, thickened nuchal skin fold, and stillbirth. These symptoms are reported in approximately 90% of patients with this condition [medquad_ans_0005211-1].

**Supported citations**: `medquad_ans_0005211-1`
**Unsupported citations**: _none_

### Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003965-2` (score=0.7569) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the ta...
2. `medquad_ans_0003966-1` (score=0.6372) - What are the signs and symptoms of Metachromatic leukodystrophy due to saposin B deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy due to sapos...
3. `medquad_ans_0003566-1` (score=0.6291) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to def...
4. `medquad_ans_0000647-2` (score=0.6157) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigra...
5. `medquad_ans_0000555-1__dup1` (score=0.6103) - The leukodystrophies are rare diseases that affect the cells of the brain. Specifically, the diseases affect the myelin sheath, the material that surrounds and protects nerve cells. Damage to this sheath slows down or...

**Answer**:

The symptoms of Metachromatic leukodystrophy include behavioral abnormality, cognitive impairment, decreased nerve conduction velocity, developmental regression, gait disturbance, genu recurvatum, incoordination, muscle weakness, neurological speech impairment, peripheral neuropathy, reduced consciousness/confusion, seizures, amaurosis fugax, hyperreflexia, hypertonia, limitation of joint mobility, muscular hypotonia, nystagmus, optic atrophy, aganglionic megacolon, ataxia, Babinski sign, bulbar palsy, cholecystitis, chorea, delusions, dysarthria, dystonia, EMG: neuropathic changes, emotional lability, gallbladder dysfunction, hallucinations, hyporeflexia, increased CSF protein, intellectual disability, loss of speech, mental deterioration, peripheral demyelination, progressive peripheral neuropathy, spastic tetraplegia, tetraplegia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Scores**: context_relevance=0.600, groundedness=0.000, answer_relevance=0.920

**Top docs**:

1. `medquad_ans_0000673-4` (score=0.6100) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
2. `medquad_ans_0000585-4` (score=0.6018) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000362-4` (score=0.6018) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000095-4` (score=0.6018) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
5. `medquad_ans_0004097-4` (score=0.5933) - Is mixed connective tissue disease inherited? The role of genetics in the onset of mixed connective tissue disease (MCTD) is still unclear. Some people with MCTD have family members who are also affected by the condit...

**Answer**:

The provided context does not contain information about whether chronic granulomatous disease is inherited. Therefore, based on the given context, it is not possible to determine if chronic granulomatous disease is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.272, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000061-4__dup1` (score=0.7505) - Our understanding of central cord syndrome has increased greatly in recent decades as a result of research funded conducted by the National Institute of Neurological Disorders and Stroke (NINDS). Much of this research...
2. `medquad_ans_0000061-3__dup1` (score=0.7214) - The prognosis for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict he likeliho...
3. `medquad_ans_0000061-2__dup1` (score=0.6540) - There is no cure for central cord syndrome although some people recover near-normal function. There is no standard course of treatment, although drug therapy, surgery, and rest are often part of the program. Magnetic ...
4. `medquad_ans_0000061-1__dup1` (score=0.6400) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to a...
5. `medquad_ans_0000085-2__dup1` (score=0.5577) - Currently, only central core disease has an effective treatment (see above). There are no known cures for any of these disorders. Supportive treatment may involve orthopedic treatments, as well as physical, occupation...

**Answer**:

The outlook for central cord syndrome is generally favorable, with most people experiencing some recovery of neurological function, particularly after trauma. Many individuals recover substantial function following their initial injury, and the ability to walk is regained in most cases, although some impairment may remain. Recovery typically begins in the legs, followed by the bladder, and hand function recovers last, if at all. Improvement is generally better in younger patients compared to those over the age of 50. Early medical intervention is associated with good outcomes, and evaluation of abnormal signals on MRI images can help predict the likelihood of natural neurological recovery. While there is no cure, early surgery may improve recovery chances, and surgery is beneficial in cases of persistent spinal cord compression and ongoing neurological deterioration [medquad_ans_0000061-3__dup1, medquad_ans_0000061-2__dup1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000061-3__dup1, medquad_ans_0000061-2__dup1`

### Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Scores**: context_relevance=0.600, groundedness=0.935, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000094-13` (score=0.7803) - - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no ...
2. `medquad_ans_0000094-1__dup6` (score=0.7768) - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no ap...
3. `medquad_ans_0000094-3__dup3` (score=0.7407) - The cause of cyclic vomiting syndrome is unknown. However, some experts believe that some possible problems with bodily functions may contribute to the cause, such as the following: - gastrointestinal motilitythe way ...
4. `medquad_ans_0000094-4__dup3` (score=0.7318) - Cyclic vomiting syndrome is more common in children than adults, although reports of the syndrome in adults have increased in recent years.1 Usually, children are about 5 years old when diagnosed with cyclic vomiting ...
5. `medquad_ans_0000094-6__dup1` (score=0.6955) - The main symptoms of cyclic vomiting syndrome are severe nausea and sudden vomiting lasting hours to days. A person may also experience one or more of the following symptoms: - retching, or making an attempt to vomit ...

**Answer**:

The cause of cyclic vomiting syndrome is unknown. However, some experts believe that possible problems with bodily functions may contribute to the cause, such as gastrointestinal motility, central nervous system function, autonomic nervous system function, and hormone imbalances [medquad_ans_0000094-3__dup3].

**Supported citations**: `medquad_ans_0000094-3__dup3`
**Unsupported citations**: _none_

### Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Scores**: context_relevance=0.875, groundedness=0.924, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000180-1__dup2` (score=0.6558) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not alwa...
2. `medquad_ans_0000603-1__dup1` (score=0.5630) - Summary : Mental health includes our emotional, psychological, and social well-being. It affects how we think, feel and act as we cope with life. It also helps determine how we handle stress, relate to others, and mak...
3. `medquad_ans_0000189-1__dup2` (score=0.5046) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...
4. `medquad_ans_0000776-5` (score=0.4927) - These resources address the diagnosis or management of Partington syndrome: - American Academy of Child and Adolescent Psychiatry: Services in School for Children with Special Needs - American Academy of Pediatrics: W...
5. `medquad_ans_0000176-1__dup2` (score=0.4916) - All kids misbehave some times. And some may have temporary behavior problems due to stress. For example, the birth of a sibling, a divorce, or a death in the family may cause a child to act out. Behavior disorders are...

**Answer**:

Yes, there is information about child mental health. Early recognition and treatment of mental illnesses in children are important because once mental illness develops, it becomes a regular part of the child's behavior and is more difficult to treat [medquad_ans_0000180-1__dup2]. Warning signs that may indicate a more serious mental health problem include problems in more than one setting (at school, at home, with peers), changes in appetite or sleep, social withdrawal or fear of things previously not feared, returning to younger behaviors like bedwetting, signs of being upset such as sadness or tearfulness, self-destructive behaviors like head-banging or frequent injuries, and repeated thoughts of death [medquad_ans_0000180-1__dup2]. Diagnosis involves evaluating the child's signs and symptoms, medical history, and family history [medquad_ans_0000180-1\_\_dup

**Supported citations**: `medquad_ans_0000180-1__dup2`
**Unsupported citations**: _none_

### Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.052, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000673-2__dup1` (score=0.5678) - What are the signs and symptoms of Behcet's disease? Symptoms of Behcet's disease include recurrent ulcers in the mouth (resembling canker sores) and on the genitals, and eye inflammation (uveitis). The disorder may a...
2. `medquad_ans_0000003-2__dup3` (score=0.5430) - Signs and symptoms of Cushing's syndrome vary, but most people with the disorder have upper body obesity, a rounded face, increased fat around the neck, and relatively slender arms and legs. Children tend to be obese ...
3. `medquad_ans_0002066-2` (score=0.5387) - What are the signs and symptoms of Eisenmenger syndrome? Symptoms of Eisenmenger include shortness of breath, chest pain, feeling tired or dizzy, fainting, abnormal heart rhythm (arrhythmia), stroke, coughing up blood...
4. `medquad_ans_0004451-2` (score=0.5262) - What are the signs and symptoms of Noonan syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome 1. If the information is available, the table below includes how...
5. `medquad_ans_0000757-2__dup1` (score=0.5108) - What are the signs and symptoms of Blepharophimosis, ptosis, and epicanthus inversus syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Blepharophimosis, ptosis, and ep...

**Answer**:

The provided context does not contain any information about the symptoms of Cowden syndrome. Therefore, it is not possible to answer the question based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.833, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0000395-5` (score=0.7255) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
2. `medquad_ans_0000395-2` (score=0.6798) - Fukuyama congenital muscular dystrophy is seen almost exclusively in Japan, where it is the second most common form of childhood muscular dystrophy (after Duchenne muscular dystrophy). Fukuyama congenital muscular dys...
3. `medquad_ans_0000395-1` (score=0.6265) - Fukuyama congenital muscular dystrophy is an inherited condition that predominantly affects the muscles, brain, and eyes. Congenital muscular dystrophies are a group of genetic conditions that cause muscle weakness an...
4. `medquad_ans_0004219-1` (score=0.6233) - Muscular dystrophy (MD) refers to a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or ...
5. `medquad_ans_0000631-1__dup1` (score=0.6135) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...

**Answer**:

The provided context does not specify treatments for Fukuyama congenital muscular dystrophy. While it mentions resources related to diagnosis and management, such as genetic counseling, drug therapy, surgery and rehabilitation, palliative care, and diagnostic tests, it does not detail specific treatments for this condition [medquad_ans_0000395-5]. Therefore, based on the available information, the exact treatments are not outlined.

**Supported citations**: `medquad_ans_0000395-5`
**Unsupported citations**: _none_

### Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000004_7-2` (score=0.7840) - Signs of childhood non-Hodgkin lymphoma include breathing problems and swollen lymph nodes. These and other signs may be caused by childhood non-Hodgkin lymphoma or by other conditions. Check with a doctor if your chi...
2. `medquad_ans_0000004_6-1` (score=0.7121) - Key Points - Childhood Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two types of childhood Hodgkin lymphoma. - Epstein-Barr virus infection increases the risk o...
3. `medquad_ans_0000483-1__dup1` (score=0.6880) - Hodgkin disease is a type of lymphoma. Lymphoma is a cancer of a part of the immune system called the lymph system. The first sign of Hodgkin disease is often an enlarged lymph node. The disease can spread to nearby l...
4. `medquad_ans_0000004_6-3` (score=0.6874) - Signs of childhood Hodgkin lymphoma include swollen lymph nodes, fever, night sweats, and weight loss. These and other signs and symptoms may be caused by childhood Hodgkin lymphoma or by other conditions. Check with ...
5. `medquad_ans_0000004_7-1` (score=0.6572) - Key Points - Childhood non-Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - The main types of lymphoma are Hodgkin lymphoma and non-Hodgkin lymphoma. - There are three major ...

**Answer**:

Tests that examine the lymph system are used to detect (find) and diagnose childhood Hodgkin lymphoma [medquad_ans_0000004_6-1].

**Supported citations**: `medquad_ans_0000004_6-1`
**Unsupported citations**: _none_

### Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000117-4` (score=0.6639) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
2. `medquad_ans_0006446-4` (score=0.6501) - Is Williams syndrome inherited?
3. `medquad_ans_0000673-4` (score=0.6143) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
4. `medquad_ans_0000786-4` (score=0.6054) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
5. `medquad_ans_0000149-4` (score=0.6017) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

The provided context does not contain information about KBG syndrome or its inheritance pattern. Therefore, it is not possible to answer the question based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Scores**: context_relevance=0.833, groundedness=0.993, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000195-1__dup1` (score=0.7732) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
2. `medquad_ans_0000193-1__dup1` (score=0.7732) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
3. `medquad_ans_0000194-1__dup1` (score=0.7732) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000946-1__dup1` (score=0.5251) - Cardiofaciocutaneous (CFC) syndrome is a disorder that affects many parts of the body, particularly the heart (cardio-), face (facio-), and the skin and hair (cutaneous). People with this condition also have developme...
5. `medquad_ans_0000674-1__dup1` (score=0.4973) - Behr syndrome is a disorder mainly characterized by early-onset optic atrophy, ataxia, and spasticity. Other signs and symptoms may be present and vary from person to person. Although the exact cause is unknown, the s...

**Answer**:

Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and physical disability. Additional symptoms may include epilepsy, painful, itchy skin lesion (chilblains), vision problems, and joint stiffness. Symptoms usually progress over several months before the disease course stabilizes. There are six different types of Aicardi-Goutieres syndrome, which are distinguished by the gene that causes the condition: TREX1, RNASEH2A, RNASEH2B, RNASEH2C, SAMHD1, and ADAR genes. Most cases are inherited in an autosomal recessive pattern, although rare autosomal dominant cases have been reported. Treatment is symptomatic and supportive [medquad_ans_0000195-1__dup1].

**Supported citations**: `medquad_ans_0000195-1__dup1`
**Unsupported citations**: _none_

### Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000026_1-7` (score=0.6747) - Key Points - There are different types of treatment for patients with renal cell cancer. - Five types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Biologic therapy - Targeted therapy ...
2. `medquad_ans_0000026_2-7` (score=0.6403) - Key Points - There are different types of treatment for patients with transitional cell cancer of the renal pelvis and ureter. - One type of standard treatment is used: - Surgery - New types of treatment are being tes...
3. `medquad_ans_0001354-2` (score=0.6294) - What treatments for metastatic clear cell renal cell carcinoma are available in North America? There are several treatments for metastatic clear cell renal cell carcinoma available in North America. IL-2 and sunitinib...
4. `medquad_ans_0000026_1-1` (score=0.5955) - Key Points - Renal cell cancer is a disease in which malignant (cancer) cells form in tubules of the kidney. - Smoking and misuse of certain pain medicines can affect the risk of renal cell cancer. - Signs of renal ce...
5. `medquad_ans_0000026_3-7` (score=0.5942) - Key Points - There are different types of treatment for patients with Wilms tumor and other childhood kidney tumors. - Children with Wilms tumor or other childhood kidney tumors should have their treatment planned by ...

**Answer**:

Research involving clinical trials is being conducted for renal cell cancer to improve current treatments or obtain information on new treatments. These clinical trials are designed to test new therapies and may include treatments that are not yet standard. Patients may consider participating in clinical trials, which can be open before, during, or after starting cancer treatment. When clinical trials demonstrate that a new treatment is more effective than the current standard treatment, it may become the new standard of care [medquad_ans_0000026_1-7].

**Supported citations**: `medquad_ans_0000026_1-7`
**Unsupported citations**: _none_

### Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Scores**: context_relevance=0.857, groundedness=0.469, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000190-6` (score=0.6593) - Treatment for primary biliary cirrhosis depends on how early a health care provider diagnoses the disease and whether complications are present. In the early stages of primary biliary cirrhosis, treatment can slow the...
2. `medquad_ans_0000190-9` (score=0.6167) - - Primary biliary cirrhosis is a chronic disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. - When chronic inflammation damages the bile ducts, bile and toxi...
3. `medquad_ans_0000183-10` (score=0.6103) - - Biliary atresia is a life-threatening condition in infants in which the bile ducts inside or outside the liver do not have normal openings. - The first symptom of biliary atresia is jaundicewhen the skin and whites ...
4. `medquad_ans_0000193-7` (score=0.5899) - To prevent cirrhosis, - see your doctor for treatment of your liver disease. Many of the causes of cirrhosis are treatable. Early treatment may prevent cirrhosis. - try to keep your weight in the normal range. Being o...
5. `medquad_ans_0000190-2__dup2` (score=0.5792) - The causes of primary biliary cirrhosis are unknown. Most research suggests it is an autoimmune disease. The immune system protects people from infection by identifying and destroying bacteria, viruses, and other pote...

**Answer**:

Treatment for primary biliary cirrhosis depends on the stage of the disease and the presence of complications. In the early stages, treatment can slow the progression of liver damage to cirrhosis. The primary medication prescribed is ursodiol (Actigall, Urso), a nontoxic bile acid that replaces more toxic bile acids produced by the liver, reducing levels of bilirubin and liver enzymes in the blood. Early treatment with ursodiol reduces the likelihood of needing a liver transplant and improves survival, although it does not cure the disease. Treatment benefits are greatest when started early, but ursodiol can still slow disease progression even when used later. Patients should avoid alcohol and illegal substances, as these can worsen liver damage. Complementary and alternative medications, such as herbs, should also be avoided. Patients should consult a healthcare provider before taking any new medications, over-the-counter drugs, or vitamins, as these can affect

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Scores**: context_relevance=0.750, groundedness=0.968, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=0.5756) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People ...
2. `medquad_ans_0000967-1__dup1` (score=0.4721) - Summary : Playing in the water - whether swimming, boating or diving - can be fun. It can also be dangerous, especially for children. Being safe can help prevent injuries and drowning. To stay safe in the water - Avoi...
3. `medquad_ans_0000378-2` (score=0.3907) - Floating-Harbor syndrome is a rare disorder; only about 50 cases have been reported in the medical literature.
4. `medquad_ans_0000378-4` (score=0.3892) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of Floating-Harbor syndrome result from new mutations...
5. `medquad_ans_0000916-1__dup2` (score=0.3387) - Summary : A tsunami is a series of huge ocean waves created by an underwater disturbance. Causes include earthquakes, landslides, volcanic eruptions, or meteorites--chunks of rock from space that strike the surface of...

**Answer**:

Drowning occurs when people get too much water in their lungs. It can happen quickly and silently, and can occur in as little as an inch or two of water. Babies can drown in a sink or bathtub, and preschoolers are most likely to drown in a swimming pool. People with seizure disorders are also at risk in the water [medquad_ans_0000299-1__dup2]. Drowning is the most common cause of death related to a tsunami [medquad_ans_0000916-1__dup2].

**Supported citations**: `medquad_ans_0000299-1__dup2`, `medquad_ans_0000916-1__dup2`
**Unsupported citations**: _none_

### Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Scores**: context_relevance=1.000, groundedness=0.466, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002058-1` (score=0.8713) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragi...
2. `medquad_ans_0002058-5` (score=0.8050) - How is Ehlers-Danlos syndrome, kyphoscoliosis type diagnosed? A diagnosis of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is typically based on the presence of characteristic signs and symptoms. The following tes...
3. `medquad_ans_0002058-3` (score=0.7661) - What causes Ehlers-Danlos syndrome, kyphoscoliosis type? Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is caused by changes (mutations) in the PLOD1 gene. This gene encodes an enzyme that helps process molecules w...
4. `medquad_ans_0000313-1` (score=0.7589) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and sympt...
5. `medquad_ans_0000313-4` (score=0.7513) - The inheritance pattern of Ehlers-Danlos syndrome varies by type. The arthrochalasia, classical, hypermobility, and vascular forms of the disorder have an autosomal dominant pattern of inheritance. Autosomal dominant ...

**Answer**:

Ehlers-Danlos syndrome, kyphoscoliosis type is an inherited connective tissue disorder caused by mutations in the PLOD1 gene, which encodes an enzyme essential for collagen formation and stability. This disorder leads to defective collagen networks, resulting in weakened connective tissues and characteristic signs and symptoms such as hyperextensible skin that is fragile and bruises easily, joint hypermobility, severe hypotonia at birth, progressive kyphoscoliosis (kyphosis and scoliosis), and fragility of the sclera. It is inherited in an autosomal recessive manner and is diagnosed based on clinical signs and symptoms, with confirmation through urine tests, skin biopsy to detect enzyme deficiencies, or genetic testing for mutations in the PLOD1 gene [medquad_ans_0002058-1, medquad_ans_0002058-5, medquad_ans_000

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001860-2` (score=0.6468) - What are the signs and symptoms of Dihydropteridine reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropteridine reductase deficiency. If the information is...
2. `medquad_ans_0001862-1` (score=0.6028) - Dihydropyrimidine dehydrogenase (DPD) deficiency is a condition in which the body cannot break down the nucleotides thymine and uracil. DPD deficiency can have a wide range of severity; some individuals may have vario...
3. `medquad_ans_0006567-2` (score=0.5845) - What are the signs and symptoms of 2,4-Dienoyl-CoA reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for 2,4-Dienoyl-CoA reductase deficiency. If the information is a...
4. `medquad_ans_0000288-2` (score=0.5800) - Severe dihydropyrimidine dehydrogenase deficiency, with its early-onset neurological symptoms, is a rare disorder. Its prevalence is unknown. However, between 2 and 8 percent of the general population may be vulnerabl...
5. `medquad_ans_0001860-1` (score=0.5759) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include cognitive impairment, microcephaly, cerebral calcification, choreoathetosis, dysphagia, dystonia, episodic fever, excessive salivation, hyperphenylalaninemia, hypertonia, infantile onset, intellectual disability, irritability, muscular hypotonia, myoclonus, progressive neurologic deterioration, seizures, tremor, and variable expressivity. These symptoms occur in approximately 90% of patients for cognitive impairment and microcephaly [medquad_ans_0001860-2].

**Supported citations**: `medquad_ans_0001860-2`
**Unsupported citations**: _none_

### Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Scores**: context_relevance=0.286, groundedness=0.725, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000669-2` (score=0.7903) - The prevalence of monilethrix is unknown.
2. `medquad_ans_0000484-2` (score=0.5723) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
3. `medquad_ans_0000770-2` (score=0.5634) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.
4. `medquad_ans_0000989-2` (score=0.5470) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000195-2` (score=0.5431) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.

**Answer**:

The prevalence of monilethrix is unknown [medquad_ans_0000669-2]. However, estimates suggest it affects 1 in 87,000 people [medquad_ans_0000484-2], 1 to 3 per million people worldwide [medquad_ans_0000770-2], 1 in 50,000 people [medquad_ans_0000989-2], and 1 in 40,000 to 50,000 people [medquad_ans_0000195-2]. Due to the variability in estimates and the lack of a definitive prevalence, the exact number of people affected remains uncertain.

**Supported citations**: `medquad_ans_0000669-2`, `medquad_ans_0000484-2`, `medquad_ans_0000770-2`, `medquad_ans_0000989-2`, `medquad_ans_0000195-2`
**Unsupported citations**: _none_

### Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Scores**: context_relevance=1.000, groundedness=0.515, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.7599) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...
2. `medquad_ans_0002733-2` (score=0.6168) - What are the signs and symptoms of Hartnup disease? The signs and symptoms of Hartnup disease may vary and include the following: Skin findings: sensitivity to sunlight Neurologic symptoms: ataxia, spasticity, headach...
3. `medquad_ans_0002152-1` (score=0.5159) - Erdheim-Chester disease is a rare condition that can affect many different organs of the body. This condition, which usually affects adults, is characterized by excessive production and accumulation of histiocytes (sp...
4. `medquad_ans_0000372-2` (score=0.5124) - Feingold syndrome appears to be a rare condition, although its exact prevalence is unknown.
5. `medquad_ans_0000708-2` (score=0.5013) - The prevalence of this condition is unknown.

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system, making it a type of aminoaciduria. It is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. Most people with Hartnup disease are asymptomatic, but those who do show symptoms typically experience onset between the ages of 3 and 9, although it may occasionally present in adulthood. The signs and symptoms include skin photosensitivity, neurologic findings (such as ataxia, spasticity, headaches, and hypotonia), psychiatric symptoms (like anxiety, emotional instability, and mood changes), and ocular findings (such as double vision, nystagmus, strabismus, and photophobia). Symptoms may be triggered by sunlight exposure, fever, drugs, or emotional or physical stress and often last 1–4 weeks before spontaneous remission

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Scores**: context_relevance=0.833, groundedness=0.774, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000427-5` (score=0.6929) - These resources address the diagnosis or management of glycogen storage disease type VI: - Gene Review: Gene Review: Glycogen Storage Disease Type VI - Genetic Testing Registry: Glycogen storage disease, type VI These...
2. `medquad_ans_0000423-5` (score=0.6888) - These resources address the diagnosis or management of glycogen storage disease type III: - Gene Review: Gene Review: Glycogen Storage Disease Type III - Genetic Testing Registry: Glycogen storage disease type III The...
3. `medquad_ans_0002622-1` (score=0.6857) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepa...
4. `medquad_ans_0000428-5` (score=0.6785) - These resources address the diagnosis or management of glycogen storage disease type VII: - Genetic Testing Registry: Glycogen storage disease, type VII - The Swedish Information Centre for Rare Diseases These resourc...
5. `medquad_ans_0002615-3` (score=0.6577) - What causes glycogen storage disease type 13? Glycogen storage disease type 13 (GSD13) is caused by changes (mutations) in the ENO3 gene. Glycogen is a substance that is stored in muscle tissue and is used as an impor...

**Answer**:

The provided context does not specify the genetic changes related to glycogen storage disease type VII. [medquad_ans_0000428-5] mentions resources for diagnosis and management of glycogen storage disease type VII, including the Genetic Testing Registry, but does not detail the underlying genetic mutations or changes. Therefore, based on the given information, the genetic changes related to glycogen storage disease type VII cannot be determined.

**Supported citations**: `medquad_ans_0000428-5`
**Unsupported citations**: _none_

### Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.921, answer_relevance=0.711

**Top docs**:

1. `medquad_ans_0000977-2` (score=0.8256) - Thiamine-responsive megaloblastic anemia syndrome has been reported in approximately 30 families worldwide. Its prevalence is unknown.
2. `medquad_ans_0006061-1` (score=0.7838) - Thiamine-responsive megaloblastic anemia syndrome is a very rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Affected individuals begin to show symptoms of this...
3. `medquad_ans_0000977-5` (score=0.7542) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megal...
4. `medquad_ans_0000977-1` (score=0.6912) - Thiamine-responsive megaloblastic anemia syndrome is a rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Megaloblastic anemia occurs when a person has a low numb...
5. `medquad_ans_0006061-2` (score=0.6758) - What are the signs and symptoms of Thiamine responsive megaloblastic anemia syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thiamine responsive megaloblastic anemia syndrom...

**Answer**:

The treatment for thiamine-responsive megaloblastic anemia syndrome is high doses of vitamin B1 (thiamine), which can treat the anemia associated with the condition [medquad_ans_0006061-1]. However, it remains unclear whether thiamine treatment can improve hearing loss or prevent its development [medquad_ans_0000977-1]. Individuals with the syndrome usually require insulin to manage their diabetes [medquad_ans_0000977-1].

**Supported citations**: `medquad_ans_0006061-1`, `medquad_ans_0000977-1`
**Unsupported citations**: _none_

### Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.962, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000587-4__dup1` (score=0.6743) - Is there a cure or treatment for autosomal recessive polycystic kidney disease? Although a cure or treatment for the underlying genetic cause of autosomal recessive polycystic kidney disease does not exist, advancemen...
2. `medquad_ans_0000804-5` (score=0.6371) - These resources address the diagnosis or management of polycystic kidney disease: - Gene Review: Gene Review: Polycystic Kidney Disease, Autosomal Dominant - Gene Review: Gene Review: Polycystic Kidney Disease, Autoso...
3. `medquad_ans_0004939-2` (score=0.6102) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their abilit...
4. `medquad_ans_0000804-1` (score=0.6044) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the...
5. `medquad_ans_0004939-1` (score=0.6015) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the ...

**Answer**:

Medical management of polycystic kidney disease is currently symptomatic and involves supportive care. This includes treatment of high blood pressure with medication, management of urinary tract infections with antibiotics, and treatment of low sodium levels (hyponatremia) with diuresis and/or sodium supplementation. When kidneys are severely enlarged, nephrectomy (removal of one or both kidneys) may be performed. Dialysis may be required during the first days of life if an infant produces little or no urine (oliguria or anuria). Kidney failure may require dialysis or kidney transplantation. Growth failure may be managed with gastrostomy tubes or growth hormone therapy. Mechanical ventilation may be used to treat breathing issues due to enlarged kidneys. For liver involvement, a shunt may be required to manage progressive high blood pressure, and liver transplantation may be considered [medquad_ans_0000587-4__dup1]. Resources for diagnosis and

**Supported citations**: `medquad_ans_0000587-4__dup1`
**Unsupported citations**: _none_

### Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Scores**: context_relevance=0.818, groundedness=0.926, answer_relevance=0.527

**Top docs**:

1. `medquad_ans_0000061-5__dup2` (score=0.5763) - Scientists are making rapid progress in understanding the complexities of rheumatoid arthritis. They are learning more about how and why it develops and why some people have more severe symptoms than others. Research ...
2. `medquad_ans_0000061-14` (score=0.5566) - Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation. Still others, called disease-modifying anti-rheumatic drugs or DMARDs, can often slow th...
3. `medquad_ans_0000061-4__dup3` (score=0.5556) - Most Symptoms Are Treatable Doctors use a variety of approaches to treat rheumatoid arthritis. The goals of treatment are to help relieve pain, reduce swelling, slow down or help prevent joint damage, increase the abi...
4. `medquad_ans_0000061-13` (score=0.5530) - Medication, exercise, and, in some cases, surgery are common treatments for this disease. Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation...
5. `medquad_ans_0000061-15` (score=0.5397) - Rest. People with rheumatoid arthritis need a good balance between rest and exercise; they should rest more when the disease is active and exercise more when it is not. Rest helps to reduce active joint inflammation a...

**Answer**:

Research on rheumatoid arthritis is focused on understanding how and why the disease develops, particularly identifying possible triggers such as infectious agents like viruses and bacteria that may contribute to the disease in individuals with a genetic predisposition [medquad_ans_0000061-5__dup2]. Scientists are also investigating the complex relationships between the hormonal, nervous, and immune systems to explain why more women than men develop the disease, including how changes in steroid hormones like estrogen and testosterone may influence disease development or flares [medquad_ans_0000061-5__dup2]. Additionally, researchers are exploring why rheumatoid arthritis often improves during pregnancy.

Efforts are underway to develop new drugs that can reduce inflammation and slow or stop the disease with fewer side effects. This includes research into disease-modifying anti-rheumatic drugs (DMARDs) such as methotrexate, leflunomide, sulfas

**Supported citations**: `medquad_ans_0000061-5__dup2`
**Unsupported citations**: _none_

### Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.976, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0002561-4` (score=0.8037) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (an...
2. `medquad_ans_0002561-2` (score=0.7160) - What are the signs and symptoms of Glass-Chapman-Hockley syndrome? Glass-Chapman-Hockley syndrome has only been described in one family with five affected family members in three generations. The signs and symptoms se...
3. `medquad_ans_0002561-1` (score=0.6835) - The Glass-Chapman-Hockley syndrome is a very rare disease. To date, the syndrome has only been reported in one family with five members affected in three generations. The first patients were two brothers that had an a...
4. `medquad_ans_0000860-2__dup1` (score=0.6004) - How might Brown-Sequard syndrome be treated?
5. `medquad_ans_0000883-5__dup1` (score=0.5753) - How might Buschke Ollendorff syndrome be treated? There is currently no cure for BOS. Surgical removal of lesions on or under the skin may be done for cosmetic purposes. In some patients, surgical treatment of deafnes...

**Answer**:

The treatment for Glass-Chapman-Hockley syndrome is typically surgery, which is based on the person's specific signs and symptoms. The goal of the surgery is to increase the space in the front (anterior) part of the skull. The operation is usually performed when the person is between 9 to 12 months of age. If other sutures, other than the coronal suture, are involved, other surgeries may be performed [medquad_ans_0002561-4].

**Supported citations**: `medquad_ans_0002561-4`
**Unsupported citations**: _none_

### Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.825, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000868-2` (score=0.6628) - Renpenning syndrome is a rare disorder; its prevalence is unknown. More than 60 affected individuals in at least 15 families have been identified.
2. `medquad_ans_0000247-2__dup2` (score=0.5580) - There is no cure for Rett syndrome. Treatment for the disorder is symptomatic, focusing on the management of symptoms, and supportive. Medication may be needed for breathing irregularities and motor difficulties, and ...
3. `medquad_ans_0003615-3` (score=0.5565) - How might CREST syndrome be treated? Unfortunately, CREST syndrome has no known cure. The condition carries both physical and psychological consequences, so a holistic approach to management should be taken. Treatment...
4. `medquad_ans_0000146-2__dup1` (score=0.5542) - There are no standard treatments for hereditary neuropathies. Treatment is mainly symptomatic and supportive. Medical treatment includes physical therapy and if needed, pain medication. Orthopedic surgery may be neede...
5. `medquad_ans_0005119-5` (score=0.5486) - How might Proud syndrome be treated? The treatment of Proud syndrome is based on the signs and symptoms present in each person. For example, spasticity may be treated with a variety of therapies including medications ...

**Answer**:

The provided context does not contain information about the treatments for Renpenning syndrome. [medquad_ans_0000868-2] mentions that Renpenning syndrome is a rare disorder with an unknown prevalence and that over 60 affected individuals in at least 15 families have been identified, but it does not specify any treatments. Therefore, based on the available context, the treatments for Renpenning syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000868-2`
**Unsupported citations**: _none_

### Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Scores**: context_relevance=1.000, groundedness=0.977, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001105-1` (score=0.7830) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is ava...
2. `medquad_ans_0001100-1__dup1` (score=0.7792) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is a...
3. `medquad_ans_0001111-1` (score=0.7737) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2N? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2N. If the information is ava...
4. `medquad_ans_0001112-1` (score=0.7697) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2O? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2O. If the information is ava...
5. `medquad_ans_0001110-1` (score=0.7641) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is ava...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 2G include onion bulb formation (7.5%), areflexia, autosomal recessive inheritance, axonal degeneration/regeneration, decreased motor nerve conduction velocity, decreased number of peripheral myelinated nerve fibers, distal amyotrophy, distal muscle weakness, distal sensory impairment, flexion contracture, neonatal onset, pes cavus, spinal deformities, split hand, and vocal cord paresis [medquad_ans_0001105-1].

**Supported citations**: `medquad_ans_0001105-1`
**Unsupported citations**: _none_

### Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001393-2` (score=0.7140) - What signs and symptoms are associated with coccygodynia? The classic symptom is pain when pressure is applied to the tailbone, such as when sitting on a hard chair. Symptoms usually improve with relief of pressure wh...
2. `medquad_ans_0000676-2__dup1` (score=0.5845) - What are the symptoms of Bell's palsy?
3. `medquad_ans_0003881-2` (score=0.5662) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speec...
4. `medquad_ans_0001393-1` (score=0.5600) - Coccygodynia is a rare condition in that causes pain in and around the coccyx (tailbone). Although various causes have been described for the condition, the more common causes are direct falls and injury.
5. `medquad_ans_0000037-2__dup1` (score=0.5418) - What are the signs and symptoms of achalasia? Most people with achalasia experience difficulty swallowing, also known as dysphagia and heartburn. Other symptoms might include: regurgitation or vomiting, noncardiac che...

**Answer**:

The symptoms of coccygodynia include pain when pressure is applied to the tailbone, such as when sitting on a hard chair, which usually improves with relief of pressure when standing or walking. Other symptoms may include immediate and severe pain when moving from sitting to standing, pain during bowel movements, pain during sex, and a deep ache in the region of the tailbone [medquad_ans_0001393-2].

**Supported citations**: `medquad_ans_0001393-2`
**Unsupported citations**: _none_

### Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Scores**: context_relevance=1.000, groundedness=0.890, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0004025-1` (score=0.7703) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgen...
2. `medquad_ans_0003783-2` (score=0.6969) - What are the signs and symptoms of Mandibulofacial dysostosis with microcephaly? Mandibulofacial dysostosis with microcephaly (MFDM) may affect multiple parts of the body but primarily affects the head and face. Peopl...
3. `medquad_ans_0003270-2` (score=0.6828) - What are the signs and symptoms of Isolated corpus callosum agenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Isolated corpus callosum agenesis. If the information is availab...
4. `medquad_ans_0002446-2` (score=0.6630) - What are the signs and symptoms of Frontonasal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontonasal dysplasia. If the information is available, the table below incl...
5. `medquad_ans_0002040-1` (score=0.6518) - What are the signs and symptoms of Ectrodactyly and ectodermal dysplasia without cleft lip/palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Ectrodactyly and ectodermal dysplas...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, and microcephaly. These symptoms are listed in the Human Phenotype Ontology as part of the signs and symptoms associated with this condition [medquad_ans_0004025-1]. Additionally, other symptoms such as preaxial hand polydactyly, proptosis, and seizures may also be present, though their frequency varies and is not always specified [medquad_ans_0004025-1].

**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

### Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.798, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000562-5` (score=0.7923) - These resources address the diagnosis or management of Klippel-Feil syndrome: - Genetic Testing Registry: Klippel Feil syndrome - Genetic Testing Registry: Klippel-Feil syndrome 1, autosomal dominant - Genetic Testing...
2. `medquad_ans_0000170-3__dup1` (score=0.7671) - The prognosis for most individuals with Klippel-Feil Syndrome is good if the disorder is treated early and appropriately. Activities that can injure the neck should be avoided.
3. `medquad_ans_0000170-2__dup1` (score=0.7516) - Treatment for Klippel-Feil Syndrome is symptomatic and may include surgery to relieve cervical or craniocervical instability and constriction of the spinal cord, and to correct scoliosis. Physical therapy may also be ...
4. `medquad_ans_0003402-2` (score=0.7503) - What are the signs and symptoms of Klippel Feil syndrome? Klippel Feil syndrome is characterized by the fusion of 2 or more spinal bones in the neck (cervical vertebrae). The condition is present from birth (congenita...
5. `medquad_ans_0003402-5` (score=0.7387) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to d...

**Answer**:

The provided context does not contain specific information on how to diagnose Klippel-Feil syndrome. While it mentions resources such as the Genetic Testing Registry and MedlinePlus that offer information on diagnosis and management, no detailed diagnostic methods or criteria are described in the given text. Therefore, based on the available context, the diagnosis of Klippel-Feil syndrome cannot be fully addressed.

[medquad_ans_0000562-5]

**Supported citations**: `medquad_ans_0000562-5`
**Unsupported citations**: _none_

### Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.833, groundedness=0.942, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000222-2` (score=0.8027) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
2. `medquad_ans_0000222-1` (score=0.7908) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
3. `medquad_ans_0001497-1` (score=0.7794) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severit...
4. `medquad_ans_0000222-5` (score=0.7393) - These resources address the diagnosis or management of congenital diaphragmatic hernia: - Boston Children's Hospital - Children's Hospital of Philadelphia - Columbia University Medical Center: DHREAMS - Columbia Unive...
5. `medquad_ans_0000222-4` (score=0.7272) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or c...

**Answer**:

Congenital diaphragmatic hernia (CDH) is a defect in the diaphragm that results from abnormal development before birth. The diaphragm, composed of muscle and other fibrous tissue, normally separates the organs in the abdomen from those in the chest cavity. CDH ranges in severity from a thinned area in the diaphragm to its complete absence. This defect allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs, which can lead to underdevelopment of the lungs (pulmonary hypoplasia) and potentially life-threatening breathing difficulties at birth [medquad_ans_0000222-1]. CDH may also present later in life with symptoms such as breathing problems or abdominal pain due to intestinal protrusion into the chest cavity, or it may be asymptomatic and detected incidentally during imaging for other reasons [medquad_ans_000022

**Supported citations**: `medquad_ans_0000222-1`
**Unsupported citations**: _none_

### Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Scores**: context_relevance=1.000, groundedness=0.517, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000173-2` (score=0.7999) - Charcot-Marie-Tooth disease is the most common inherited disorder that involves the peripheral nerves, affecting an estimated 150,000 people in the United States. It occurs in populations worldwide with a prevalence o...
2. `medquad_ans_0001088-1__dup1` (score=0.7195) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
3. `medquad_ans_0001093-1__dup1` (score=0.7134) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is ava...
4. `medquad_ans_0001090-1__dup1` (score=0.7092) - Charcot-Marie-Tooth disease type 1A (CMT1A) is a type of inherited neurological disorder that affects the peripheral nerves. Affected individuals experience weakness and wasting (atrophy) of the muscles of the lower l...
5. `medquad_ans_0000173-1` (score=0.7058) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...

**Answer**:

Charcot-Marie-Tooth disease is the most common inherited disorder that involves the peripheral nerves, affecting an estimated 150,000 people in the United States and occurring in populations worldwide with a prevalence of about 1 in 2,500 individuals. It is a group of disorders that affect the peripheral nerves, which run from the brain and spine to the rest of the body. Defects in many different genes cause different forms of this disease. Common symptoms include foot drop, foot deformity, loss of lower leg muscle, numbness in the foot or leg, slapping gait, and weakness of the hips, legs, or feet. There is currently no cure, but management can include physical therapy, occupational therapy, braces, orthopedic devices, pain medication, and orthopedic surgery. There are over 40 types of Charcot-Marie-Tooth disease [medquad_ans_0000

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000405-4` (score=0.3668) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000891-4` (score=0.3668) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0001090-4` (score=0.3668) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000741-4` (score=0.3668) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000801-4` (score=0.3668) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether monilethrix is inherited. While several documents describe autosomal recessive inheritance patterns, none explicitly mention monilethrix or its mode of inheritance. Therefore, based on the given information, it is not possible to determine if monilethrix is inherited.

[Note: Insufficient context to answer the question.]

**Supported citations**: _none_
**Unsupported citations**: `Note: Insufficient context to answer the question.`

### Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Scores**: context_relevance=0.875, groundedness=0.914, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000223-12` (score=0.6809) - - A urinary tract infection (UTI) is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscope. - Most UTIs are caused by bacteria that live in the bowel, ...
2. `medquad_ans_0000215-15` (score=0.6789) - - Urinary tract infections (UTIs) usually occur when the body fails to remove bacteria rapidly from the urinary tract. - UTIs affect about 3 percent of children in the United States every year. - Most UTIs are not ser...
3. `medquad_ans_0000215-1__dup3` (score=0.6685) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Norma...
4. `medquad_ans_0000214-1__dup4` (score=0.6685) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Norma...
5. `medquad_ans_0000071-7__dup1` (score=0.6470) - Urinary tract infections (UTI) are a common bladder problem, especially as people age. UTIs are the second most common type of infection in the body. Each year, UTIs cause more than 8 million visits to health care pro...

**Answer**:

A urinary tract infection (UTI) in children is an infection in the urinary tract caused by microorganisms too small to be seen without a microscope, most commonly bacteria [medquad_ans_0000215-15]. These infections occur when bacteria enter the urinary tract and overcome the body's natural defenses, which normally remove bacteria rapidly before they cause symptoms [medquad_ans_0000214-1__dup4]. Bacteria that live in the bowel are a common source of UTIs [medquad_ans_0000223-12]. UTIs can affect any child, though girls are more likely to get them than boys [medquad_ans_0000223-12]. Symptoms may range from mild, such as a burning sensation during urination or unusual-smelling urine, to severe, including high fever and pain, or the child may have no symptoms

**Supported citations**: `medquad_ans_0000215-15`, `medquad_ans_0000214-1__dup4`, `medquad_ans_0000223-12`
**Unsupported citations**: _none_

### Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0004939-2` (score=0.7448) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their abilit...
2. `medquad_ans_0000804-1` (score=0.7228) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the...
3. `medquad_ans_0004939-1` (score=0.6869) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the ...
4. `medquad_ans_0000804-2` (score=0.6816) - Polycystic kidney disease is a fairly common genetic disorder. It affects about 500,000 people in the United States. The autosomal dominant form of the disease is much more common than the autosomal recessive form. Au...
5. `medquad_ans_0000167-1__dup4` (score=0.6680) - Polycystic kidney disease is a genetic disorder that causes numerous cysts to grow in the kidneys. A kidney cyst is an abnormal sac filled with fluid. PKD cysts can greatly enlarge the kidneys while replacing much of ...

**Answer**:

Complications of Polycystic Kidney Disease include dangerously high blood pressure (hypertension), severe pain in the back or sides, blood in the urine (hematuria), recurrent urinary tract infections, kidney stones (nephrolithiasis), heart valve abnormalities, and an increased risk of aortic aneurysm in the brain (cerebral aneurysm), which can be life-threatening if it tears or ruptures. Other complications include cystic liver disease, anemia, proteinuria, renal insufficiency, abnormality of prenatal development or birth, abnormality of the pancreas, abnormality of the respiratory system, dilatation of the ascending aorta, hydrocephalus, recurrent fractures, reduced bone mineral density, and sarcoma [medquad_ans_0004939-2].

**Supported citations**: `medquad_ans_0004939-2`
**Unsupported citations**: _none_

### Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Scores**: context_relevance=0.800, groundedness=0.519, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000649-1__dup1` (score=0.7043) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contami...
2. `medquad_ans_0000949-1__dup1` (score=0.5342) - Viruses are capsules with genetic material inside. They are very tiny, much smaller than bacteria. Viruses cause familiar infectious diseases such as the common cold, flu and warts. They also cause severe illnesses su...
3. `medquad_ans_0000123-4__dup2` (score=0.5042) - Four types of viruses cause most cases of viral gastroenteritis. Rotavirus Rotavirus is the leading cause of gastroenteritis among infants and young children. Rotavirus infections are most common in infants 3 to 15 mo...
4. `medquad_ans_0000092-1__dup1` (score=0.4840) - Cytomegalovirus (CMV) is a virus found throughout the world that infects between 50 to 80 percent of all adults in the United States by the age of 40. CMV is in the same family of viruses that causes cold sores (herpe...
5. `medquad_ans_0000228-4__dup2` (score=0.4722) - Laboratory tests that are used to diagnose Hendra virus (HV) and Nipah virus (NV) include detection of antibody by ELISA (IgG and IgM), real time polymerase chain reaction (RT-PCR), and virus isolation attempts. In mo...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by noroviruses, which are a group of related viruses. Infection leads to an inflammation of the stomach and intestines, with common symptoms including diarrhea, nausea and vomiting, and stomach pain. Other symptoms may include fever, headache, or body aches. The illness can spread from person to person, through contaminated food or water, or by touching a contaminated surface. Norovirus is the most common calicivirus and the most common cause of viral gastroenteritis in adults, often responsible for epidemics, especially from October to April. Symptoms typically appear 1 to 2 days after exposure and last for 1 to 3 days. There is no specific medicine to treat norovirus infections; treatment includes bed rest and plenty of fluids to prevent dehydration. Proper hand washing and safe food preparation can help prevent infections [medquad_ans_0000649

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.900, groundedness=0.941, answer_relevance=0.740

**Top docs**:

1. `medquad_ans_0000222-4` (score=0.7629) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or c...
2. `medquad_ans_0000222-2` (score=0.7452) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
3. `medquad_ans_0001497-3` (score=0.7450) - What causes congenital diaphragmatic hernia? Congenital diaphragmatic hernia (CDH) can occur as an isolated finding, as part of a genetic syndrome or chromosome abnormality, or as part of a complex but nonsyndromic se...
4. `medquad_ans_0000222-5` (score=0.7217) - These resources address the diagnosis or management of congenital diaphragmatic hernia: - Boston Children's Hospital - Children's Hospital of Philadelphia - Columbia University Medical Center: DHREAMS - Columbia Unive...
5. `medquad_ans_0000222-3` (score=0.7118) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...

**Answer**:

Genetic changes related to congenital diaphragmatic hernia (CDH) include:

- Recognized chromosome abnormalities [medquad_ans_0001497-3].
- Single gene disorders [medquad_ans_0001497-3].
- Currently undetectable chromosomal microdeletions (tiny loss of genetic material) or microduplications (an extra copy of genetic material) [medquad_ans_0001497-3].
- Mutations in major genes important for diaphragm development [medquad_ans_0001497-3].
- Combined effects of multiple minor genetic mutations or variants (polygenic inheritance) [medquad_ans_0001497-3].
- Gene-environment interactions (multifactorial inheritance) [medquad_ans_0001497-3].

**Supported citations**: `medquad_ans_0001497-3`
**Unsupported citations**: _none_

### Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.052, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0002533-3` (score=0.6103) - What causes Gerstmann-Straussler-Scheinker disease? Gerstmann-Straussler-Scheinker disease (GSS) is usually caused by certain changes (mutations) in the PRNP gene. PRNP encodes a protein called prion protein. Although...
2. `medquad_ans_0004784-4` (score=0.5441) - How is Pelizaeus-Merzbacher disease inherited?
3. `medquad_ans_0004750-3` (score=0.5425) - What causes Parsonage Turner syndrome? The exact cause of Parsonage Turner syndrome (PTS) is unknown. Researchers suspect that most cases are due to an autoimmune response following exposure to an illness or environme...
4. `medquad_ans_0002840-1` (score=0.5383) - Hereditary neuralgic amyotrophy is a type of nervous system disease that affects the brachial plexus. Common signs and symptoms include episodes of severe pain and muscle wasting in one or both shoulders and arms. Att...
5. `medquad_ans_0000011-13` (score=0.5329) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...

**Answer**:

The provided context does not contain information about the causes of Omenn syndrome. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000097-1` (score=0.7377) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...
2. `medquad_ans_0005792-1` (score=0.6785) - Spinocerebellar ataxia autosomal recessive 7, also called SCAR7, is a slowly progressive hereditary form of spinocerebellar ataxia. Symptoms of SCAR7 can include difficulty walking and writing, speech difficulties (dy...
3. `medquad_ans_0005761-1` (score=0.6486) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal...
4. `medquad_ans_0000097-5` (score=0.6432) - These resources address the diagnosis or management of ARCA1: - Gene Review: Gene Review: SYNE1-Related Autosomal Recessive Cerebellar Ataxia - Genetic Testing Registry: Spinocerebellar ataxia, autosomal recessive 8 -...
5. `medquad_ans_0005794-1` (score=0.6284) - What are the signs and symptoms of Spinocerebellar ataxia autosomal recessive with axonal neuropathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia autosomal...

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the cerebellum, the part of the brain that coordinates movement. Signs and symptoms first appear in early to mid-adulthood and include impaired speech (dysarthria), problems with coordination and balance (ataxia), or both. Individuals may also have difficulty with movements that involve judging distance or scale (dysmetria). Other features include abnormal eye movements (nystagmus) and problems following the movements of objects with the eyes. The movement problems are slowly progressive and may eventually result in the need for a cane, walker, or wheelchair [medquad_ans_0000097-1].

**Supported citations**: `medquad_ans_0000097-1`
**Unsupported citations**: _none_

### Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Scores**: context_relevance=0.750, groundedness=0.927, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000050-2__dup1` (score=0.6923) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
2. `medquad_ans_0000049-2__dup1` (score=0.6923) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
3. `medquad_ans_0000050-4__dup1` (score=0.6775) - The NINDS conducts and supports research on injuries to the nervous system such as brachial plexus injuries. Much of this research is aimed at finding ways to prevent and treat these disorders.
4. `medquad_ans_0000049-4__dup1` (score=0.6775) - The NINDS conducts and supports research on injuries to the nervous system such as brachial plexus injuries. Much of this research is aimed at finding ways to prevent and treat these disorders.
5. `medquad_ans_0000123-1__dup2` (score=0.6724) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Symptoms may include - A limp or paralyzed...

**Answer**:

The treatments for brachial plexus injuries include physical therapy and, in some cases, surgery [medquad_ans_0000050-2__dup1].

**Supported citations**: `medquad_ans_0000050-2__dup1`
**Unsupported citations**: _none_

### Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Scores**: context_relevance=0.833, groundedness=0.938, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000216-7` (score=0.8066) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
2. `medquad_ans_0000216-8` (score=0.7832) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PU...
3. `medquad_ans_0000216-4__dup1` (score=0.7448) - When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis. Hydronephrosis is the most common problem found during prenatal ...
4. `medquad_ans_0000216-3__dup1` (score=0.7331) - Before leaving the hospital, a baby with urine blockage may urinate only small amounts or may not urinate at all. As part of the routine newborn exam, the health care provider may feel an enlarged kidney or find a clo...
5. `medquad_ans_0000154-7` (score=0.6843) - The complications of kidney dysplasia can include - hydronephrosis of the working kidney. A baby with kidney dysplasia in only one kidney might have other urinary tract defects. When other defects in the urinary tract...

**Answer**:

Urine blockage in newborns can be caused by various defects in the urinary tract, including vesicoureteral reflux (VUR), ureteropelvic junction (UPJ) obstruction, bladder outlet obstruction (BOO) such as posterior urethral valves (PUV), and ureterocele. Genetic conditions affecting multiple body systems, such as prune belly syndrome (PBS), esophageal atresia (EA), and congenital heart defects, can also contribute. Additionally, spina bifida and other birth defects affecting the spinal cord may cause urine blockage [medquad_ans_0000216-8]. When a defect blocks urine flow, it causes urine to back up, leading to hydroureter and hydronephrosis [medquad_ans_0000216-8].

**Supported citations**: `medquad_ans_0000216-8`
**Unsupported citations**: _none_

### Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Scores**: context_relevance=0.556, groundedness=0.036, answer_relevance=0.644

**Top docs**:

1. `medquad_ans_0003179-1` (score=0.5487) - Inclusion body myopathy 2, also known as hereditary inclusion body myopathy (HIBM), GNE-related myopathy, distal myopathy with rimmed vacuoles, and Nonaka myopathy, is an inherited condition that primarily affects the...
2. `medquad_ans_0000707-1__dup1` (score=0.5231) - Bethlem myopathy is an inherited movement disorder characterized by progressive muscle weakness and joint stiffness (contractures) in the fingers, wrists, elbows, and ankles. Due to a progressive course, up to two-thi...
3. `medquad_ans_0001035-1__dup1` (score=0.5118) - Centronuclear myopathy refers to a group of rare, inherited conditions that affect the muscles. There are three main forms of the condition that are differentiated by their pattern of inheritance: X-linked Myotubular ...
4. `medquad_ans_0005086-3` (score=0.5014) - How might progressive hemifacial atrophy be treated?
5. `medquad_ans_0006236-4` (score=0.4985) - Is tubular aggregate myopathy genetic? It is evident from family history studies that the condition can be passed through families in either an autosomal dominant or autosomal recessive fashion. Some cases appear to b...

**Answer**:

The provided context does not contain information about the number of people affected by intranuclear rod myopathy. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Scores**: context_relevance=0.833, groundedness=0.525, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000058-3__dup4` (score=0.7196) - Blood pressure is the force of blood flow inside your blood vessels. Blood pressure is written with two numbers separated by a slash. For example, a blood pressure result of 130/80 is said as 130 over 80. The first nu...
2. `medquad_ans_0000036-11` (score=0.6827) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. There are two main types of high blood pressure: primary and secondary high blood pressure...
3. `medquad_ans_0000036-9` (score=0.6810) - Blood pressure is the force of blood pushing against the walls of the blood vessels as the heart pumps blood. If your blood pressure rises and stays high over time, its called high blood pressure. High blood pressure ...
4. `medquad_ans_0000471-1__dup1` (score=0.6726) - Blood pressure is the force of your blood pushing against the walls of your arteries. Each time your heart beats, it pumps blood into the arteries. Your blood pressure is highest when your heart beats, pumping the blo...
5. `medquad_ans_0000150-1__dup3` (score=0.6706) - Blood pressure is the force of blood pushing against blood vessel walls as the heart pumps out blood, and high blood pressure, also called hypertension, is an increase in the amount of force that blood places on blood...

**Answer**:

High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. There are two main types: primary (or essential) high blood pressure, which is the most common type and tends to develop over years as a person ages, and secondary high blood pressure, which is caused by another medical condition or the use of certain medicines and usually resolves after the cause is treated or removed. High blood pressure is dangerous because it makes the heart work too hard and the high force of blood flow can harm arteries and organs such as the heart, kidneys, brain, and eyes. It is typically defined as a blood pressure reading of 140/90 or higher, with prehypertension occurring between 120–139 for the top number or 80–89 for the bottom number. High blood pressure usually has no symptoms but can lead to serious problems such as stroke, heart

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Scores**: context_relevance=0.750, groundedness=0.047, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000139-2__dup2` (score=0.5443) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...
2. `medquad_ans_0000692-1__dup2` (score=0.5359) - Parkinson's disease (PD) is a type of movement disorder. It happens when nerve cells in the brain don't produce enough of a brain chemical called dopamine. Sometimes it is genetic, but most cases do not seem to run in...
3. `medquad_ans_0001065-4` (score=0.5341) - This condition is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In females (who have two X chromosomes), a mutation ...
4. `medquad_ans_0004758-4` (score=0.5257) - Is Partington syndrome inherited? Partington syndrome is inherited in an X-linked recessive manner. A condition is considered X-linked if the mutated gene that causes the condition is located on the X chromosome, one ...
5. `medquad_ans_0000307-1__dup1` (score=0.5219) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...

**Answer**:

The provided context does not contain information about X-linked dystonia-parkinsonism or its inheritance pattern. Therefore, based on the given context, it is not possible to determine whether X-linked dystonia-parkinsonism is inherited.

[No supporting source available]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source available`

### Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.520, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000120-1` (score=0.8285) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process keto...
2. `medquad_ans_0000702-1__dup1` (score=0.7735) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ket...
3. `medquad_ans_0000120-2` (score=0.7485) - Beta-ketothiolase deficiency appears to be very rare. It is estimated to affect fewer than 1 in 1 million newborns.
4. `medquad_ans_0000702-2__dup1` (score=0.7195) - What are the signs and symptoms of Beta ketothiolase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Beta ketothiolase deficiency. If the information is available, the ta...
5. `medquad_ans_0000120-5` (score=0.6894) - These resources address the diagnosis or management of beta-ketothiolase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of acetyl-CoA acetyltransferase These resources from MedlinePlus offer in...

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. Signs and symptoms typically appear between the ages of 6 months and 24 months and include episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, known as ketoacidotic attacks, may lead to coma and are frequently triggered by infections, periods without food (fasting), or increased intake of protein-rich foods. The condition is inherited in an autosomal recessive fashion and is caused by mutations in the ACAT1 gene. It is very rare, affecting fewer than 1 in 1 million newborns [medquad*ans_0000120-1, medquad_ans*

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 285 (`0000006_5-7`)

**Question**: What are the stages of Childhood Brain Stem Glioma ?

**Scores**: context_relevance=0.778, groundedness=0.954, answer_relevance=0.511

**Top docs**:

1. `medquad_ans_0000006_5-7` (score=0.7049) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the...
2. `medquad_ans_0000006_5-1` (score=0.6894) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - T...
3. `medquad_ans_0000006_5-4` (score=0.6403) - The signs and symptoms of brain stem glioma are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain. - The size of the tumor and whether it has spread all thro...
4. `medquad_ans_0000006_5-6` (score=0.6147) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis and treatment options depend on: - The type of brain stem glioma. - Where the tumor is found in the brain and if it has spread...
5. `medquad_ans_0002574-1` (score=0.6074) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...

**Answer**:

There is no standard staging system for childhood brain stem glioma [medquad_ans_0000006_5-7].

**Supported citations**: `medquad_ans_0000006_5-7`
**Unsupported citations**: _none_

### Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.533, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000266-1__dup2` (score=0.8150) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal c...
2. `medquad_ans_0000266-3__dup1` (score=0.7430) - With treatment, individuals with tethered spinal cord syndrome have a normal life expectancy. However, some neurological and motor impairments may not be fully correctable. Surgery soon after symptoms emerge appears t...
3. `medquad_ans_0000061-1__dup1` (score=0.5742) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to a...
4. `medquad_ans_0004257-2` (score=0.5616) - What are the signs and symptoms of myelomeningocele? A baby born with a myelomeningocele may have a sac sticking out of the mid to lower back that the doctor cannot see through when shining a light behind it. Symptoms...
5. `medquad_ans_0000257-1__dup2` (score=0.5605) - Spinal cord infarction is a stroke either within the spinal cord or the arteries that supply it. It is caused by arteriosclerosis or a thickening or closing of the major arteries to the spinal cord. Frequently spinal ...

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course. In children, symptoms may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is closely linked to spina bifida and results from improper growth of the neural tube during fetal development. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. Delayed symptom presentation is related to the degree of strain on the spinal cord over time and may

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.558, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000058-1__dup1` (score=0.7148) - Carpal tunnel syndrome (CTS) occurs when the median nerve, which runs from the forearm into the palm of the hand, becomes pressed or squeezed at the wrist. The carpal tunnel is a narrow, rigid passageway of ligament a...
2. `medquad_ans_0000058-3__dup1` (score=0.7138) - In general, carpal tunnel syndrome responds well to treatment, but less than half of individuals report their hand(s) feeling completely normal following surgery. Some residual numbness or weakness is common. At work,...
3. `medquad_ans_0000155-1__dup3` (score=0.6159) - You're working at your desk, trying to ignore the tingling or numbness you've had for some time in your hand and wrist. Suddenly, a sharp, piercing pain shoots through the wrist and up your arm. Just a passing cramp? ...
4. `medquad_ans_0000469-1` (score=0.6061) - Hereditary neuropathy with liability to pressure palsies is a disorder that affects peripheral nerves. These nerves connect the brain and spinal cord to muscles as well as sensory cells that detect touch, pain, and te...
5. `medquad_ans_0000970-1` (score=0.5738) - Tarsal-carpal coalition syndrome is a rare, inherited bone disorder that affects primarily the hands and feet. Several individual bones make up each wrist (carpal bones) and ankle (tarsal bones). In tarsal-carpal coal...

**Answer**:

The outlook for carpal tunnel syndrome is generally positive, as the condition responds well to treatment. However, less than half of individuals report their hand(s) feeling completely normal following surgery, and some residual numbness or weakness is common [medquad_ans_0000058-3__dup1]. Early diagnosis and treatment are important to prevent permanent nerve damage [medquad_ans_00000155-1__dup3]. With appropriate management, such as splints, rest, medications, or surgery, symptoms can be effectively managed, although full recovery is not guaranteed in all cases.

**Supported citations**: `medquad_ans_0000058-3__dup1`
**Unsupported citations**: `medquad_ans_00000155-1__dup3`

### Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.036, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0005077-3` (score=0.5900) - How is progressive bulbar palsy treated? Treatments aim to help people cope with the symptoms of progressive bulbar palsy, such as feeding tubes, devices to help with talking, and medicines to treat muscle spasms, wea...
2. `medquad_ans_0000188-2__dup1` (score=0.5821) - MJD is incurable, but some symptoms of the disease can be treated. For those individuals who show parkinsonian features, levodopa therapy can help for many years. Treatment with antispasmodic drugs, such as baclofen, ...
3. `medquad_ans_0006133-6` (score=0.5792) - Is there treatment for Townes-Brocks syndrome? Treatment is directed towards the specific symptoms, including immediate surgical intervention for imperforate anus; surgery for severe malformations of the hands; routin...
4. `medquad_ans_0000860-2__dup1` (score=0.5773) - How might Brown-Sequard syndrome be treated?
5. `medquad_ans_0003799-5` (score=0.5726) - How might Marden-Walker syndrome be treated? Very little information is available regarding the treatment of Marden-Walker syndrome. In general, treatment is symptomatic, with a multidisciplinary approach. The team of...

**Answer**:

The provided context does not contain any information about the treatments for Donnai-Barrow syndrome. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 289 (`0000811-1`)

**Question**: What is (are) porphyria ?

**Scores**: context_relevance=0.750, groundedness=0.537, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000811-1` (score=0.5887) - Porphyria is a group of disorders caused by abnormalities in the chemical steps that lead to heme production. Heme is a vital molecule for all of the body's organs, although it is most abundant in the blood, bone marr...
2. `medquad_ans_0000729-1__dup1` (score=0.5878) - Porphyrias are a group of genetic disorders caused by problems with how your body makes a substance called heme. Heme is found throughout the body, especially in your blood and bone marrow, where it carries oxygen. Th...
3. `medquad_ans_0001134-1` (score=0.5827) - Chester porphyria is a unique type of porphyria with the signs and symptoms of acute intermittent porphyria (AIP) and the biochemical defects of both AIP and variegate porphyria (VP). Chester porphyria does not confor...
4. `medquad_ans_0000189-4__dup2` (score=0.5726) - The exact rates of porphyria are unknown and vary around the world. For example, porphyria cutanea tarda is most common in the United States, and variegate porphyria is most common in South America.1
5. `medquad_ans_0000811-3` (score=0.5647) - Each form of porphyria results from mutations in one of these genes: ALAD, ALAS2, CPOX, FECH, HMBS, PPOX, UROD, or UROS. The genes related to porphyria provide instructions for making the enzymes needed to produce hem...

**Answer**:

Porphyria is a group of disorders caused by abnormalities in the chemical steps that lead to heme production. Heme is a vital molecule for all of the body's organs, although it is most abundant in the blood, bone marrow, and liver. Heme is a component of several iron-containing proteins called hemoproteins, including hemoglobin (the protein that carries oxygen in the blood). Researchers have identified several types of porphyria, which are distinguished by their genetic cause and their signs and symptoms. Some types of porphyria, called cutaneous porphyrias, primarily affect the skin, causing areas of skin exposed to the sun to become fragile and blistered, which can lead to infection, scarring, changes in skin coloring (pigmentation), and increased hair growth. Cutaneous porphyrias include congenital erythropoietic porphyria, erythropoietic protoporphyria, hepatoerythropoiet

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 290 (`0000585-1__dup1`)

**Question**: What is (are) Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy ?

**Scores**: context_relevance=0.917, groundedness=0.335, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000585-1__dup1` (score=0.9048) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
2. `medquad_ans_0004557-1` (score=0.7238) - Optic atrophy 1 is a condition that mainly affects vision, but may include other features. Vision loss typically begins within the first decade of life; severity varies widely among affected people (from nearly normal...
3. `medquad_ans_0005818-1` (score=0.5979) - Spondylocarpotarsal synostosis (SCT) syndrome is an inherited syndrome characterized by disproportionate short stature, abnormalities of the vertebrae in the spine, scoliosis and lordosis, carpal and tarsal fusion (fu...
4. `medquad_ans_0004797-1` (score=0.5920) - Pendred syndrome is a condition usually characterized by sensorineural hearing loss in both ears (bilateral) and euthyroid goiter (enlargement of the thyroid gland with normal thyroid gland function). The amount of he...
5. `medquad_ans_0001196-1` (score=0.5718) - Choroideremia is a genetic condition that causes vision loss. This disorder typically affects males. The first symptom is usually impairment of night vision (night blindness), which can occur in childhood. People with...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. Autosomal recessive inheritance has been suggested [medquad_ans_0000585-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000585-1`

### Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.042, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000167-2__dup3` (score=0.5310) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up...
2. `medquad_ans_0004167-3` (score=0.5089) - What causes Muir-Torre syndrome? Muir-Torre syndrome is a subtype of Lynch syndrome and may be caused by changes (mutations) in either the MLH1, MSH2, or MSH6 gene. These genes give the body instructions to make prote...
3. `medquad_ans_0002833-3` (score=0.5021) - What causes hereditary leiomyomatosis and renal cell cancer? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is caused by changes (mutations) in the FH gene. This gene gives the body instructions for making an...
4. `medquad_ans_0003635-1` (score=0.4969) - Lipoic acid synthetase deficiency is a rare condition that affects the mitochondria. Mitochondria are tiny structures found in almost every cell of the body. They are responsible for creating most of the energy necess...
5. `medquad_ans_0001047-3` (score=0.4904) - X-linked chondrodysplasia punctata 1 is caused by genetic changes involving the ARSE gene. This gene provides instructions for making an enzyme called arylsulfatase E. The function of this enzyme is unknown, although ...

**Answer**:

The provided context does not contain information about the genetic changes related to phosphoglycerate kinase deficiency. Therefore, a definitive answer cannot be provided based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Scores**: context_relevance=0.300, groundedness=0.000, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000673-4` (score=0.5113) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
2. `medquad_ans_0004784-4` (score=0.5096) - How is Pelizaeus-Merzbacher disease inherited?
3. `medquad_ans_0000928-4` (score=0.5069) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000498-4` (score=0.5069) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000705-4` (score=0.5069) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The provided context does not contain any information about the genetic changes related to pulmonary veno-occlusive disease. Therefore, the question cannot be answered based on the given context. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Scores**: context_relevance=0.750, groundedness=0.961, answer_relevance=0.450

**Top docs**:

1. `medquad_ans_0001006-3` (score=0.5288) - Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. In the liver, enzymes break down tyrosine in a five step process, resulting in molecules that are either excreted by t...
2. `medquad_ans_0000189-5__dup1` (score=0.5092) - Most porphyrias are inherited disorders. Scientists have identified genes for all eight enzymes in the heme biosynthetic pathway. Most porphyrias result from inheriting an abnormal gene, also called a gene mutation, f...
3. `medquad_ans_0004086-1` (score=0.4711) - Mitochondrial genetic disorders refer to a group of conditions that affect the mitochondria (the structures in each cell of the body that are responsible for making energy). People with these conditions can present at...
4. `medquad_ans_0000396-1__dup2` (score=0.4693) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...
5. `medquad_ans_0000133-3__dup2` (score=0.4666) - Wilson disease is caused by an inherited autosomal recessive mutation, or change, in the ATP7B gene. In an autosomal recessive disease, the child has to inherit the gene mutation from both parents to have an increased...

**Answer**:

Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. These mutations result in a decrease in the activity of one of the enzymes involved in the breakdown of tyrosine, leading to the accumulation of toxic levels of tyrosine and its byproducts, which can damage cells in the liver, kidneys, nervous system, and other organs [medquad_ans_0001006-3].

**Supported citations**: `medquad_ans_0001006-3`
**Unsupported citations**: _none_

### Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Scores**: context_relevance=0.857, groundedness=0.465, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000436-5` (score=0.6597) - These resources address the diagnosis or management of Graves disease: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Graves disease 2 - Genetic Testing Registry: Graves disease 3 -...
2. `medquad_ans_0000004-6__dup2` (score=0.6585) - People with Graves disease have three treatment options: radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves disease in the United States. Graves disease ...
3. `medquad_ans_0000004-2__dup3` (score=0.6027) - People with Graves disease may have common symptoms of hyperthyroidism such as - nervousness or irritability - fatigue or muscle weakness - heat intolerance - trouble sleeping - hand tremors - rapid and irregular hear...
4. `medquad_ans_0000004-3__dup4` (score=0.6021) - Graves ophthalmopathy is a condition associated with Graves disease that occurs when cells from the immune system attack the muscles and other tissues around the eyes. The result is inflammation and a buildup of tissu...
5. `medquad_ans_0000004-9` (score=0.5858) - - Graves disease is the most common cause of hyperthyroidism in the United States. - In Graves disease, the immune system stimulates the thyroid gland to make too much thyroid hormone. - Common symptoms of hyperthyroi...

**Answer**:

The treatments for Graves' disease include radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves' disease in the United States. In this therapy, patients take radioactive iodine-131 by mouth, which is collected by the thyroid gland and gradually destroys thyroid cells, reducing hormone production. The dose may be adjusted to either shut down the thyroid completely or bring hormone production into the normal range. Multiple rounds may be needed, and improvements in symptoms may take several weeks or months. Radioiodine therapy is not used in pregnant women or women who are breastfeeding due to potential harm to the fetus or passage to the infant through breast milk. Experts recommend waiting at least one year after treatment before becoming pregnant. Almost everyone who receives radioiodine therapy eventually develops hypothyroidism. Alternative treatments, such as prescription steroids, may help prevent complications like Graves' ophthalmopathy (GO) in some cases [med

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Scores**: context_relevance=0.800, groundedness=0.467, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000039-1__dup2` (score=0.7279) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, ...
2. `medquad_ans_0000039-2__dup1` (score=0.6514) - As yet, no specific treatment is known that can halt or reverse the symptoms of Batten disease. However, seizures can sometimes be reduced or controlled with anticonvulsant drugs, and other medical problems can be tre...
3. `medquad_ans_0000039-3__dup1` (score=0.6369) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten dise...
4. `medquad_ans_0000545-2` (score=0.6196) - Juvenile Batten disease is the most common type of NCL, but its exact prevalence is unknown. Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, wher...
5. `medquad_ans_0000545-1` (score=0.6149) - Juvenile Batten disease is an inherited disorder that primarily affects the nervous system. After a few years of normal development, children with this condition develop progressive vision loss, intellectual and motor...

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. It is characterized by a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. Early signs may include personality and behavior changes, slow learning, clumsiness, or stumbling. Vision loss is often an early sign and may lead to suspicion during an eye exam. The disease progresses over time, leading to cognitive impairment, worsening seizures, progressive loss of sight and motor skills, and eventually blindness, bedridden state, and dementia. Juvenile Batten disease, the most common type of neuronal ceroid lipofuscinosis (NCL), typically presents with progressive vision loss starting between ages 4 and 8, followed by developmental regression, intellectual decline, speech difficulties, and loss of motor skills such as walking or sitting. It is often fatal by the late teens or twenties [medquad_ans_000

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Scores**: context_relevance=0.714, groundedness=0.640, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000318-1__dup1` (score=0.6902) - Ehlers-Danlos syndrome (EDS) is a group of inherited disorders that weaken connective tissues. Connective tissues are proteins that support skin, bones, blood vessels, and other organs. EDS usually affects your skin, ...
2. `medquad_ans_0005116-1` (score=0.5480) - Proteus-like syndrome describes people who do not meet the diagnostic criteria for Proteus syndrome but who share many of the characteristic signs and symptoms associated with the condition. Affected people may experi...
3. `medquad_ans_0004784-4` (score=0.5422) - How is Pelizaeus-Merzbacher disease inherited?
4. `medquad_ans_0000673-4` (score=0.4895) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
5. `medquad_ans_0000886-1__dup1` (score=0.4766) - C syndrome, also known as Opitz trigonocephaly syndrome, is characterized by trigonocephaly, severe intellectual disability, hypotonia, variable cardiac defects, redundant (extra folds of) skin, joint and limb abnorma...

**Answer**:

The provided context does not contain information about Ehlers-Danlos syndrome, progeroid type. Therefore, a specific answer cannot be generated based on the given context. [medquad_ans_0000318-1__dup1] discusses general Ehlers-Danlos syndrome but does not mention the progeroid type.

**Supported citations**: `medquad_ans_0000318-1__dup1`
**Unsupported citations**: _none_

### Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.520

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.7765) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
2. `medquad_ans_0000416-3` (score=0.6894) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
3. `medquad_ans_0002601-1` (score=0.6813) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...
4. `medquad_ans_0000416-5` (score=0.6718) - These resources address the diagnosis or management of glutamate formiminotransferase deficiency: - Baby's First Test - Genetic Testing Registry: Glutamate formiminotransferase deficiency These resources from MedlineP...
5. `medquad_ans_0002601-2` (score=0.6615) - What are the signs and symptoms of Glutamate formiminotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Glutamate formiminotransferase deficiency. If the inform...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the last two steps of the breakdown of the amino acid histidine and in the production of one of several forms of the vitamin folate [medquad_ans_0000416-3].

**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

### Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Scores**: context_relevance=1.000, groundedness=0.937, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0001047-2` (score=0.8128) - The prevalence of X-linked chondrodysplasia punctata 1 is unknown. Several dozen affected males have been reported in the scientific literature.
2. `medquad_ans_0001169-1` (score=0.7567) - Chondrodysplasia punctata 1, X-linked recessive (CDPX1) is a genetic disorder present from birth that affects bone and cartilage development. On x-ray, affected infants have characteristic spots at the ends of their b...
3. `medquad_ans_0001048-2` (score=0.7483) - X-linked chondrodysplasia punctata 2 has been estimated to affect fewer than 1 in 400,000 newborns. However, the disorder may actually be more common than this estimate because it is likely underdiagnosed, particularl...
4. `medquad_ans_0001047-1` (score=0.7370) - X-linked chondrodysplasia punctata 1 is a disorder of cartilage and bone development that occurs almost exclusively in males. Chondrodysplasia punctata is an abnormality that appears on x-rays as spots (stippling) nea...
5. `medquad_ans_0001169-2` (score=0.7352) - What are the signs and symptoms of Chondrodysplasia punctata 1, X-linked recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 1, X-linked recessive. I...

**Answer**:

Symptoms of Chondrodysplasia punctata 1, X-linked recessive (CDPX1) include characteristic spots (stippled epiphyses) on x-rays at the ends of bones, which typically disappear between ages 2 and 3 [medquad_ans_0001169-1]. Other common features are shortened fingers, a flat nose, breathing abnormalities, hearing loss, abnormalities of the spinal bones in the neck, and delayed intellectual development [medquad_ans_0001169-1]. Additionally, affected individuals may have short stature, unusually short fingertips and ends of toes, and distinctive facial features such as a flattened nose with crescent-shaped nostrils and a flat nasal bridge [medquad_ans_0001047-1]. Some individuals may experience serious or life-threatening complications, including airway stenosis that restricts breathing and spinal abnormalities leading to nerve compression

**Supported citations**: `medquad_ans_0001169-1`, `medquad_ans_0001047-1`
**Unsupported citations**: _none_

### Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Scores**: context_relevance=0.714, groundedness=0.980, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000165-2__dup1` (score=0.8085) - The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfa...
2. `medquad_ans_0000132-9` (score=0.7615) - Experts have not yet found a way to prevent Whipple disease.
3. `medquad_ans_0000132-10` (score=0.6805) - A person with Whipple disease and malabsorption may need - a diet high in calories and protein - vitamins - nutritional supplements
4. `medquad_ans_0000132-6` (score=0.6486) - People with Whipple disease may have complications caused by malnutrition, which is due to damaged villi in the small intestine. As a result of delayed diagnosis or treatment, people may experience the following compl...
5. `medquad_ans_0000132-1__dup3` (score=0.6323) - Whipple disease is a rare bacterial infection that primarily affects the small intestine. The infection may spread to any organ in the body; however, it more commonly affects the - joints - central nervous system, whi...

**Answer**:

The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfadizine or solfamethoxazole can treat neurological symptoms. Relapsing neurologic Whipple's disease, marked by bouts of worsening of symptoms, is sometimes treated with a combination of antibiotics and weekly injections of interferon gamma, a substance made by the body that activates the immune system [medquad_ans_0000165-2__dup1].

**Supported citations**: `medquad_ans_0000165-2__dup1`
**Unsupported citations**: _none_

### Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Scores**: context_relevance=0.857, groundedness=0.973, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000676-4__dup1` (score=0.9124) - How might Bell's palsy be treated?
2. `medquad_ans_0000676-3__dup1` (score=0.8100) - What causes Bell's palsy?
3. `medquad_ans_0000676-2__dup1` (score=0.8035) - What are the symptoms of Bell's palsy?
4. `medquad_ans_0000042-2__dup2` (score=0.6745) - Steroids such as prednisone -- used to reduce inflammation and swelling -- are an effective treatment for Bell's palsy. Antiviral drugs may have some benefit in shortening the course of the disease. Analgesics such as...
5. `medquad_ans_0000042-3__dup2` (score=0.6719) - The prognosis for individuals with Bell's palsy is generally very good. The extent of nerve damage determines the extent of recovery. With or without treatment, most individuals begin to get better within 2 weeks afte...

**Answer**:

Steroids such as prednisone, which reduce inflammation and swelling, are an effective treatment for Bell's palsy. Antiviral drugs may have some benefit in shortening the course of the disease. Analgesics such as aspirin, acetaminophen, or ibuprofen may relieve pain. Individuals should always consult their doctors before taking over-the-counter medicines due to possible drug interactions. Keeping the eye moist and protecting it from debris and injury, especially at night, is important; lubricating eye drops can help. Other therapies such as physical therapy, facial massage, or acupuncture may provide a potential small improvement in facial nerve function and pain [medquad_ans_0000042-2__dup2].

**Supported citations**: `medquad_ans_0000042-2__dup2`
**Unsupported citations**: _none_
