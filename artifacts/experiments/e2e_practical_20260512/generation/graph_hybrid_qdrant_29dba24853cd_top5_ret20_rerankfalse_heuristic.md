# Generation Evaluation

## Summary

- `avg_answer_chars`: 570.1067
- `avg_answer_relevance`: 0.7749
- `avg_citation_presence_rate`: 0.8267
- `avg_context_relevance`: 0.9196
- `avg_empty_answer`: 0.0000
- `avg_end_to_end_latency_ms`: 7238.1967
- `avg_generation_latency_ms`: 6905.2253
- `avg_groundedness`: 0.7924
- `avg_insufficient_context`: 0.0133
- `avg_num_retrieved_docs`: 5.0000
- `avg_rerank_latency_ms`: 0.0000
- `avg_retrieval_latency_ms`: 332.7300
- `avg_supported_citation_rate`: 0.7650
- `avg_unsupported_citation_rate`: 0.0617
- `generation_remote_concurrency`: 4.0000
- `num_examples`: 300
- `reranker_enabled_rate`: 0.0000

## Examples

### Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006511-1` (score=0.7000) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the informatio...
2. `medquad_ans_0006510-1` (score=0.6904) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the informatio...
3. `medquad_ans_0006509-2` (score=0.6615) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is...
4. `medquad_ans_0001059-5` (score=0.0842) - These resources address the diagnosis or management of XLP: - Children's Hospital of Philadelphia - Gene Review: Gene Review: Lymphoproliferative Disease, X-Linked - Genetic Testing Registry: Lymphoproliferative syndr...
5. `medquad_ans_0002834-2` (score=0.0652) - What are the signs and symptoms of Hereditary lymphedema type II? Hereditary lymphedema type II is characterized by the abnormal transport of lymph fluid. This causes the lymph fluid to build up, causing swelling (lym...

**Answer**:

The symptoms of X-linked lymphoproliferative syndrome 1 include cellular immunodeficiency (90%), decreased antibody level in blood (50%), hepatomegaly (50%), lymphadenopathy (50%), lymphoma (50%), splenomegaly (50%), anemia (7.5%), encephalitis, fulminant hepatitis, hepatic encephalopathy, IgG deficiency, immunodeficiency, increased IgM level, meningitis, pancytopenia, recurrent pharyngitis, reduced natural killer cell activity, and thrombocytopenia [medquad_ans_0006510-1].

**Supported citations**: `medquad_ans_0006510-1`
**Unsupported citations**: _none_

### Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.977, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000312-1` (score=0.7000) - Early-onset primary dystonia is a condition characterized by progressive problems with movement, typically beginning in childhood. Dystonia is a movement disorder that involves involuntary tensing of the muscles (musc...
2. `medquad_ans_0000307-1__dup1` (score=0.7000) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...
3. `medquad_ans_0000312-4` (score=0.6122) - Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, which means one of the two copies of the gene is altered in each cell. Many people who have a mutation in this gene are not affected by the d...
4. `medquad_ans_0004750-4` (score=0.6122) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is pass...
5. `medquad_ans_0000312-5` (score=0.5272) - These resources address the diagnosis or management of early-onset primary dystonia: - Gene Review: Gene Review: DYT1 Early-Onset Primary Dystonia - Genetic Testing Registry: Dystonia 1 - MedlinePlus Encyclopedia: Mov...

**Answer**:

Yes, early-onset primary dystonia can be inherited. Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, which means one of the two copies of the gene is altered in each cell. Only 30 to 40 percent of people who inherit a TOR1A mutation will ever develop signs and symptoms of early-onset primary dystonia [medquad_ans_0000312-4]. Everyone who has been diagnosed with early-onset primary dystonia has inherited a mutation in this gene [medquad_ans_0000312-4].

**Supported citations**: `medquad_ans_0000312-4`
**Unsupported citations**: _none_

### Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Scores**: context_relevance=0.857, groundedness=0.691, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000689-1__dup2` (score=0.7000) - Parasites are living things that use other living things - like your body - for food and a place to live. You can get them from contaminated food or water, a bug bite, or sexual contact. Some parasitic diseases are ea...
2. `medquad_ans_0000038-2__dup3` (score=0.6141) - People can get infected with Babesia parasites in several ways: - The main way is through the bite of an infected tick—during outdoor activities in areas where babesiosis is found (see below). - A less common way is b...
3. `medquad_ans_0000660-1__dup1` (score=0.5648) - Baylisascaris roundworms are intestinal parasites found in many different animals. Baylisascaris infection in humans is uncommon but can be severe. While Baylisascaris can infect different types of animals, Baylisasca...
4. `medquad_ans_0000344-1__dup2` (score=0.4820) - Baylisascaris worms are intestinal parasites found in a wide variety of animals. Different species of Baylisascaris are associated with different animal hosts. For example, Baylisascaris procyonis is found in raccoons...
5. `medquad_ans_0000265-5__dup1` (score=0.3748) - Loa loa parasites are found in West and Central Africa. Ten countries have areas where there are high rates of infection (i.e., where more than 40% of the people who live in that area report that they have had eye wor...

**Answer**:

The provided context does not mention taeniasis or who is at risk for this specific parasitic infection. Therefore, based on the given information, it is not possible to answer the question. [medquad_ans_0000689-1__dup2] discusses general parasitic diseases and transmission routes but does not address taeniasis specifically. [medquad_ans_0000038-2__dup3] and [medquad_ans_0000660-1__dup1] discuss Babesia and Baylisascaris, respectively, but neither covers taeniasis.

Answer: Insufficient context to determine who is at risk for parasites - taeniasis.

**Supported citations**: `medquad_ans_0000689-1__dup2`, `medquad_ans_0000038-2__dup3`, `medquad_ans_0000660-1__dup1`
**Unsupported citations**: _none_

### Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.916, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000194-2__dup1` (score=0.7000) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
2. `medquad_ans_0000029-1` (score=0.6643) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. Howeve...
3. `medquad_ans_0000197-1__dup1` (score=0.6307) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000193-2__dup1` (score=0.6193) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 1. If the information is availab...
5. `medquad_ans_0000196-2__dup1` (score=0.6113) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 4. If the information is availab...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy [medquad_ans_0000194-2__dup1]. These symptoms are based on the Human Phenotype Ontology, which compiles data from sources such as Orphanet, and represent approximate frequencies observed in patients with this condition [medquad_ans_0000194-2__dup1].

**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

### Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003421-1` (score=0.7000) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0005502-2` (score=0.0765) - What are the signs and symptoms of Schnitzler syndrome? The signs and symptoms of Schnitzler syndrome vary but may include: Red raised patches of skin (urticaria) that may become itchy Recurrent fevers Join pain and i...
3. `medquad_ans_0000108-2__dup1` (score=0.0765) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
4. `medquad_ans_0003339-2` (score=0.0669) - What are the signs and symptoms of Kallmann syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kallmann syndrome. If the information is available, the table below includes how...
5. `medquad_ans_0005488-1` (score=0.0669) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

### Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.902, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000069-1__dup3` (score=0.7000) - Espaol Hemophilia (heem-o-FILL-ee-ah) is a rare bleeding disorder in which the blood doesn't clot normally. If you have hemophilia, you may bleed for a longer time than others after an injury. You also may bleed insid...
2. `medquad_ans_0000069-3__dup1` (score=0.6538) - A defect in one of the genes that determines how the body makes blood clotting factor VIII or IX causes hemophilia. These genes are located on the X chromosomes (KRO-muh-somz). Chromosomes come in pairs. Females have ...
3. `medquad_ans_0000455-1__dup1` (score=0.6276) - Hemophilia is a rare disorder in which the blood does not clot normally. It is usually inherited. Hemophilia usually occurs in males. If you have hemophilia, you have little or no clotting factor. Clotting factor is a...
4. `medquad_ans_0000069-5__dup1` (score=0.3737) - If you or your child appears to have a bleeding problem, your doctor will ask about your personal and family medical histories. This will reveal whether you or your family members, including women and girls, have blee...
5. `medquad_ans_0002788-1` (score=0.0954) - Hemophilia B is a bleeding disorder that slows the blood clotting process. People with this disorder experience prolonged bleeding or oozing following an injury or surgery. In severe cases of hemophilia, heavy bleedin...

**Answer**:

Yes, hemophilia is inherited. It is usually inherited through genes located on the X chromosomes, and individuals are born with the disorder due to a defect in the genes that determine how the body makes blood clotting factor VIII or IX [medquad_ans_0000069-1__dup3]. Males are more commonly affected because they have only one X chromosome, so if that X chromosome carries the hemophilia gene, they will have the disorder. Females with the gene on one X chromosome are typically carriers and can pass the gene to their children [medquad_ans_0000069-1__dup3].

**Supported citations**: `medquad_ans_0000069-1__dup3`
**Unsupported citations**: _none_

### Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.7000) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often t...
2. `medquad_ans_0003103-2` (score=0.0765) - What are the signs and symptoms of IBIDS syndrome? The most common symptoms of Tay syndrome are brittle hair (trichothiodystrophy); dry, thickened, scaling skin (ichthyosis); photosensitivity (abnormal light sensitivi...
3. `medquad_ans_0000108-2__dup1` (score=0.0765) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
4. `medquad_ans_0004163-2` (score=0.0669) - What are the signs and symptoms of Mucopolysaccharidosis type IVA? The Human Phenotype Ontology provides the following list of signs and symptoms for Mucopolysaccharidosis type IVA. If the information is available, th...
5. `medquad_ans_0005488-1` (score=0.0669) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...

**Answer**:

The signs and symptoms of IVIC syndrome include:

- Hearing impairment (90%)
- Limitation of joint mobility (90%)
- Short stature (90%)
- Strabismus (90%)
- Abnormal dermatoglyphics (50%)
- Aplasia/hypoplasia of the thumb (50%)
- Radioulnar synostosis (50%)
- Scoliosis (50%)
- Synostosis of carpal bones (50%)
- Triphalangeal thumb (50%)
- Abnormality of the clavicle (7.5%)
- Arrhythmia (7.5%)
- Leukocytosis (7.5%)
- Preaxial hand polydactyly (7.5%)
- Thrombocytopenia (7.5%)
- U

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Scores**: context_relevance=0.833, groundedness=0.520, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000775-1` (score=0.7000) - Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. The disorder affects red blood cells (erythrocytes), which carry oxygen; white bloo...
2. `medquad_ans_0000775-2` (score=0.6905) - Paroxysmal nocturnal hemoglobinuria is a rare disorder, estimated to affect between 1 and 5 per million people.
3. `medquad_ans_0000775-5` (score=0.6806) - These resources address the diagnosis or management of paroxysmal nocturnal hemoglobinuria: - Duke University School of Medicine: Hemostasis & Thrombosis Center - Genetic Testing Registry: Paroxysmal nocturnal hemoglo...
4. `medquad_ans_0004747-1` (score=0.6351) - Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It can occur at any age, but is usually diagnosed in young adulthood. People ...
5. `medquad_ans_0000775-3` (score=0.6106) - Mutations in the PIGA gene cause paroxysmal nocturnal hemoglobinuria. The PIGA gene provides instructions for making a protein called phosphatidylinositol glycan class A. This protein takes part in a series of steps t...

**Answer**:

Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells, affecting red blood cells (erythrocytes), white blood cells (leukocytes), and platelets (thrombocytes). It is characterized by sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to hemoglobinuria—dark-colored urine due to the presence of hemoglobin. This condition can cause hemolytic anemia, resulting in symptoms like fatigue, weakness, pallor, shortness of breath, and increased heart rate. Individuals are also at risk for infections due to white blood cell deficiency, abnormal blood clotting (thrombosis), especially in large abdominal veins, or severe bleeding (hemorrhage). Additionally, people with this disorder are

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.980, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000037_3-1` (score=0.7000) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...
2. `medquad_ans_0000037_4-1` (score=0.6425) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...
3. `medquad_ans_0000010-1__dup5` (score=0.6425) - How Tumors Form The body is made up of many types of cells. Normally, cells grow, divide, and produce more cells as needed to keep the body healthy and functioning properly. Sometimes, however, the process goes wrong ...
4. `medquad_ans_0000010-10` (score=0.6109) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the li...
5. `medquad_ans_0000221-1__dup2` (score=0.5359) - The colon and rectum are part of the large intestine. Colorectal cancer occurs when tumors form in the lining of the large intestine. It is common in both men and women. The risk of developing colorectal cancer rises ...

**Answer**:

Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. Cancer that begins in the colon is called colon cancer, and cancer that begins in the rectum is called rectal cancer. Cancer that affects either of these organs may also be called colorectal cancer [medquad_ans_0000037_3-1].

**Supported citations**: `medquad_ans_0000037_3-1`
**Unsupported citations**: _none_

### Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Scores**: context_relevance=0.833, groundedness=0.550, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000544-5` (score=0.7000) - These resources address the diagnosis or management of junctional epidermolysis bullosa: - Epidermolysis Bullosa Center, Cincinnati Children's Hospital Medical Center - Gene Review: Gene Review: Junctional Epidermolys...
2. `medquad_ans_0000544-2` (score=0.6804) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.
3. `medquad_ans_0002121-1` (score=0.6602) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main ty...
4. `medquad_ans_0003317-1` (score=0.5482) - Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz ...
5. `medquad_ans_0000544-3` (score=0.3690) - Junctional epidermolysis bullosa results from mutations in the LAMA3, LAMB3, LAMC2, and COL17A1 genes. Mutations in each of these genes can cause Herlitz JEB or non-Herlitz JEB. LAMB3 gene mutations are the most commo...

**Answer**:

Junctional epidermolysis bullosa (JEB) is a type of epidermolysis bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz type and the Non-Herlitz type. The Herlitz type is very severe and individuals with this condition often do not survive infancy. The Non-Herlitz type includes several subtypes that cause mild to severe blistering of the skin present at birth or shortly thereafter. JEB is inherited in an autosomal recessive pattern and is caused by mutations in the LAMB3, COL17A1, or LAMC2, and LAMA3 genes. Mutations in each of these genes can cause Herlitz JEB or non-Herlitz JEB, with LAMB3 gene mutations being the most common, causing about 70

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.943, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000668-3` (score=0.7000) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same ...
2. `medquad_ans_0000668-2` (score=0.6811) - Molybdenum cofactor deficiency is a rare condition that is estimated to occur in 1 in 100,000 to 200,000 newborns worldwide. More than 100 cases have been reported in the medical literature, although it is thought tha...
3. `medquad_ans_0000668-5` (score=0.6804) - These resources address the diagnosis or management of molybdenum cofactor deficiency: - Genetic Testing Registry: Combined molybdoflavoprotein enzyme deficiency - Genetic Testing Registry: Molybdenum cofactor deficie...
4. `medquad_ans_0000668-1` (score=0.6615) - Molybdenum cofactor deficiency is a rare condition characterized by brain dysfunction (encephalopathy) that worsens over time. Babies with this condition appear normal at birth, but within a week they have difficulty ...
5. `medquad_ans_0000651-3` (score=0.2705) - Methylmalonic acidemia with homocystinuria can be caused by mutations in one of several genes: MMACHC, MMADHC, LMBRD1, ABCD4, or HCFC1. Mutations in these genes account for the different types of the disorder, which a...

**Answer**:

The genetic changes related to molybdenum cofactor deficiency are mutations in the MOCS1, MOCS2, or GPHN gene. Mutations in the MOCS1 gene cause type A, mutations in the MOCS2 gene cause type B, and mutations in the GPHN gene cause type C [medquad_ans_0000668-3]. These mutations reduce or eliminate the function of the associated protein, impairing molybdenum cofactor biosynthesis [medquad_ans_0000668-3].

**Supported citations**: `medquad_ans_0000668-3`
**Unsupported citations**: _none_

### Example 12 (`0000756-1`)

**Question**: What is (are) ornithine translocase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000756-2` (score=0.7000) - Ornithine translocase deficiency is a very rare disorder. Fewer than 100 affected individuals have been reported worldwide.
2. `medquad_ans_0000756-1` (score=0.6402) - Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The...
3. `medquad_ans_0000755-1` (score=0.6262) - Ornithine transcarbamylase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high...
4. `medquad_ans_0004581-3` (score=0.6175) - What causes ornithine transcarbamylase (OTC) deficiency? Ornithine transcarbamylase (OTC) deficiency is caused by mutations in the OTC gene. OTC deficiency belongs to a class of genetic diseases called urea cycle diso...
5. `medquad_ans_0004582-1` (score=0.5962) - What are the signs and symptoms of Ornithine translocase deficiency syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Ornithine translocase deficiency syndrome. If the inform...

**Answer**:

Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The nervous system is especially sensitive to the effects of excess ammonia. Ornithine translocase deficiency varies widely in its severity and age of onset. An infant with ornithine translocase deficiency may be lacking in energy (lethargic) or refuse to eat, or have poorly controlled breathing or body temperature. Some babies with this disorder may experience seizures or unusual body movements, or go into a coma. Episodes of illness may coincide with the introduction of high-protein formulas or solid foods into the diet. In most affected individuals, signs and symptoms of ornithine translocase deficiency do not appear until later in life. Later-onset forms of ornithine translocase deficiency are usually less severe than the

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Scores**: context_relevance=0.857, groundedness=0.966, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000362-1__dup2` (score=0.7000) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
2. `medquad_ans_0000045-28` (score=0.6709) - The general number for Medicare is 1-800-Medicare (1-800-633-4227). TTY users should call 1-877-486-2048. You can also visit http://www.medicare.gov. The "Medicare & You" handbook is mailed out to all Medicare enrolle...
3. `medquad_ans_0000036-8` (score=0.6358) - Medicaid is a state health insurance program for those with low incomes and few assets. Each state runs its own program. The Federal Government requires that Medicaid programs cover a specific set of services; however...
4. `medquad_ans_0000036-16` (score=0.6253) - - Diabetes management and treatment is expensive. Many people who have diabetes need help paying for their care. For those who qualify, a variety of government and nongovernment programs can help cover health care exp...
5. `medquad_ans_0000042-18` (score=0.5717) - Many agencies and organizations in the community provide assistance and information to people who have low vision and to their families and caregivers. State agencies for the blind and visually impaired can make refer...

**Answer**:

Yes, information about financial assistance is available. If you don't have health insurance or need help with costs that aren't covered, financial assistance might be available through certain government programs and nonprofit organizations. You can also discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2].

**Supported citations**: `medquad_ans_0000362-1__dup2`
**Unsupported citations**: _none_

### Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0004105-1` (score=0.7000) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About hal...
2. `medquad_ans_0004105-2` (score=0.7000) - What are the symptoms of Mollaret meningitis? The symptoms of Mollaret meningitis are the same as those found in other types of meningitis. In Mollaret meningitis, however, the symptoms are recurring and are often acc...
3. `medquad_ans_0000115-1__dup1` (score=0.6619) - Meningitis is an infection of the meninges, the membranes that surround the brain and spinal cord. Encephalitis is inflammation of the brain itself. Causes of encephalitis and meningitis include viruses, bacteria, fun...
4. `medquad_ans_0000598-1__dup2` (score=0.6619) - Meningitis is inflammation of the thin tissue that surrounds the brain and spinal cord, called the meninges. There are several types of meningitis. The most common is viral meningitis, which you get when a virus enter...
5. `medquad_ans_0000115-2__dup1` (score=0.6009) - Anyone experiencing symptoms of meningitis or encephalitis should see a doctor immediately. Antibiotics for most types of meningitis can greatly reduce the risk of dying from the disease. Antiviral medications may be ...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches, separated by weeks or months of no symptoms. About half of affected individuals may also experience long-term abnormalities of the nervous system that come and go, such as seizures, double vision, abnormal reflexes, paralysis of a cranial nerve (paresis), hallucinations, or coma. The exact cause remains unknown, but recent data suggests that herpes simplex virus (HSV-2 and, less frequently, HSV-1) may cause some, if not most cases. Other potential causes include trauma and viral infections other than herpes simplex [medquad_ans_0004105-1]. The symptoms of Mollaret meningitis are similar to those of other types of meningitis and include high fever, severe headache, nausea, vomiting, stiff neck, photophobia (s

**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

### Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.801, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003331-3` (score=0.7000) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...
2. `medquad_ans_0003319-1` (score=0.6995) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. U...
3. `medquad_ans_0003331-1` (score=0.6553) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
4. `medquad_ans_0003319-2` (score=0.5882) - What are the signs and symptoms of juvenile amyotrophic lateral sclerosis? Signs and symptoms of juvenile ALS vary but include slowly to very slowly progressive muscle weakness, increased muscle tone, Babinski reflex,...
5. `medquad_ans_0000551-2` (score=0.5577) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.

**Answer**:

The provided context does not specify the cause of juvenile amyotrophic lateral sclerosis (ALS). While it states that juvenile ALS is often genetic and may be inherited in an autosomal dominant or autosomal recessive fashion, it does not identify specific genes or mutations responsible for the condition [medquad_ans_0003319-1]. Therefore, based on the available information, the exact cause of juvenile ALS cannot be determined.

**Supported citations**: `medquad_ans_0003319-1`
**Unsupported citations**: _none_

### Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Scores**: context_relevance=0.889, groundedness=0.859, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0000219-4__dup3` (score=0.7000) - Erectile dysfunction affects men of all races and in all parts of the world. Men are more likely to have ED as they get older. For example, ED occurs in - about 12 percent of men younger than 60 - 22 percent of men ag...
2. `medquad_ans_0000219-8` (score=0.6336) - Your doctor can offer you a number of treatments for ED. For many men, the answer is as simple as taking a pill. Other men have to try two or three options before they find a treatment that works for them. Dont give u...
3. `medquad_ans_0000219-5__dup2` (score=0.6317) - Having ED can cause you to feel depressed or anxious. ED may also cause low self-esteem. When you have ED, you may not have a satisfying sex life. You may not feel as close with your sexual partner, which may strain y...
4. `medquad_ans_0000219-11` (score=0.6180) - - Erectile dysfunction (ED) is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last...
5. `medquad_ans_0000207-12` (score=0.5447) - - Peyronies disease is a disorder in which scar tissue, called a plaque, forms in the penisthe male organ used for urination and sex. - Medical experts do not know the exact cause of Peyronies disease. Many believe th...

**Answer**:

Erectile dysfunction (ED) affects men of all races and ages globally, with prevalence increasing with age: about 12% of men younger than 60, 22% of men aged 60 to 69, and 30% of men age 70 or older [medquad_ans_0000219-4__dup3].

Treatment options include addressing the underlying cause of ED. This may involve lifestyle changes such as quitting smoking, reducing alcohol intake, increasing physical activity, and avoiding illegal drug use [medquad_ans_0000219-8]. Your doctor may also adjust medications you are taking for other health conditions, as some medications can contribute to ED [medquad_ans_0000219-8]. Counseling can help couples manage the emotional impact of ED and strengthen their relationship [medquad_ans_0000219

**Supported citations**: `medquad_ans_0000219-4__dup3`, `medquad_ans_0000219-8`
**Unsupported citations**: _none_

### Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.946, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0006267-3` (score=0.7000) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
2. `medquad_ans_0000202-5` (score=0.5877) - These resources address the diagnosis or management of collagen VI-related myopathy: - Gene Review: Gene Review: Collagen Type VI-Related Disorders - Genetic Testing Registry: Bethlem myopathy - Genetic Testing Regist...
3. `medquad_ans_0000576-5` (score=0.5409) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...
4. `medquad_ans_0000631-1__dup1` (score=0.5247) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...
5. `medquad_ans_0003601-1` (score=0.4861) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...

**Answer**:

Physical therapy, including early mobilization, regular stretching, and splinting, is the main focus of supportive care for Ullrich congenital muscular dystrophy. Respiratory support and night-time ventilation may become necessary in the first or second decade of life. Prevention of chest infections can be achieved with the use of antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that cyclosporin A may benefit individuals with Ullrich congenital muscular dystrophy, although more studies are needed to confirm its efficacy [medquad_ans_0006267-3].

**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

### Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004006-1` (score=0.7000) - Microcephalic osteodysplastic primordial dwarfism type 2 (MOPD2) is a condition characterized by short stature (dwarfism), skeletal abnormalities and an unusually small head size (microcephaly). Other signs and sympto...
2. `medquad_ans_0004005-5` (score=0.7000) - How might microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) be treated? At this time there are no specific treatments for MOPD1. Treatment is generally supportive. The prognosis is poor for affected ind...
3. `medquad_ans_0004005-1` (score=0.6825) - Microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) is a genetic condition that is mainly characterized by intrauterine and post-natal growth retardation; an abnormally small head size (microcephaly); abn...
4. `medquad_ans_0000653-5` (score=0.6445) - These resources address the diagnosis or management of MOPDII: - Genetic Testing Registry: Microcephalic osteodysplastic primordial dwarfism type 2 These resources from MedlinePlus offer information about the diagnosi...
5. `medquad_ans_0000653-1` (score=0.6361) - Microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is a condition characterized by short stature (dwarfism) with other skeletal abnormalities (osteodysplasia) and an unusually small head size (microcep...

**Answer**:

Yes, microcephalic osteodysplastic primordial dwarfism type II (MOPD2) is inherited in an autosomal recessive manner [medquad_ans_0004006-1].

**Supported citations**: `medquad_ans_0004006-1`
**Unsupported citations**: _none_

### Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.670, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000742-2` (score=0.7000) - Ochoa syndrome is a rare disorder. About 150 cases have been reported in the medical literature.
2. `medquad_ans_0000742-5` (score=0.6704) - These resources address the diagnosis or management of Ochoa syndrome: - Gene Review: Gene Review: Urofacial Syndrome - Genetic Testing Registry: Ochoa syndrome - National Institute of Diabetes and Digestive and Kidne...
3. `medquad_ans_0000742-1` (score=0.6620) - Ochoa syndrome is a disorder characterized by urinary problems and unusual facial expressions. The urinary problems associated with Ochoa syndrome typically become apparent in early childhood or adolescence. People wi...
4. `medquad_ans_0004262-1` (score=0.2739) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, a...
5. `medquad_ans_0000210-1__dup1` (score=0.1779) - Alagille syndrome is an inherited disorder in which a person has fewer than the normal number of small bile ducts inside the liver. It is a complex disorder that can affect other parts of the body including the heart,...

**Answer**:

The provided context does not specify whether Ochoa syndrome is inherited. [medquad_ans_0000742-1] describes the symptoms and associated complications of Ochoa syndrome but does not mention inheritance. [medquad_ans_0000742-5] lists resources for diagnosis and management, but none indicate inheritance patterns. Therefore, based on the given information, it is not possible to determine if Ochoa syndrome is inherited.

**Supported citations**: `medquad_ans_0000742-1`, `medquad_ans_0000742-5`
**Unsupported citations**: _none_

### Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.689, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000531-1` (score=0.7000) - Isodicentric chromosome 15 syndrome is a developmental disorder with a broad spectrum of features. The signs and symptoms vary among affected individuals. Poor muscle tone is commonly seen in individuals with isodicen...
2. `medquad_ans_0000531-3` (score=0.6994) - Isodicentric chromosome 15 syndrome results from the presence of an abnormal extra chromosome, called an isodicentric chromosome 15, in each cell. An isodicentric chromosome contains mirror-image segments of genetic m...
3. `medquad_ans_0000531-2` (score=0.6907) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
4. `medquad_ans_0000531-5` (score=0.6616) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management...
5. `medquad_ans_0006555-1` (score=0.3849) - 15q13.3 microdeletion syndrome is a type of contiguous gene deletion syndrome. Individuals with this microdeletion may have very different signs and symptoms from other affected individuals (even within the same famil...

**Answer**:

The provided context does not mention any specific treatments for isodicentric chromosome 15 syndrome. Therefore, based on the available information, treatments cannot be determined. [medquad_ans_0000531-1] [medquad_ans_0000531-3]

**Supported citations**: `medquad_ans_0000531-1`, `medquad_ans_0000531-3`
**Unsupported citations**: _none_

### Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.908, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006224-3` (score=0.7000) - What causes Troyer syndrome? Troyer syndrome is caused by mutations in the SPG20 gene. This gene gives the body instructions to make a protein called spartin, which is present in many body tissues, including those of ...
2. `medquad_ans_0004414-3` (score=0.7000) - What causes Nicolaides-Baraitser syndrome? Nicolaides-Baraitser syndrome (NCBRS) is caused by mutations in the SMARCA2 gene, which is located on the small arm of chromosome 9. All mutations that have been identified i...
3. `medquad_ans_0002439-1` (score=0.6122) - Friedreich ataxia is an inherited condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features...
4. `medquad_ans_0000998-1` (score=0.6122) - Troyer syndrome is part of a group of genetic disorders known as hereditary spastic paraplegias. These disorders are characterized by progressive muscle stiffness (spasticity) and the development of paralysis of the l...
5. `medquad_ans_0000998-5` (score=0.5272) - These resources address the diagnosis or management of Troyer syndrome: - Gene Review: Gene Review: Hereditary Spastic Paraplegia Overview - Gene Review: Gene Review: Troyer Syndrome - Genetic Testing Registry: Troyer...

**Answer**:

Troyer syndrome is caused by mutations in the SPG20 gene. This gene provides instructions for making a protein called spartin, which is present in many body tissues, including those of the nervous system. It is thought that the syndrome is caused by a loss of function of the spartin protein, although the exact role of this protein in cellular functions is not fully understood [medquad_ans_0006224-3].

**Supported citations**: `medquad_ans_0006224-3`
**Unsupported citations**: _none_

### Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.939, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003672-1` (score=0.7000) - Timothy syndrome is a type of long QT syndrome. It affects many parts of the body including the heart, fingers, toes, face, and the nervous system. It is characterized by long QT syndrome, although some people with Ti...
2. `medquad_ans_0000983-1` (score=0.6711) - Timothy syndrome is a rare disorder that affects many parts of the body including the heart, digits (fingers and toes), and the nervous system. Timothy syndrome is characterized by a heart condition called long QT syn...
3. `medquad_ans_0000983-5` (score=0.6437) - These resources address the diagnosis or management of Timothy syndrome: - Gene Review: Gene Review: Timothy Syndrome - Genetic Testing Registry: Timothy syndrome - MedlinePlus Encyclopedia: Arrhythmias - MedlinePlus ...
4. `medquad_ans_0000983-2` (score=0.6341) - Timothy syndrome is a rare condition; fewer than 20 people with this disorder have been reported worldwide. The classic type of Timothy syndrome appears to be more common than the atypical type, which has been identif...
5. `medquad_ans_0000983-4` (score=0.4660) - This condition is considered to have an autosomal dominant pattern of inheritance, which means one copy of the altered CACNA1C gene in each cell is sufficient to cause the disorder. Most cases result from new mutation...

**Answer**:

The genetic changes related to Timothy syndrome are changes in the CACNA1C gene. All cases of Timothy syndrome appear to be due to changes in the CACNA1C gene [medquad_ans_0003672-1].

**Supported citations**: `medquad_ans_0003672-1`
**Unsupported citations**: _none_

### Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000875-1` (score=0.7000) - Rheumatoid arthritis is a disease that causes chronic abnormal inflammation, primarily affecting the joints. The most common signs and symptoms are pain, swelling, and stiffness of the joints. Small joints in the hand...
2. `medquad_ans_0000061-3__dup3` (score=0.7000) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by i...
3. `medquad_ans_0000061-1__dup4` (score=0.6171) - An Inflammatory, Autoimmune Disease Rheumatoid arthritis is an inflammatory disease that causes pain, swelling, stiffness, and loss of function in the joints. It can cause mild to severe symptoms. Rheumatoid arthritis...
4. `medquad_ans_0000061-7__dup1` (score=0.6144) - Rheumatoid arthritis is an inflammatory disease that causes pain, swelling, stiffness, and loss of function in the joints. It can cause mild to severe symptoms. People with rheumatoid arthritis may feel sick, tired, a...
5. `medquad_ans_0000782-1__dup1` (score=0.5800) - Rheumatoid arthritis (RA) is a form of arthritis that causes pain, swelling, stiffness and loss of function in your joints. It can affect any joint but is common in the wrist and fingers. More women than men get rheum...

**Answer**:

The symptoms of rheumatoid arthritis include pain, swelling, and stiffness of the joints, with small joints in the hands and feet being most commonly affected. The disease often affects joints in a symmetrical pattern, such as both hands or both knees. Joint pain and stiffness are typically worse in the morning or after prolonged rest. Additional signs and symptoms may include fatigue, low fever, weight loss, anemia, and the development of rheumatoid nodules—firm, noncancerous lumps under the skin or elsewhere in the body. The condition can also cause inflammation of other tissues and organs, such as the eyes, lungs, and blood vessels [medquad_ans_0000875-1]. Some individuals experience episodes of symptoms (flares) followed by periods of remission [medquad_ans_0000875-1].

**Supported citations**: `medquad_ans_0000875-1`
**Unsupported citations**: _none_

### Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Scores**: context_relevance=1.000, groundedness=0.990, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0002796-1` (score=0.7000) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is no...
2. `medquad_ans_0002796-4` (score=0.6805) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...
3. `medquad_ans_0002796-2` (score=0.6707) - What are the signs and symptoms of Henoch-Schonlein purpura? The Human Phenotype Ontology provides the following list of signs and symptoms for Henoch-Schonlein purpura. If the information is available, the table belo...
4. `medquad_ans_0000149-1__dup3` (score=0.6001) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, d...
5. `medquad_ans_0005207-1` (score=0.5922) - What are the signs and symptoms of Purpura simplex? The Human Phenotype Ontology provides the following list of signs and symptoms for Purpura simplex. If the information is available, the table below includes how oft...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is not fully understood, it may develop as an immune response to an infection. HSP is usually seen in children, but it may affect people of any age. Most cases go away on their own without treatment. [medquad_ans_0002796-1]

**Supported citations**: `medquad_ans_0002796-1`
**Unsupported citations**: _none_

### Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0003166-1` (score=0.7000) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information ...
2. `medquad_ans_0003162-1` (score=0.6906) - Hyper IgM syndrome is a type of primary immunodeficiency syndrome. Primary immunodeficiency occurs when part of a persons immune system is missing or does not work correctly. The bodies of people with primary immunode...
3. `medquad_ans_0003163-1` (score=0.6707) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information ...
4. `medquad_ans_0003162-2` (score=0.6624) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 1? Symptoms and physical findings associated with hyper IgM syndrome usually become apparent in the first or second year of life. This condition ...
5. `medquad_ans_0001052-5` (score=0.6374) - These resources address the diagnosis or management of X-linked hyper IgM syndrome: - Gene Review: Gene Review: X-Linked Hyper IgM Syndrome - Genetic Testing Registry: Immunodeficiency with hyper IgM type 1 - MedlineP...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [medquad_ans_0003166-1].

**Supported citations**: `medquad_ans_0003166-1`
**Unsupported citations**: _none_

### Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.943, answer_relevance=0.844

**Top docs**:

1. `medquad_ans_0002533-4` (score=0.7000) - How is Gerstmann-Straussler-Scheinker disease inherited? Gerstmann-Straussler-Scheinker disease (GSS) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutatio...
2. `medquad_ans_0000896-3` (score=0.7000) - Senior-Lken syndrome can be caused by mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopic, finger...
3. `medquad_ans_0000896-1` (score=0.6122) - Senior-Lken syndrome is a rare disorder characterized by the combination of two specific features: a kidney condition called nephronophthisis and an eye condition known as Leber congenital amaurosis. Nephronophthisis ...
4. `medquad_ans_0000883-3__dup1` (score=0.6122) - How is Buschke Ollendorff syndrome inherited? Buschke Ollendorff syndrome (BOS) is caused by mutations in the LEMD3 gene and is inherited in an autosomal dominant manner. This means that only one changed (mutated) cop...
5. `medquad_ans_0000700-4` (score=0.5272) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

Genetic changes related to Senior-Lken syndrome include mutations in one of at least five genes. These genes are known or suspected to play roles in cell structures called cilia, which are involved in signaling pathways between cells. Mutations in these genes likely lead to problems with the structure and function of cilia, disrupting important chemical signaling pathways within cells. However, it remains unclear how these defects specifically lead to nephronophthisis and Leber congenital amaurosis. In some cases, individuals with Senior-Lken syndrome do not have identified mutations in these five genes, and the genetic cause in those cases is unknown [medquad_ans_0000896-3].

**Supported citations**: `medquad_ans_0000896-3`
**Unsupported citations**: _none_

### Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000610-2` (score=0.7000) - The prevalence of lymphedema-distichiasis syndrome is unknown. Because the extra eyelashes can be overlooked during a medical examination, researchers believe that some people with this condition may be misdiagnosed a...
2. `medquad_ans_0003700-2` (score=0.6805) - What are the signs and symptoms of Lymphedema-distichiasis syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lymphedema-distichiasis syndrome. If the information is available...
3. `medquad_ans_0003700-1` (score=0.6706) - Lymphedema distichiasis syndrome is a condition that affects the normal function of the lymphatic system (part of the immune system that produces and transports fluids and immune cells throughout the body). People wit...
4. `medquad_ans_0000610-1` (score=0.6115) - Lymphedema-distichiasis syndrome is a condition that affects the normal function of the lymphatic system, which is a part of the circulatory and immune systems. The lymphatic system produces and transports fluids and ...
5. `medquad_ans_0000610-5` (score=0.0865) - These resources address the diagnosis or management of lymphedema-distichiasis syndrome: - Gene Review: Gene Review: Lymphedema-Distichiasis Syndrome - Genetic Testing Registry: Distichiasis-lymphedema syndrome - Medl...

**Answer**:

The prevalence of lymphedema-distichiasis syndrome is unknown [medquad_ans_0000610-2].

**Supported citations**: `medquad_ans_0000610-2`
**Unsupported citations**: _none_

### Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Scores**: context_relevance=1.000, groundedness=0.948, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0005761-3` (score=0.7000) - How is spinocerebellar ataxia type 11 inherited? SCA11 is inherited in an autosomal dominant manner. The rate of de novo mutations is not known. Each child of an individual with SCA11 has a 50% chance of inheriting th...
2. `medquad_ans_0000049-4__dup4` (score=0.6365) - Testing for monogenic diabetes involves providing a blood sample from which DNA is isolated. The DNA is analyzed for changes in the genes that cause monogenic diabetes. Abnormal results can determine the gene responsi...
3. `medquad_ans_0000013-26` (score=0.5350) - Finding out your test results can affect you emotionally. Learning that you are someone in your family has or is at risk for a disease can be scary. Some people can also feel guilty, angry, anxious, or depressed when ...
4. `medquad_ans_0000013-22` (score=0.0986) - In a genetic test, a small sample of blood, saliva, or tissue is taken to examine a person's genes. Sometimes, genetic testing can detect diseases that may be preventable or treatable. This type of testing is availabl...
5. `medquad_ans_0000399-1__dup1` (score=0.0873) - Summary : Genetic tests are tests on blood and other tissue to find genetic disorders. Over 2000 tests are available. Doctors use genetic tests for several reasons. These include - Finding genetic diseases in unborn b...

**Answer**:

Yes, information about genetic testing for spinocerebellar ataxia type 11 is available. Genetic testing of adults who do not have any symptoms but are at-risk of having inherited the mutation is possible. However, testing is not useful in predicting age of onset, severity, type of symptoms, or rate of progression in individuals who do not have any symptoms. The affected family member should be tested first to confirm the molecular diagnosis in the family. For detailed information on testing, inheritance, and genetic counseling, including a comprehensive review of spinocerebellar ataxia type 11, the GeneReviews website provides current, evidence-based information [medquad_ans_0005761-3]. Additionally, the Genetic Testing Registry (GTR) offers information about genetic tests, though it is intended for healthcare providers and researchers [medquad_ans_0005761-3].

**Supported citations**: `medquad_ans_0005761-3`
**Unsupported citations**: _none_

### Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.514, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000149-1` (score=0.7000) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxy...
2. `medquad_ans_0000654-1` (score=0.6319) - Microcephaly-capillary malformation syndrome is an inherited disorder characterized by an abnormally small head size (microcephaly) and abnormalities of small blood vessels in the skin called capillaries (capillary ma...
3. `medquad_ans_0005994-1` (score=0.6072) - Systemic capillary leak syndrome is a condition in which fluid and proteins leak out of tiny blood vessels and flow into surrounding tissues, resulting in dangerously low blood pressure. Attacks frequently last for se...
4. `medquad_ans_0000654-2` (score=0.5967) - Microcephaly-capillary malformation syndrome is rare. About a dozen people have been diagnosed with the disorder.
5. `medquad_ans_0000149-5` (score=0.5759) - These resources address the diagnosis or management of CM-AVM: - Gene Review: Gene Review: RASA1-Related Disorders - Genetic Testing Registry: Capillary malformation-arteriovenous malformation These resources from Med...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin. These malformations appear as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs, and may be visible at birth or during childhood. While capillary malformations alone usually do not cause health problems, some individuals with CM-AVM also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs), which are abnormal connections between arteries, veins, and capillaries affecting blood circulation. These abnormalities can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure, some of which can be life-threatening. Complications of AVM

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002000-1` (score=0.7000) - What are the signs and symptoms of Dystonia 7, torsion? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 7, torsion. If the information is available, the table below includes...
2. `medquad_ans_0001992-1` (score=0.6618) - What are the signs and symptoms of Dystonia 16? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 16. If the information is available, the table below includes how often the s...
3. `medquad_ans_0001999-1` (score=0.6472) - What are the signs and symptoms of Dystonia 6, torsion? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 6, torsion. If the information is available, the table below includes...
4. `medquad_ans_0001994-1` (score=0.4440) - What are the signs and symptoms of Dystonia 18? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 18. If the information is available, the table below includes how often the s...
5. `medquad_ans_0001986-1` (score=0.4267) - What are the signs and symptoms of Dystonia 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 1. If the information is available, the table below includes how often the sym...

**Answer**:

The symptoms of Dystonia 16 include cognitive impairment (5%), abnormal pyramidal signs, bradykinesia, delayed speech and language development, dysarthria, dysphagia, gait disturbance, hyperreflexia, involuntary movements, laryngeal dystonia, limb dystonia, lower limb pain, morphological abnormality of the pyramidal tract, motor delay, parkinsonism, postural tremor, progressive course, and retrocollis [medquad_ans_0001992-1].

**Supported citations**: `medquad_ans_0001992-1`
**Unsupported citations**: _none_

### Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Scores**: context_relevance=0.857, groundedness=0.980, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0005548-5` (score=0.7000) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not tr...
2. `medquad_ans_0005548-1` (score=0.6450) - Septo-optic dysplasia is a disorder of early brain development. The signs and symptoms vary from person to person; however, underdevelopment (hypoplasia) of the optic nerve, abnormal formation of structures along the ...
3. `medquad_ans_0005548-2` (score=0.6360) - What are the signs and symptoms of Septo-optic dysplasia? Symptoms may include blindness in one or both eyes, pupil dilation in response to light, nystagmus (a rapid, involuntary to-and-fro movement of the eyes), inwa...
4. `medquad_ans_0000899-5` (score=0.6205) - These resources address the diagnosis or management of septo-optic dysplasia: - Genetic Testing Registry: Septo-optic dysplasia sequence - MedlinePlus Encyclopedia: Growth Hormone Deficiency - MedlinePlus Encyclopedia...
5. `medquad_ans_0000899-1` (score=0.5965) - Septo-optic dysplasia is a disorder of early brain development. Although its signs and symptoms vary, this condition is traditionally defined by three characteristic features: underdevelopment (hypoplasia) of the opti...

**Answer**:

Treatment for septo-optic dysplasia is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not treatable, and vision, physical, and occupational therapies may be required [medquad_ans_0005548-5].

**Supported citations**: `medquad_ans_0005548-5`
**Unsupported citations**: _none_

### Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Scores**: context_relevance=0.800, groundedness=0.941, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000208-1__dup4` (score=0.7000) - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. Benign prostatic hyperplasia is also called benign prostatic hypertrophy or benign prostatic ...
2. `medquad_ans_0000208-14` (score=0.7000) - - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. - The prostate is a walnut-shaped gland that is part of the male reproductive system. - The...
3. `medquad_ans_0000208-7` (score=0.6265) - The complications of benign prostatic hyperplasia may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney da...
4. `medquad_ans_0000036_3-1` (score=0.5933) - Key Points - Prostate cancer is a disease in which malignant (cancer) cells form in the tissues of the prostate. - Prostate cancer is the most common nonskin cancer among men in the United States. - Different factors ...
5. `medquad_ans_0000208-11` (score=0.5095) - The complications of benign prostatic hyperplasia treatment depend on the type of treatment. Medications Medications used to treat benign prostatic hyperplasia may have side effects that sometimes can be serious. Men ...

**Answer**:

The complications of benign prostatic hyperplasia (prostate enlargement) may include acute urinary retention, chronic or long-lasting urinary retention, blood in the urine, urinary tract infections (UTIs), bladder damage, kidney damage, and bladder stones [medquad_ans_0000208-14].

**Supported citations**: `medquad_ans_0000208-14`
**Unsupported citations**: _none_

### Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Scores**: context_relevance=0.889, groundedness=0.775, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0004784-5` (score=0.7000) - How might Pelizaeus-Merzbacher disease be treated?
2. `medquad_ans_0000001-3__dup1` (score=0.7000) - When the absence of the septum pellucidum is part of septo-optic dysplasia, the prognosis varies according to the presence and severity of associated symptoms. By itself, absence of the septum pellucidum is not life-t...
3. `medquad_ans_0000001-1__dup1` (score=0.6122) - The septum pellucidum (SP) is a thin membrane located at the midline of the brain between the two cerebral hemispheres, or halves of the brain.. It is connected to the corpus callosum -- a collection of nerve fibers t...
4. `medquad_ans_0000029-18` (score=0.6122) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
5. `medquad_ans_0004770-2` (score=0.5272) - How might patulous eustacian tube be treated? While no standard treatment has been found to work for every patient, there are several options that have been used to successfully manage the symptoms in a number of case...

**Answer**:

The provided context does not contain specific information about the treatments for the absence of the septum pellucidum. While it mentions that absence of the septum pellucidum is part of septo-optic dysplasia and that this condition may involve pituitary deficiencies and abnormal optic disk development, it does not detail any treatments for this specific condition [medquad_ans_0000001-1__dup1]. Therefore, based on the available information, the treatments for absence of the septum pellucidum cannot be determined.

**Supported citations**: `medquad_ans_0000001-1__dup1`
**Unsupported citations**: _none_

### Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Scores**: context_relevance=1.000, groundedness=0.511, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.7000) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (...
2. `medquad_ans_0002322-1` (score=0.6806) - Febrile ulceronecrotic Mucha-Habermann disease (FUMHD) is a rare and severe form of pityriasis lichenoides et varioliformis acuta (PLEVA). PLEVA is characterized by skin lesions that ulcerate, breakdown, form open sor...
3. `medquad_ans_0002322-3` (score=0.6636) - What causes febrile ulceronecrotic Mucha-Habermann disease? The cause of FUMHD is not known (idiopathic). A hypersensitivity to an infectious agent is suggested to be the main cause. Single cases of people with FUMHD ...
4. `medquad_ans_0002322-4` (score=0.6619) - How is febrile ulceronecrotic Mucha-Habermann disease definitively diagnosed? FUMHD is diagnosed based upon the clinical symptoms in the patient, with confirmation by skin biopsy. Skin biopsy findings suggestive of FU...
5. `medquad_ans_0002322-5` (score=0.6437) - How is febrile ulceronecrotic Mucha-Habermann disease (FUMHD) treated? It is important that FUMHD is diagnosed and treated as soon as possible. While a number of treatments have been tried, it is hard to asses the ben...

**Answer**:

The symptoms of Febrile Ulceronecrotic Mucha-Habermann disease (FUMHD) include initial red scaly skin lesions (papules) that ulcerate, break down, form open sores, and then develop a red-brown crust (i.e., PLEVA). The condition can rapidly progress to large, destructive ulcers with extensive, painful loss of skin tissue. Skin lesions may become infected, leading to pus and a putrid odor. FUMHD is often associated with high fever (up to 104°F), which may be persistent or intermittent. Other symptoms may include feeling ill, sore throat, congestion, muscle soreness or pain, joint pain, diarrhea, central nervous system symptoms, abdominal pain, enlarged spleen, arthritis, megaloblastic anemia, interstitial pneumonitis, lymphocytic (viral) myocarditis, and sepsis. FUMHD can

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0002833-4` (score=0.7000) - How is hereditary leiomyomatosis and renal cell cancer inherited? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the ...
2. `medquad_ans_0003951-3` (score=0.7000) - Is Merkel cell carcinoma inherited? To our knowledge, there currently is no evidence that Merkel cell carcinoma (MCC) is inherited. While DNA changes (mutations) found in the cells of MCC tumors can lead to MCC, these...
3. `medquad_ans_0000167-5__dup1` (score=0.6122) - Autosomal dominant PKD is the most common form of PKD and the most common inherited disorder of the kidneys.3 The term autosomal dominant means a child can get the disorder by inheriting the gene mutation from only on...
4. `medquad_ans_0000465-5` (score=0.6122) - These resources address the diagnosis or management of HLRCC: - Gene Review: Gene Review: Hereditary Leiomyomatosis and Renal Cell Cancer - Genetic Testing Registry: Hereditary leiomyomatosis and renal cell cancer - M...
5. `medquad_ans_0002833-3` (score=0.5272) - What causes hereditary leiomyomatosis and renal cell cancer? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is caused by changes (mutations) in the FH gene. This gene gives the body instructions for making an...

**Answer**:

Yes, hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the gene in each cell is enough to cause symptoms of the condition. In some cases, an affected person inherits the mutated copy of the gene from an affected parent. Other cases result from new mutations in the gene that occur for the first time in the affected individual. When an individual with an autosomal dominant condition has children, each child has a 50% (1 in 2) risk to inherit the mutated gene [medquad_ans_0002833-4].

**Supported citations**: `medquad_ans_0002833-4`
**Unsupported citations**: _none_

### Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000389-1` (score=0.7000) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which al...
2. `medquad_ans_0002445-1` (score=0.6719) - What are the signs and symptoms of Frontometaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontometaphyseal dysplasia. If the information is available, the tabl...
3. `medquad_ans_0000389-5` (score=0.6707) - These resources address the diagnosis or management of frontometaphyseal dysplasia: - Gene Review: Gene Review: Otopalatodigital Spectrum Disorders - Genetic Testing Registry: Frontometaphyseal dysplasia These resourc...
4. `medquad_ans_0005841-2` (score=0.4496) - What are the signs and symptoms of Spondyloepiphyseal dysplasia Maroteaux type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia Maroteaux type. If the in...
5. `medquad_ans_0000390-1` (score=0.3485) - Frontonasal dysplasia is a condition that results from abnormal development of the head and face before birth. People with frontonasal dysplasia have at least two of the following features: widely spaced eyes (ocular ...

**Answer**:

The symptoms of Frontometaphyseal dysplasia include abnormalities in dental morphology (90%), abnormality of the frontal sinus (90%), abnormality of the metaphyses (90%), bowing of the long bones (90%), camptodactyly of the fingers (90%), craniofacial hyperostosis (90%), hypertelorism (90%), limitation of joint mobility (90%), prominent supraorbital ridges (90%), abnormal form of the vertebral bodies (50%), abnormality of the palate (50%), accelerated skeletal maturation (50%), aplasia/hypoplasia of the thumb (50%), arachnodactyly (50%), conductive hearing impairment (50%), elbow dislocation (50%), scoliosis (50%), sensorineural hearing impairment (50%), skeletal muscle atrophy (50%), and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Scores**: context_relevance=0.818, groundedness=0.897, answer_relevance=0.527

**Top docs**:

1. `medquad_ans_0000128-10` (score=0.7000) - Eat a high-fiber diet to help prevent problems. Talk to your doctor about using fiber products like Benefiber, Citrucel, or Metamucil. Daily use can help you get the fiber you need if you do not get it through your di...
2. `medquad_ans_0000097-11` (score=0.6940) - - Diverticular disease is a condition that occurs when a person has problems from small pouches, or sacs, that have formed and pushed outward through weak spots in the colon wall. The problems that occur with divertic...
3. `medquad_ans_0000128-7` (score=0.6531) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding,...
4. `medquad_ans_0000097-3__dup2` (score=0.5111) - Scientists are not certain what causes diverticulosis and diverticular disease. For more than 50 years, the most widely accepted theory was that a low-fiber diet led to diverticulosis and diverticular disease. Diverti...
5. `medquad_ans_0000097-7` (score=0.4499) - Diverticulosis Health care providers often find diverticulosis during a routine x ray or a colonoscopy, a test used to look inside the rectum and entire colon to screen for colon cancer or polyps or to evaluate the so...

**Answer**:

Treatments for diverticular disease include a high-fiber diet or fiber supplements, medications, and possibly probiotics, especially for diverticulosis with symptoms [medquad_ans_0000097-11]. For mild diverticulitis without complications, treatment may involve rest, oral antibiotics, and a liquid diet [medquad_ans_0000097-11]. Diverticular bleeding, though rare, may stop on its own and not require treatment; however, if it does not stop, abdominal surgery with colon resection may be necessary [medquad_ans_0000097-11]. In cases of complications such as abscess, perforation, peritonitis, fistula, or intestinal obstruction, medical intervention is required to prevent worsening and serious illness [medquad_ans_0000097-11]. Additionally, patients should consult their doctor

**Supported citations**: `medquad_ans_0000097-11`
**Unsupported citations**: _none_

### Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000619-1__dup1` (score=0.7000) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is avail...
2. `medquad_ans_0006066-1` (score=0.6706) - What are the signs and symptoms of Thompson Baraitser syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thompson Baraitser syndrome. If the information is available, the tabl...
3. `medquad_ans_0002441-2` (score=0.2749) - What are the signs and symptoms of Froelich syndrome? Signs and symptoms of Froelich syndrome include obesity, small testes, delay in the onset of puberty, short stature (compared to other family members of the same s...
4. `medquad_ans_0000676-2__dup1` (score=0.0765) - What are the symptoms of Bell's palsy?
5. `medquad_ans_0000108-2__dup1` (score=0.0669) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The symptoms of Baraitser Brett Piesowicz syndrome include cerebral calcification, hyperreflexia, hypertonia, microcephaly, seizures, abnormality of movement, cerebral cortical atrophy, cataract, opacification of the corneal stroma, renal insufficiency, anteverted nares, cerebellar hypoplasia, decreased liver function, elevated hepatic transaminases, failure to thrive, hepatomegaly, high palate, increased CSF protein, intellectual disability, profound, jaundice, lissencephaly, long philtrum, low-set ears, microretrognathia, muscular hypotonia of the trunk, nystagmus, pachygyria, petechiae, phenotypic variability, polymicrogyria, sloping forehead, spasticity, splenomegaly, thrombocytopenia, ventriculomeg

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Scores**: context_relevance=0.833, groundedness=0.979, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000052-2__dup1` (score=0.7000) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
2. `medquad_ans_0003911-2` (score=0.7000) - How might meningiomas be treated? The treatment varies depending on the location of the meningioma and the symptoms caused by the tumor. Careful observation is sometimes the best course of action for people with a men...
3. `medquad_ans_0000029-18` (score=0.6122) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
4. `medquad_ans_0003911-1` (score=0.6122) - Meningiomas originate in the meninges, the membranes that surround the brain and spinal cord. Most meningiomas are benign, though a minority of meningiomas can be classified as atypical or malignant. Though rare, mali...
5. `medquad_ans_0000273-2__dup2` (score=0.5272) - There is no cure for TSC, although treatment is available for a number of the symptoms. Rapamycin and related drugs are not yet approved by the U.S. Food and Drug Administration (FDA) for any purpose in individuals wi...

**Answer**:

The treatments for meningioma vary depending on the location of the meningioma and the symptoms it causes. Careful observation may be the best course of action for some individuals. When treatment is necessary, surgery and radiation are the most common forms of treatment. Radiation may be used if the meningioma cannot be operated on or if it is only partially removed by surgery. Radiation is also used in cases of malignant, atypical, or recurrent tumors. Other treatments that have been tried or are being explored include hydroxyurea, epidermal growth factor receptor inhibitors, platelet-derived growth factor receptor inhibitors, vascular endothelial growth factor inhibitors, immunotherapy to stimulate the immune system, and somatostatin analogs which prevent the release of growth hormones [medquad_ans_0003911-2].

**Supported citations**: `medquad_ans_0003911-2`
**Unsupported citations**: _none_

### Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000006_7-2` (score=0.7000) - Key Points - Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and s...
2. `medquad_ans_0000006_7-8` (score=0.6678) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a te...
3. `medquad_ans_0000006_7-1` (score=0.6540) - Key Points - Childhood central nervous system (CNS) germ cell tumors form from germ cells. - There are different types of childhood CNS germ cell tumors. - Germinomas - Nongerminomas - The cause of most childhood CNS ...
4. `medquad_ans_0000006_4-8` (score=0.4708) - Key Points - There are different types of treatment for patients with central nervous system atypical teratoid/rhabdoid tumor. - Children with atypical teratoid/rhabdoid tumor should have their treatment planned by a ...
5. `medquad_ans_0000006_6-8` (score=0.4396) - Key Points - There are different types of treatment for children who have central nervous system (CNS) embryonal tumors. - Children who have CNS embryonal tumors should have their treatment planned by a team of health...

**Answer**:

The treatments for Childhood Central Nervous System (CNS) Germ Cell Tumors include radiation therapy, chemotherapy, surgery, high-dose chemotherapy with stem cell rescue, and new treatments being tested in clinical trials. Treatment is planned by a team of healthcare providers who are experts in treating cancer in children, and patients may consider participating in a clinical trial, which can occur before, during, or after starting treatment [medquad_ans_0000006_7-8].

**Supported citations**: `medquad_ans_0000006_7-8`
**Unsupported citations**: _none_

### Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Scores**: context_relevance=1.000, groundedness=0.471, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0000729-2` (score=0.7000) - Neutral lipid storage disease with myopathy is a rare condition; its incidence is unknown.
2. `medquad_ans_0000729-1` (score=0.6710) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the ...
3. `medquad_ans_0004401-3` (score=0.6586) - What causes neutral lipid storage disease with myopathy? Neutral lipid storage disease with myopathy is caused by mutations in the PNPLA2 gene. This gene provides instructions for making an enzyme called adipose trigl...
4. `medquad_ans_0004401-1` (score=0.6041) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (my...
5. `medquad_ans_0004401-5` (score=0.6029) - How might neutral lipid storage disease with myopathy be treated? To date, there is no treatment for the underlying metabolic problem. Current therapies include adhering to strict dietary guidelines and utilizing trea...

**Answer**:

Neutral lipid storage disease with myopathy is a rare condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. This abnormal accumulation leads to muscle weakness (myopathy) due to fat buildup in muscle tissue. Other possible features include a fatty liver, weakened and enlarged heart (cardiomyopathy), inflammation of the pancreas (pancreatitis), reduced thyroid activity (hypothyroidism), and type 2 diabetes mellitus. The condition is caused by mutations in the PNPLA2 gene, which results in impaired function of the adipose triglyceride lipase (ATGL) enzyme, leading to the inability to break down triglycerides properly and their subsequent accumulation. It is inherited in an autosomal recessive pattern, and there is currently no treatment to correct the underlying metabolic problem. Management focuses on dietary guidelines and symptom-related treatments, with recent studies suggesting potential benefit from bezafibrate for lipid accumulation and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Scores**: context_relevance=0.800, groundedness=0.941, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0006546-1` (score=0.7000) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. ...
2. `medquad_ans_0000981-1__dup1` (score=0.6700) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been repo...
3. `medquad_ans_0006546-2` (score=0.6605) - How might a Zika virus infection be treated? There is no vaccine to prevent Zika virus infections, nor is there a specific medicine to treat Zika. Individuals infected with the Zika virus should get plenty of rest, dr...
4. `medquad_ans_0000193-1__dup2` (score=0.6507) - Microcephaly is a medical condition in which the circumference of the head is smaller than normal because the brain has not developed properly or has stopped growing. Microcephaly can be present at birth or it may dev...
5. `medquad_ans_0000174-1__dup3` (score=0.6228) - Chikungunya is a virus that spread by the same kinds of mosquitoes that spread dengue and Zika virus. Rarely, it can spread from mother to newborn around the time of birth. It may also possibly spread through infected...

**Answer**:

Zika virus is a virus spread primarily through the bite of an infected mosquito. It can also be transmitted from a pregnant mother to her child during pregnancy or around the time of birth, through sexual contact with an affected male partner, and through blood transfusions. The illness associated with Zika virus infection is usually mild, with symptoms such as fever, rash, joint pain, and conjunctivitis, which typically last for several days to a week. In rare cases, Zika virus infection has been linked to serious birth defects, including microcephaly and other brain anomalies, as well as Guillain-Barre syndrome. There is currently no vaccine or medicine to treat Zika virus infection, and prevention measures include avoiding mosquito bites, postponing travel to affected areas, and taking precautions for sexual transmission [medquad_ans_0006546-1][medquad_ans_0000981-1__dup1].

**Supported citations**: `medquad_ans_0006546-1`, `medquad_ans_0000981-1__dup1`
**Unsupported citations**: _none_

### Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000019-1__dup2` (score=0.7000) - Broken heart syndrome is a condition in which extreme stress can lead to heart muscle failure. The failure is severe, but often short-term. Most people who experience broken heart syndrome think they may be having a h...
2. `medquad_ans_0000019-3__dup2` (score=0.6700) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack...
3. `medquad_ans_0000019-4__dup2` (score=0.6513) - Broken heart syndrome affects women more often than men. Often, people who experience broken heart syndrome have previously been healthy. Research shows that the traditional risk factors for heart disease may not appl...
4. `medquad_ans_0000019-5__dup1` (score=0.6231) - All chest pain should be checked by a doctor. Because symptoms of broken heart syndrome are similar to those of a heart attack, it is important to seek help right away. Your doctor may not be able to diagnose broken h...
5. `medquad_ans_0000019-8` (score=0.5630) - Researchers are still learning about broken heart syndrome, and no treatments have been shown to prevent it. For people who have experienced the condition, the risk of recurrence is low. An emotionally upsetting or se...

**Answer**:

The cause of broken heart syndrome is not fully known. In most cases, symptoms are triggered by extreme emotional or physical stress, such as intense grief, anger, or surprise. Researchers think that the stress releases hormones that stun the heart and affect its ability to pump blood to the body. [medquad_ans_0000019-1__dup2]

**Supported citations**: `medquad_ans_0000019-1__dup2`
**Unsupported citations**: _none_

### Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Scores**: context_relevance=0.800, groundedness=0.971, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000009-3__dup4` (score=0.7000) - Hypothyroidism has several causes, including - Hashimotos disease - thyroiditis, or inflammation of the thyroid - congenital hypothyroidism, or hypothyroidism that is present at birth - surgical removal of part or all...
2. `medquad_ans_0000005-1__dup5` (score=0.6711) - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. An autoimmune disease is a disorder in which the bodys immune system attacks the bodys own cells and...
3. `medquad_ans_0000008-3__dup4` (score=0.6407) - Hyperthyroidism has several causes, including - Graves disease - thyroid nodules - thyroiditis, or inflammation of the thyroid - consuming too much iodine - overmedicating with synthetic thyroid hormone, which is used...
4. `medquad_ans_0000005-9__dup1` (score=0.5907) - - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. - Hashimotos disease often leads to reduced thyroid function, or hypothyroidism. Hypothyroidism is...
5. `medquad_ans_0000892-1__dup1` (score=0.5338) - Your thyroid is a butterfly-shaped gland in your neck, just above your collarbone. It is one of your endocrine glands, which make hormones. Thyroid hormones control the rate of many activities in your body. These incl...

**Answer**:

Hashimoto thyroiditis, also called chronic lymphocytic thyroiditis, is the most common cause of hypothyroidism in the United States. It is a form of chronic inflammation of the thyroid gland and an autoimmune disorder in which the immune system attacks the thyroid, causing inflammation and interfering with its ability to produce thyroid hormones [medquad_ans_0000009-3__dup4].

**Supported citations**: `medquad_ans_0000009-3__dup4`
**Unsupported citations**: _none_

### Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Scores**: context_relevance=1.000, groundedness=0.492, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0006463-1` (score=0.7000) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include...
2. `medquad_ans_0001043-1` (score=0.6904) - Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). In affected individuals, harmful amounts of lipids accumulate in the spleen, liver, ...
3. `medquad_ans_0006463-2` (score=0.6341) - What are the signs and symptoms of Wolman disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Wolman disease. If the information is available, the table below includes how often...
4. `medquad_ans_0006463-3` (score=0.6166) - How can I find additional comprehensive information on the treatment of Wolman disease? You can find relevant journal articles on Wolman syndrome and its treatment through a service called PubMed, a searchable databas...
5. `medquad_ans_0001043-3` (score=0.5798) - Mutations in the LIPA gene cause Wolman disease. The LIPA gene provides instructions for producing an enzyme called lysosomal acid lipase. This enzyme is found in the lysosomes (compartments that digest and recycle ma...

**Answer**:

Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). It is a type of lysosomal storage disorder caused by mutations in the LIPA gene and inherited in an autosomal recessive manner. The condition leads to a buildup of lipids in various organs, including the spleen, liver, bone marrow, small intestine, adrenal glands, and lymph nodes, along with calcium deposits in the adrenal glands. Infants are typically healthy at birth but soon develop symptoms such as enlarged liver and spleen (hepatosplenomegaly), poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor absorption of nutrients from food. The disease is severe and life-threatening, with most affected children not surviving past early childhood. New therapies, such as bone marrow transplantation and enzyme replacement therapy, are being explored to improve outcomes [medquad

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002753-3` (score=0.7000) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma a...
2. `medquad_ans_0002753-1` (score=0.6706) - Hemangiopericytoma is a term used to described a group of tumors that are derived from pericytes, the cells normally arranged along specific types of blood vessels called capillaries and venules. These types of tumors...
3. `medquad_ans_0004053-3` (score=0.4679) - What causes microscopic polyangiitis (MPA)? The cause of MPA is unknown. It is not contagious, does not usually run in families, and is not a form of cancer. The immune system is thought to play a critical role in the...
4. `medquad_ans_0001853-3` (score=0.0765) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0002750-1` (score=0.0669) - The term hemangioendothelioma describes several types of vascular neosplasms and includes both non-cancerous (benign) and cancerous (malignant) growths. The term has also been applied to those that show "borderline" b...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

### Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Scores**: context_relevance=1.000, groundedness=0.111, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0003520-3` (score=0.7000) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. Th...
2. `medquad_ans_0005069-3` (score=0.5141) - What genes are related to Hutchinson-Gilford progeria syndrome? Mutations in the LMNA gene cause Hutchinson-Gilford progeria syndrome. The LMNA gene provides instructions for making a protein called lamin A. This prot...
3. `medquad_ans_0004086-3` (score=0.3216) - What causes mitochondrial genetic disorders? Mitochondrial genetic disorders can be caused by changes (mutations) in either the mitochondrial DNA or nuclear DNA that lead to dysfunction of the mitochondria. Most DNA (...
4. `medquad_ans_0000593-3` (score=0.0889) - Mutations in at least two genes on the X chromosome are thought to be responsible for Lenz microphthalmia syndrome. Only one of these genes, BCOR, has been identified. The BCOR gene provides instructions for making a ...
5. `medquad_ans_0000396-1__dup2` (score=0.0889) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...

**Answer**:

The provided context does not contain information about genetic changes related to microphthalmia. Therefore, a specific answer cannot be generated from the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Scores**: context_relevance=0.833, groundedness=0.982, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000409-1__dup1` (score=0.7000) - Gout is a common, painful form of arthritis. It causes swollen, red, hot and stiff joints. Gout happens when uric acid builds up in your body. Uric acid comes from the breakdown of substances called purines. Purines a...
2. `medquad_ans_0000028-1__dup4` (score=0.6634) - Sudden, Intense Joint Pain Gout is a form of arthritis that causes the sudden onset of intense pain and swelling in the joints, which also may be warm and red. Attacks frequently occur at night and can be triggered by...
3. `medquad_ans_0000028-3__dup3` (score=0.6439) - Attacks Usually Start at Night Gout is a form of arthritis that causes sudden onset of intense pain and swelling in the joints, which also may be warm and red. Gout typically attacks one joint at a time, and the attac...
4. `medquad_ans_0000028-13` (score=0.6335) - Gout is a form of arthritis that frequently affects joints in the lower part of the body such as the knees, ankles, or toes. The affected joint may become swollen, red, or warm. Attacks usually occur at night. Sometim...
5. `medquad_ans_0000028-10` (score=0.6210) - These risk factors are associated with gout. - Genetics. Many people with gout have a family history of the disease. - Gender and age. Gout is more common in men than in women and more common in adults than in childre...

**Answer**:

The symptoms of gout include swollen, red, hot, and stiff joints. Gout causes sudden onset of intense pain and swelling in the joints, which may also be warm and red. Attacks often occur at night and can be triggered by stressful events, alcohol, or drugs. Early attacks usually subside within 3 to 10 days, even without treatment [medquad_ans_0000409-1__dup1]. Gout frequently affects joints in the lower part of the body such as the ankles, heels, knees, or toes, and often first attacks the big toe [medquad_ans_0000028-1__dup4].

**Supported citations**: `medquad_ans_0000409-1__dup1`, `medquad_ans_0000028-1__dup4`
**Unsupported citations**: _none_

### Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Scores**: context_relevance=0.667, groundedness=0.075, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000307-1__dup1` (score=0.7000) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...
2. `medquad_ans_0002400-1` (score=0.6103) - Focal dystonia is a movement disorder that is localized to a specific part of the body. The dystonias are a group of movement problems characterized by involuntary, sustained muscle contractions, tremors, and other un...
3. `medquad_ans_0001903-1` (score=0.5536) - Dopa-responsive dystonia (DRD) is an inherited type of dystonia that typically begins during childhood but may begin in adolescence or adulthood. Depending on the specific type of DRD, specific symptoms can vary. Feat...
4. `medquad_ans_0000859-1` (score=0.5383) - Rapid-onset dystonia parkinsonism is a rare movement disorder. "Rapid-onset" refers to the abrupt appearance of signs and symptoms over a period of hours to days. Dystonia is a condition characterized by involuntary, ...
5. `medquad_ans_0000312-1` (score=0.4371) - Early-onset primary dystonia is a condition characterized by progressive problems with movement, typically beginning in childhood. Dystonia is a movement disorder that involves involuntary tensing of the muscles (musc...

**Answer**:

The provided context does not mention "Dystonia 8" or any specific subtype referred to as Dystonia 8. Therefore, the cause of Dystonia 8 cannot be determined from the given information. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Scores**: context_relevance=0.889, groundedness=0.931, answer_relevance=0.444

**Top docs**:

1. `medquad_ans_0003719-1` (score=0.7000) - Lysinuric protein intolerance is a metabolic disorder caused by the body's inability to digest and use the amino acids lysine, arginine, and ornithine. Because the body cannot effectively break down these amino acids,...
2. `medquad_ans_0000612-2` (score=0.6815) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unk...
3. `medquad_ans_0005111-1` (score=0.1881) - Protein C deficiency is a disorder that increases a person's risk to develop abnormal blood clots. The condition can be mild or severe. People with mild protein C deficiency are at risk for a type of clot called deep ...
4. `medquad_ans_0000612-1` (score=0.0889) - Lysinuric protein intolerance is a disorder caused by the body's inability to digest and use certain protein building blocks (amino acids), namely lysine, arginine, and ornithine. Because the body cannot effectively b...
5. `medquad_ans_0000612-5` (score=0.0689) - These resources address the diagnosis or management of lysinuric protein intolerance: - Gene Review: Gene Review: Lysinuric Protein Intolerance - Genetic Testing Registry: Lysinuric protein intolerance - MedlinePlus E...

**Answer**:

Lysinuric protein intolerance affects approximately 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations, the condition occurs less frequently, but the exact incidence is unknown [medquad_ans_0000612-2].

**Supported citations**: `medquad_ans_0000612-2`
**Unsupported citations**: _none_

### Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000170-10` (score=0.7000) - - Renal artery stenosis (RAS) is the narrowing of one or both renal arteries. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood vessel that carries blood from the heart ...
2. `medquad_ans_0000170-1__dup3` (score=0.6703) - Renal artery stenosis is the narrowing of one or both renal arteries. Renal means kidney and stenosis means narrowing. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood ...
3. `medquad_ans_0004850-1` (score=0.4692) - What are the signs and symptoms of Pheochromocytoma, childhood? The Human Phenotype Ontology provides the following list of signs and symptoms for Pheochromocytoma, childhood. If the information is available, the tabl...
4. `medquad_ans_0005299-2` (score=0.4143) - What are the signs and symptoms of renal nutcracker syndrome? The signs and symptoms of renal nutcracker syndrome and the disease severity can vary from person to person. Some affected people may be asymptomatic while...
5. `medquad_ans_0005294-1` (score=0.3428) - What are the signs and symptoms of Renal dysplasia-limb defects syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Renal dysplasia-limb defects syndrome. If the information is...

**Answer**:

The first symptoms of renal artery stenosis (RAS) are usually either high blood pressure or decreased kidney function, or both [medquad_ans_0000170-10]. RAS is often overlooked as a cause of high blood pressure [medquad_ans_0000170-10]. High blood pressure is defined as a systolic pressure of 140 or above or a diastolic pressure of 90 or above [medquad_ans_0004850-1].

**Supported citations**: `medquad_ans_0000170-10`, `medquad_ans_0004850-1`
**Unsupported citations**: _none_

### Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Scores**: context_relevance=0.900, groundedness=0.813, answer_relevance=0.660

**Top docs**:

1. `medquad_ans_0000712-5` (score=0.7000) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis -...
2. `medquad_ans_0000712-1` (score=0.6707) - Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis (NFJS/DPR) represents a rare type of ectodermal dysplasia, a group of about 150 conditions characterized by abnormal development of ectoder...
3. `medquad_ans_0000871-5` (score=0.3897) - These resources address the diagnosis or management of retinitis pigmentosa: - American Foundation for the Blind: Living with Vision Loss - Foundation Fighting Blindness: Treatment of Retinitis Pigmentosa - Gene Revie...
4. `medquad_ans_0004909-4` (score=0.3830) - How might pityriasis rubra pilaris be treated? Treatment of pityriasis rubra pilaris (PRP) is mainly based on reports of patients' experiences. No controlled trials have been done, so the effectiveness and safety of t...
5. `medquad_ans_0005502-5` (score=0.0889) - How might Schnitzler syndrome be treated? The treatment of Schnitzler syndrome is aimed at alleviating the signs and symptoms associated with the condition. The following medications have been used with variable succe...

**Answer**:

The provided context does not mention any specific treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathosa reticularis (NFJS/DPR). While it describes diagnostic features such as reticulate hyperpigmentation, palmoplantar keratoderma, nail abnormalities, and other associated conditions, there is no information on therapeutic interventions or management strategies for NFJS/DPR [medquad_ans_0000712-1]. Therefore, based on the given context, treatments for NFJS/DPR cannot be determined.

**Supported citations**: `medquad_ans_0000712-1`
**Unsupported citations**: _none_

### Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.7000) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0000353-5` (score=0.6518) - These resources address the diagnosis or management of familial hyperaldosteronism: - Genetic Testing Registry: Familial hyperaldosteronism type 1 - Genetic Testing Registry: Familial hyperaldosteronism type 3 - Hormo...
3. `medquad_ans_0000353-1` (score=0.6512) - Familial hyperaldosteronism is a group of inherited conditions in which the adrenal glands, which are small glands located on top of each kidney, produce too much of the hormone aldosterone. Aldosterone helps control ...
4. `medquad_ans_0000353-2` (score=0.6071) - The prevalence of familial hyperaldosteronism is unknown. Familial hyperaldosteronism type II appears to be the most common variety. All types of familial hyperaldosteronism combined account for fewer than 1 out of 10...
5. `medquad_ans_0000589-2__dup1` (score=0.5241) - What are the signs and symptoms of Autosomal recessive pseudohypoaldosteronism type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal recessive pseudohypoaldosteronism typ...

**Answer**:

The symptoms of Familial hyperaldosteronism type III include hypercalciuria (5%), metabolic acidosis (5%), polydipsia (5%), polyuria (5%), adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

**Supported citations**: `medquad_ans_0002259-1`
**Unsupported citations**: _none_

### Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Scores**: context_relevance=1.000, groundedness=0.952, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004530-2` (score=0.7000) - Is olivopontocerebellar atrophy inherited? Olivopontocerebellar atrophy (OPCA) may be associated with conditions that are inherited (genetic), or it may occur sporadically. Genetic forms of OPCA may be inherited in an...
2. `medquad_ans_0005761-3` (score=0.0911) - How is spinocerebellar ataxia type 11 inherited? SCA11 is inherited in an autosomal dominant manner. The rate of de novo mutations is not known. Each child of an individual with SCA11 has a 50% chance of inheriting th...
3. `medquad_ans_0005761-1` (score=0.0797) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal...
4. `medquad_ans_0004750-4` (score=0.0797) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is pass...
5. `medquad_ans_0000673-4` (score=0.0686) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.

**Answer**:

Yes, Spinocerebellar ataxia type 11 (SCA11) is inherited in an autosomal dominant manner. Each child of an individual with SCA11 has a 50% chance of inheriting the mutation [medquad_ans_0005761-3].

**Supported citations**: `medquad_ans_0005761-3`
**Unsupported citations**: _none_

### Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Scores**: context_relevance=1.000, groundedness=0.948, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000503-2` (score=0.7000) - Hypochromic microcytic anemia with iron overload is likely a rare disorder; at least five affected families have been reported in the scientific literature.
2. `medquad_ans_0000503-1` (score=0.7000) - Hypochromic microcytic anemia with iron overload is a condition that impairs the normal transport of iron in cells. Iron is an essential component of hemoglobin, which is the substance that red blood cells use to carr...
3. `medquad_ans_0000503-5` (score=0.6629) - These resources address the diagnosis or management of hypochromic microcytic anemia with iron overload: - Genetic Testing Registry: Hypochromic microcytic anemia with iron overload These resources from MedlinePlus of...
4. `medquad_ans_0003040-1` (score=0.6455) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload....
5. `medquad_ans_0005622-4` (score=0.4624) - How is sideroblastic anemia diagnosed? The principle feature of sideroblastic anemia is slowly progressive, mild, life-long anemia which often goes unnoticed. Symptoms of iron overload may lead to the discovery of thi...

**Answer**:

The symptoms of Hypochromic microcytic anemia with iron overload include pale skin (pallor), tiredness (fatigue), and slow growth. Additionally, there is an abnormality of the liver, which can impair its function over time, and this liver problem typically becomes apparent in adolescence or early adulthood [medquad_ans_0000503-1].

**Supported citations**: `medquad_ans_0000503-1`
**Unsupported citations**: _none_

### Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000698-1__dup2` (score=0.7000) - Pemphigus is an autoimmune disorder. If you have it, your immune system attacks healthy cells in your skin and mouth, causing blisters and sores. No one knows the cause. Pemphigus does not spread from person to person...
2. `medquad_ans_0000115-2` (score=0.0938) - Benign chronic pemphigus is a rare condition; its prevalence is unknown.
3. `medquad_ans_0000115-1` (score=0.0830) - Benign chronic pemphigus, often called Hailey-Hailey disease, is a rare skin condition that usually appears in early adulthood. The disorder is characterized by red, raw, and blistered areas of skin that occur most of...
4. `medquad_ans_0000784-4` (score=0.0830) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000115-5` (score=0.0726) - These resources address the diagnosis or management of benign chronic pemphigus: - American Osteopathic College of Dermatology - Genetic Testing Registry: Familial benign pemphigus These resources from MedlinePlus off...

**Answer**:

Yes, benign chronic pemphigus is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with this condition each carry one copy of the mutated gene but typically do not show signs or symptoms [medquad_ans_0000784-4]. This pattern of inheritance is associated with familial benign pemphigus, which is a form of benign chronic pemphigus [medquad_ans_0000115-5].

**Supported citations**: `medquad_ans_0000784-4`, `medquad_ans_0000115-5`
**Unsupported citations**: _none_

### Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.953, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000916-1__dup1` (score=0.7000) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
2. `medquad_ans_0000915-1__dup1` (score=0.6997) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
3. `medquad_ans_0000916-4__dup1` (score=0.6906) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is...
4. `medquad_ans_0000916-5__dup1` (score=0.0756) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene...
5. `medquad_ans_0004784-4` (score=0.0661) - How is Pelizaeus-Merzbacher disease inherited?

**Answer**:

Yes, Camurati-Engelmann disease is inherited in an autosomal dominant fashion. This means that having a mutation in only one copy of the responsible gene in each cell is enough to cause the features of the condition. In some cases, the mutation is inherited from an affected parent, and in other cases, it occurs for the first time in the individual, known as a de novo mutation [medquad_ans_0000916-1__dup1].

**Supported citations**: `medquad_ans_0000916-1__dup1`
**Unsupported citations**: _none_

### Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Scores**: context_relevance=0.700, groundedness=0.882, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0002895-1` (score=0.7000) - Histiocytosis-lymphadenopathy plus syndrome is a group of conditions with overlapping signs and symptoms that affect many parts of the body. This group of disorders includes H syndrome, pigmented hypertrichosis with i...
2. `medquad_ans_0000483-5` (score=0.0877) - These resources address the diagnosis or management of histiocytosis-lymphadenopathy plus syndrome: - Genetic Testing Registry: Histiocytosis-lymphadenopathy plus syndrome These resources from MedlinePlus offer inform...
3. `medquad_ans_0006446-4` (score=0.0776) - Is Williams syndrome inherited?
4. `medquad_ans_0000673-4` (score=0.0679) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
5. `medquad_ans_0000483-3` (score=0.0679) - Histiocytosis-lymphadenopathy plus syndrome is caused by mutations in the SLC29A3 gene, which provides instructions for making a protein called equilibrative nucleoside transporter 3 (ENT3). ENT3 belongs to a family o...

**Answer**:

The genetic changes related to histiocytosis-lymphadenopathy plus syndrome are mutations in the SLC29A3 gene [medquad_ans_0002895-1].

**Supported citations**: `medquad_ans_0002895-1`
**Unsupported citations**: _none_

### Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.956, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000261-4` (score=0.7000) - In most cases of cyclic vomiting syndrome, affected people have no known history of the disorder in their family. However, many affected individuals have a family history of related conditions, such as migraines, irri...
2. `medquad_ans_0004453-1` (score=0.7000) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0000646-4` (score=0.6122) - Menkes syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome)...
4. `medquad_ans_0004454-1` (score=0.6122) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004456-1` (score=0.5272) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

Yes, Noonan syndrome is typically inherited in an autosomal dominant manner, but many cases are due to a new mutation and are not inherited from an affected parent [medquad_ans_0004453-1].

**Supported citations**: `medquad_ans_0004453-1`
**Unsupported citations**: _none_

### Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Scores**: context_relevance=0.875, groundedness=0.068, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0001093-2` (score=0.7000) - 3-methylglutaconyl-CoA hydratase deficiency is a rare disorder; at least 20 cases have been reported in the scientific literature.
2. `medquad_ans_0001093-1` (score=0.7000) - 3-methylglutaconyl-CoA hydratase deficiency is an inherited condition that causes neurological problems. Beginning in infancy to early childhood, children with this condition often have delayed development of mental a...
3. `medquad_ans_0001093-3` (score=0.6805) - Mutations in the AUH gene cause 3-methylglutaconyl-CoA hydratase deficiency. This gene provides instructions for producing 3-methylglutaconyl-CoA hydratase, an enzyme that is involved in breaking down a protein buildi...
4. `medquad_ans_0001093-5` (score=0.6805) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Aci...
5. `medquad_ans_0001859-5` (score=0.5133) - How might dihydrolipoamide dehydrogenase deficiency be treated? There are currently no consensus recommendations for the management of dihydrolipoamide dehydrogenase (DLD) deficiency. Management can be hard because va...

**Answer**:

The provided context does not mention any treatments for 3-methylglutaconyl-CoA hydratase deficiency. [medquad_ans_0001093-1, medquad_ans_0001093-2, medquad_ans_0001093-3] describe the genetic basis, symptoms, and biochemical mechanisms of the disorder but do not include information on therapeutic interventions. Therefore, based on the given context, treatments cannot be determined.

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0001093-1, medquad_ans_0001093-2, medquad_ans_0001093-3`

### Example 61 (`0003000-1`)

**Question**: What is (are) Hyperkalemic periodic paralysis ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.7000) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0000494-5` (score=0.7000) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis ...
3. `medquad_ans_0000494-1` (score=0.6318) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to...
4. `medquad_ans_0000126-1__dup1` (score=0.6158) - Familial periodic paralyses are a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. They are characterized by episodes in which the affect...
5. `medquad_ans_0003000-2` (score=0.5838) - What are the signs and symptoms of Hyperkalemic periodic paralysis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hyperkalemic periodic paralysis. If the information is available, ...

**Answer**:

Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms and legs. Episodes tend to increase in frequency until mid-adulthood, after which they occur less frequently. Factors that can trigger attacks include rest after exercise, potassium-rich foods such as bananas and potatoes, stress, fatigue, alcohol, pregnancy, exposure to cold temperatures, certain medications, and periods without food (fasting). Muscle strength usually returns to normal between attacks, although many affected people continue to experience mild stiffness (myotonia), particularly in muscles of the face and hands. Most people with hyperkalemic periodic paralysis have increased levels of potassium in their blood (hyperkalemia) during attacks, which results when the weak or paralyzed muscles release potassium ions into the bloodstream. In other cases, attacks are associated with normal blood potassium levels

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Scores**: context_relevance=0.800, groundedness=0.831, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000225-5` (score=0.7000) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of...
2. `medquad_ans_0000225-1` (score=0.7000) - Congenital fibrosis of the extraocular muscles is a disorder that affects the muscles that surround the eyes. These muscles control eye movement and the position of the eyes (for example, looking straight ahead). Cong...
3. `medquad_ans_0000225-2` (score=0.6431) - CFEOM1 is the most common form of congenital fibrosis of the extraocular muscles, affecting at least 1 in 230,000 people. CFEOM1 and CFEOM3 have been reported worldwide, whereas CFEOM2 has been seen in only a few fami...
4. `medquad_ans_0000224-1` (score=0.3936) - Congenital fiber-type disproportion is a condition that primarily affects skeletal muscles, which are muscles used for movement. People with this condition typically experience muscle weakness (myopathy), particularly...
5. `medquad_ans_0000225-4` (score=0.0679) - The different types of congenital fibrosis of the extraocular muscles have different patterns of inheritance. CFEOM1 and CFEOM3 are inherited in an autosomal dominant pattern, which means one copy of the altered gene ...

**Answer**:

The provided context does not specify treatments for congenital fibrosis of the extraocular muscles. While it mentions resources related to diagnosis and management, such as diagnostic tests, genetic counseling, drug therapy, surgery and rehabilitation, palliative care, and extraocular muscle function testing, it does not detail specific therapeutic interventions for this condition [medquad_ans_0000225-5]. Therefore, based on the available information, the treatments for congenital fibrosis of the extraocular muscles cannot be determined.

**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

### Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Scores**: context_relevance=0.800, groundedness=0.529, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000491-1__dup2` (score=0.7000) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms...
2. `medquad_ans_0000489-5` (score=0.6255) - These resources address the diagnosis or management of Huntington disease: - Gene Review: Gene Review: Huntington Disease - Genetic Testing Registry: Huntington's chorea - Huntington's Disease Society of America: HD C...
3. `medquad_ans_0000075-3__dup1` (score=0.5898) - The prognosis for individuals with chorea varies depending on the type of chorea and the associated disease. Huntington's disease is a progressive, and ultimately, fatal disease. Syndenham's chorea is treatable and cu...
4. `medquad_ans_0000152-1__dup1` (score=0.5834) - Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. Symptoms of the disease, which gets progressiv...
5. `medquad_ans_0000075-1__dup2` (score=0.5199) - Chorea is an abnormal involuntary movement disorder, one of a group of neurological disorders called dyskinesias, which are caused by overactivity of the neurotransmitter dopamine in the areas of the brain that contro...

**Answer**:

Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. It is caused by a mutation in the gene for a protein called huntingtin, where the cytosine, adenine, and guanine (CAG) building blocks of DNA repeat many more times than is normal. Symptoms, which progressively worsen, include uncontrolled movements (called chorea), abnormal body postures, changes in behavior, emotion, judgment, and cognition, impaired coordination, slurred speech, and difficulty feeding and swallowing. HD typically begins between ages 30 and 50, with an earlier onset form called juvenile HD occurring under age 20. People are born with the defective gene, but symptoms usually do not appear until middle age. The disease is progressive and ultimately fatal. There is no cure; medicines can help manage symptoms but cannot

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.556

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.7000) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
2. `medquad_ans_0000416-1` (score=0.6807) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with ...
3. `medquad_ans_0002601-1` (score=0.6710) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...
4. `medquad_ans_0000416-3` (score=0.6643) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
5. `medquad_ans_0002601-2` (score=0.6628) - What are the signs and symptoms of Glutamate formiminotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Glutamate formiminotransferase deficiency. If the inform...

**Answer**:

Approximately 20 affected individuals have been identified with glutamate formiminotransferase deficiency [medquad_ans_0000416-2].

**Supported citations**: `medquad_ans_0000416-2`
**Unsupported citations**: _none_

### Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.750, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000738-1` (score=0.7000) - Noonan syndrome is a condition that affects many areas of the body. It is characterized by mildly unusual facial characteristics, short stature, heart defects, bleeding problems, skeletal malformations, and many other...
2. `medquad_ans_0004450-4` (score=0.6246) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
3. `medquad_ans_0004451-3` (score=0.6219) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004453-3` (score=0.6196) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004454-3` (score=0.6179) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not specify the number of people affected by Noonan syndrome. [medquad_ans_0000738-1] describes the symptoms and characteristics of the condition but does not provide population statistics or prevalence rates. Therefore, the number of people affected by Noonan syndrome cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000738-1`
**Unsupported citations**: _none_

### Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Scores**: context_relevance=0.571, groundedness=0.989, answer_relevance=0.571

**Top docs**:

1. `medquad_ans_0000339-5__dup1` (score=0.7000) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on t...
2. `medquad_ans_0000339-1__dup3` (score=0.6523) - Also called crab lice or "crabs," pubic lice are parasitic insects found primarily in the pubic or genital area of humans. Pubic lice infestation is found worldwide and occurs in all races, ethnic groups, and levels o...
3. `medquad_ans_0000339-2__dup1` (score=0.6518) - Pubic ("crab") lice infestation is found worldwide and occurs in all races and ethnic groups and in all levels of society. Pubic lice usually are spread through sexual contact and are most common in adults. Occasional...
4. `medquad_ans_0000339-7` (score=0.6428) - Pubic ("crab") lice most commonly are spread directly from person to person by sexual contact. Pubic lice very rarely may be spread by clothing, bedding, or a toilet seat. The following are steps that can be taken to ...
5. `medquad_ans_0000557-1__dup2` (score=0.6248) - Summary : Lice are parasitic insects that can be found on people's heads and bodies. They survive by feeding on human blood. Lice found on each area of the body are different from each other. The three types of lice t...

**Answer**:

Pubic lice infestation is diagnosed by finding a “crab” louse or eggs (nits) on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp). Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

### Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.925, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005571-3` (score=0.7000) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
2. `medquad_ans_0005571-2` (score=0.6809) - What are the signs and symptoms of Shapiro syndrome? Shapiro syndrome generally consists of three findings: spontaneous periodic hypothermia, excessive sweating, and agenesis of the corpus callosum. However, there has...
3. `medquad_ans_0005571-1` (score=0.6802) - Shapiro syndrome is a rare disease affecting about 50 people worldwide that is typically characterized by recurrent episodes of excessive sweating and hypothermia and the agenesis of the corpus callosum. The duration ...
4. `medquad_ans_0003845-5` (score=0.3713) - How might McCune Albright syndrome be treated? Although there is no cure for McCune Albright syndrome (MAS), drug treatments may help some of the endocrine symptoms, and surgery can help repair some of the bone proble...
5. `medquad_ans_0004155-5` (score=0.3711) - What treatment is available for mucopolysaccharidosis I (MPS I)? The two main treatments for MPS I are enzyme replacement therapy (ERT) and bone marrow transplant. Both of these treatments work by replacing the missin...

**Answer**:

Treatment options for Shapiro syndrome that have been attempted include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, and sympathectomy. The effectiveness of these treatments varies among individuals. It is recommended that treatment options be discussed with a healthcare provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

### Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.964, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002359-1` (score=0.7000) - FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. People wi...
2. `medquad_ans_0000373-1` (score=0.6625) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects ...
3. `medquad_ans_0000373-2` (score=0.6538) - The prevalence of FG syndrome is unknown, although several hundred cases have been reported worldwide. Researchers suspect that FG syndrome may be overdiagnosed because many of its signs and symptoms are also seen wit...
4. `medquad_ans_0000373-5` (score=0.6532) - These resources address the diagnosis or management of FG syndrome: - Gene Review: Gene Review: MED12-Related Disorders - Genetic Testing Registry: FG syndrome - Genetic Testing Registry: FG syndrome 2 - Genetic Testi...
5. `medquad_ans_0002359-3` (score=0.6278) - How might FG syndrome be treated? Treatment is aimed at addressing the individual symptoms present in each case. This often involves care by a team of providers which may include pediatricians, neurologists, cardiolog...

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. It is characterized by intellectual disability ranging from mild to severe, hypotonia, constipation and/or anal anomalies, a distinctive facial appearance, broad thumbs and great toes, relative macrocephaly, and abnormalities of the corpus callosum. Medical problems such as heart defects, seizures, undescended testicles, and inguinal hernia have also been reported. The condition is inherited in an X-linked recessive pattern, and mutations in genes such as MED12 (FGS1), FLNA (FGS2), CASK (FGS4), UPF3B (FGS6), and BRWD3 (FGS7) are associated with the disorder [medquad_ans_0002359-1]. Affected individuals often

**Supported citations**: `medquad_ans_0002359-1`
**Unsupported citations**: _none_

### Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006242-1` (score=0.7000) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0005488-1` (score=0.4577) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
3. `medquad_ans_0000225-1` (score=0.0877) - Congenital fibrosis of the extraocular muscles is a disorder that affects the muscles that surround the eyes. These muscles control eye movement and the position of the eyes (for example, looking straight ahead). Cong...
4. `medquad_ans_0001335-1` (score=0.0776) - What are the signs and symptoms of Chudley Rozdilsky syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Chudley Rozdilsky syndrome. If the information is available, the table ...
5. `medquad_ans_0000225-4` (score=0.0679) - The different types of congenital fibrosis of the extraocular muscles have different patterns of inheritance. CFEOM1 and CFEOM3 are inherited in an autosomal dominant pattern, which means one copy of the altered gene ...

**Answer**:

The symptoms of Tukel syndrome include carpal bone aplasia, carpal synostosis, compensatory chin elevation, congenital fibrosis of extraocular muscles, nonprogressive restrictive external ophthalmoplegia, postaxial oligodactyly, ptosis, restrictive external ophthalmoplegia, syndactyly [medquad_ans_0006242-1].

**Supported citations**: `medquad_ans_0006242-1`
**Unsupported citations**: _none_

### Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Scores**: context_relevance=1.000, groundedness=0.807, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000199-1__dup1` (score=0.7000) - The mucolipidoses (ML) are a group of inherited metabolic diseases that affect the bodys ability to carry out the normal turnover of various materials within cells. In ML, abnormal amounts of carbohydrates and fatty m...
2. `medquad_ans_0004155-5` (score=0.3558) - What treatment is available for mucopolysaccharidosis I (MPS I)? The two main treatments for MPS I are enzyme replacement therapy (ERT) and bone marrow transplant. Both of these treatments work by replacing the missin...
3. `medquad_ans_0000674-5` (score=0.0697) - These resources address the diagnosis or management of mucolipidosis II alpha/beta: - Gene Review: Gene Review: Mucolipidosis II - Genetic Testing Registry: I cell disease - MedlinePlus Encyclopedia: Clubfoot - Medlin...
4. `medquad_ans_0003103-4` (score=0.0697) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...
5. `medquad_ans_0000108-2__dup1` (score=0.0601) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The provided context does not specify treatments for mucolipidoses. While it details the pathophysiology and genetic basis of mucolipidoses, including their classification as lysosomal storage diseases and the accumulation of carbohydrates and lipids due to enzyme deficiencies, no specific treatments are mentioned.

Although the context discusses enzyme replacement therapy and bone marrow transplant for mucopolysaccharidosis I (MPS I), which is a different condition from mucolipidoses, this information cannot be directly applied to mucolipidoses. Therefore, based on the available information, the treatments for mucolipidoses are not available.

[medquad_ans_0000199-1__dup1]

**Supported citations**: `medquad_ans_0000199-1__dup1`
**Unsupported citations**: _none_

### Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0005084-2` (score=0.7000) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis...
2. `medquad_ans_0001157-1` (score=0.6815) - What are the signs and symptoms of Cholestasis, progressive familial intrahepatic 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Cholestasis, progressive familial intrahepatic 4....
3. `medquad_ans_0000118-1` (score=0.6803) - Benign recurrent intrahepatic cholestasis (BRIC) is characterized by episodes of liver dysfunction called cholestasis. During these episodes, the liver cells have a reduced ability to release a digestive fluid called ...
4. `medquad_ans_0005084-1` (score=0.6612) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching...
5. `medquad_ans_0000829-5` (score=0.6392) - These resources address the diagnosis or management of progressive familial intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Progressive familial intrahepatic cholest...

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 include cirrhosis, conjugated hyperbilirubinemia, death in childhood, diarrhea, elevated alkaline phosphatase, failure to thrive, fat malabsorption, hepatocellular carcinoma, hepatomegaly, infantile onset, intermittent jaundice, intrahepatic cholestasis, pruritus, short stature, and splenomegaly [medquad_ans_0005084-2].

**Supported citations**: `medquad_ans_0005084-2`
**Unsupported citations**: _none_

### Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Scores**: context_relevance=0.800, groundedness=0.958, answer_relevance=0.480

**Top docs**:

1. `medquad_ans_0000211-6` (score=0.7000) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he i...
2. `medquad_ans_0000070-7` (score=0.6726) - Urinary incontinence means a person leaks urine by accident. Urinary incontinence is a common bladder problem as people age. Women are more likely than men to leak urine. If this problem is happening to you, there is ...
3. `medquad_ans_0000070-1__dup5` (score=0.6411) - Urinary incontinence means a person leaks urine by accident. Urinary incontinence is a common bladder problem as people age. Women are more likely than men to leak urine. If this problem is happening to you, there is ...
4. `medquad_ans_0000070-8` (score=0.5941) - There are different types of urinary incontinence. Stress urinary incontinence happens when urine leaks as pressure is put on the bladder, for example, during exercise, coughing, sneezing, laughing, or lifting heavy o...
5. `medquad_ans_0000213-4__dup2` (score=0.4554) - Urinary retention in men becomes more common with age. - In men 40 to 83 years old, the overall incidence of urinary retention is 4.5 to 6.8 per 1,000 men.2 - For men in their 70s, the overall incidence increases to 1...

**Answer**:

Urinary incontinence occurs in 11 to 34 percent of older men [medquad_ans_0000211-6]. Two to 11 percent of older men report daily urinary incontinence [medquad_ans_0000211-6].

**Supported citations**: `medquad_ans_0000211-6`
**Unsupported citations**: _none_

### Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.527, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000882-1` (score=0.7000) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
2. `medquad_ans_0005415-1` (score=0.6906) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
3. `medquad_ans_0005415-4` (score=0.6827) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...
4. `medquad_ans_0000882-4` (score=0.6658) - As discussed above, Robinow syndrome can have either an autosomal recessive or an autosomal dominant pattern of inheritance. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. T...
5. `medquad_ans_0000882-2` (score=0.6544) - Both the autosomal recessive and autosomal dominant forms of Robinow syndrome are rare. Fewer than 200 people with autosomal recessive Robinow syndrome have been described in the medical literature. This form of the c...

**Answer**:

Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. It is characterized by two major types: autosomal recessive and autosomal dominant. Autosomal recessive Robinow syndrome is marked by skeletal abnormalities such as shortening of the long bones in the arms and legs, abnormally short fingers and toes (brachydactyly), wedge-shaped spinal bones leading to kyphoscoliosis, fused or missing ribs, and short stature. Affected individuals also have distinctive facial features, including a broad forehead, widely spaced eyes, a short nose with an upturned tip, a wide nasal bridge, and a broad, triangle-shaped mouth, which are sometimes described as "fetal facies." Other features may include underdeveloped genitalia, dental problems (such as crowded teeth and overgrown gums), kidney or heart defects, and delayed development in 10 to 15 percent of cases

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 74 (`0000551-1`)

**Question**: What is (are) juvenile primary lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.456, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003331-1` (score=0.7000) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
2. `medquad_ans_0000551-2` (score=0.6798) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.
3. `medquad_ans_0003331-3` (score=0.6627) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...
4. `medquad_ans_0000551-1` (score=0.6505) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. The features of this disorder are caused by damage to moto...
5. `medquad_ans_0000551-5` (score=0.6227) - These resources address the diagnosis or management of juvenile primary lateral sclerosis: - Gene Review: Gene Review: ALS2-Related Disorders - Genetic Testing Registry: Juvenile primary lateral sclerosis These resour...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness (spasticity) of muscles in the arms, legs, and face. It is caused by mutations in the ALS2 gene, which leads to the production of an unstable or nonfunctional protein called alsin. Alsin is abundant in motor neurons, and its dysfunction results in damage to these nerve cells, which control muscle movement. Symptoms begin in early childhood and progress slowly over 15 to 20 years, initially presenting as clumsiness, muscle weakness, and spasticity in the legs, with later progression including difficulty with balance, slurred speech, drooling, difficulty swallowing, and loss of ability to walk. The disorder is inherited in an autosomal recessive pattern [medquad_ans_0003331-1, medquad_ans_0003331-3, medquad_ans_000055

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Scores**: context_relevance=0.750, groundedness=0.907, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000040-1__dup3` (score=0.7000) - Leukemia is a cancer of the blood cells. It is the most common type of blood cancer and affects 10 times as many adults as children. Most people diagnosed with leukemia are over 50 years old. Leukemia Starts in Bone M...
2. `medquad_ans_0000188-1__dup2` (score=0.6207) - Leukemia is cancer of the white blood cells. It is the most common type of childhood cancer. Your blood cells form in your bone marrow. White blood cells help your body fight infection. In leukemia, the bone marrow pr...
3. `medquad_ans_0000001_6-1` (score=0.5449) - Key Points - Childhood acute lymphoblastic leukemia (ALL) is a type of cancer in which the bone marrow makes too many immature lymphocytes (a type of white blood cell). - Leukemia may affect red blood cells, white blo...
4. `medquad_ans_0000040-6` (score=0.5296) - Researchers are conducting clinical trials in many parts of the country. Clinical trials test an intervention such as a drug, therapy, medical device, or behavior in many people to see if it is safe and effective. Cli...
5. `medquad_ans_0000010-1__dup4` (score=0.5279) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. Thes...

**Answer**:

Leukemia is a cancer of the blood cells, specifically originating in the bone marrow, where it produces abnormal white blood cells called leukemia cells. These abnormal cells initially function almost normally but, over time, crowd out healthy blood cells, impairing the blood's ability to carry out its normal functions such as fighting infection, carrying oxygen, and controlling bleeding. It is the most common type of blood cancer and affects 10 times as many adults as children, with most diagnoses occurring in individuals over 50 years old. There are four common types: chronic lymphocytic leukemia, chronic myeloid leukemia, acute myeloid leukemia, and acute lymphocytic leukemia, of which chronic lymphocytic leukemia is the most common in older adults and acute lymphocytic leukemia is more common in children [medquad_ans_0000040-1__dup3].

**Supported citations**: `medquad_ans_0000040-1__dup3`
**Unsupported citations**: _none_

### Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002485-2` (score=0.7000) - What are the signs and symptoms of gangliocytomas? Signs and symptoms caused by the presence of a gangliocytoma can vary depending on the tumor's location. Seizures are the most common symptom. Other symptoms may incl...
2. `medquad_ans_0002485-1` (score=0.0990) - Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. Gangliocytomas may occur in all age groups but most often occur in people between the ages of 10 and 30. The most common si...
3. `medquad_ans_0000135-1__dup1` (score=0.0990) - The gangliosidoses are a group of inherited metabolic diseases caused by a deficiency of the different proteins needed to break down fatty substances called lipids. Excess buildup of these fatty materials (oils, waxes...
4. `medquad_ans_0004525-1` (score=0.0876) - Oligoastrocytoma is a brain tumor that forms when two types of cells in the brain, called oligodendrocytes and astrocytes, rapidly increase in number to form a mass. These brain cells are known as glial cells, which n...
5. `medquad_ans_0000737-1` (score=0.0767) - Paraganglioma is a type of noncancerous (benign) tumor that occurs in structures called paraganglia. Paraganglia are groups of cells that are found near nerve cell bunches called ganglia. Paragangliomas are usually fo...

**Answer**:

Gangliocytoma is a rare type of central nervous system (CNS) tumor composed of mature neurons. It may occur in all age groups but most commonly affects individuals between the ages of 10 and 30. The most frequent site is the temporal lobe of the brain, although it can arise anywhere in the CNS, including the cerebellum, brainstem, floor of the third ventricle, and spinal cord. Gangliocytomas are among the most frequent tumors associated with epilepsy. Signs and symptoms vary depending on the tumor's location and may include seizures (most common), increased brain pressure, endocrine disorders, and focal symptoms. These tumors are generally slow-growing and usually do not become malignant. Treatment typically involves surgical removal of the tumor [medquad_ans_0002485-1].

**Supported citations**: `medquad_ans_0002485-1`
**Unsupported citations**: _none_

### Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.488, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000130-1__dup1` (score=0.7000) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of ...
2. `medquad_ans_0000130-2__dup1` (score=0.6621) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportiv...
3. `medquad_ans_0004068-1` (score=0.6284) - Miller syndrome is a rare condition that mainly affects the development of the face and limbs. Characteristic features include underdeveloped cheek bones, a very small lower jaw, cleft lip and/or palate, abnormalities...
4. `medquad_ans_0000130-3__dup1` (score=0.6271) - The prognosis for most individuals with Miller Fisher syndrome is good. In most cases, recovery begins within 2 to 4 weeks of the onset of symptoms, and may be almost complete within 6 months. Some individuals are lef...
5. `medquad_ans_0000658-2` (score=0.6112) - Miller syndrome is a rare disorder; it is estimated to affect fewer than 1 in 1 million newborns. At least 30 cases have been reported in the medical literature.

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, absence of tendon reflexes, generalized muscle weakness, and may be preceded by a viral illness. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder. The prognosis for most individuals is good, with recovery beginning within 2 to 4 weeks of symptom onset and often being almost complete within 6 months, although some may have residual deficits and rare relapses (less than 3 percent of cases). Treatment is identical to that for Guillain-Barr syndrome, including intravenous immunoglobulin (IVIg) or plasmapheresis and supportive care [medquad_ans_0000130-1**dup1, medquad_ans_0000130-2**dup1, medquad

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0001084-1` (score=0.7000) - 22q11.2 deletion syndrome (which is also known by several other names, listed below) is a disorder caused by the deletion of a small piece of chromosome 22. The deletion occurs near the middle of the chromosome at a l...
2. `medquad_ans_0006570-3` (score=0.7000) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location desig...
3. `medquad_ans_0001084-3` (score=0.5511) - Most people with 22q11.2 deletion syndrome are missing a sequence of about 3 million DNA building blocks (base pairs) on one copy of chromosome 22 in each cell. This region contains 30 to 40 genes, many of which have ...
4. `medquad_ans_0006579-3` (score=0.5053) - What causes 2q37 deletion syndrome? 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at ...
5. `medquad_ans_0006562-3` (score=0.3328) - What causes 17q23.2q23.2 microdeletion syndrome? The syndrome is caused by an interstitial deletion (a deletion that does not involve the ends of a chromosome) encompassing bands 23.1 to 23.2 on the long (q) arm of ch...

**Answer**:

22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in the region designated q11.2 [medquad_ans_0001084-1].

**Supported citations**: `medquad_ans_0001084-1`
**Unsupported citations**: _none_

### Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0000148-1__dup2` (score=0.7000) - Shingles (herpes zoster) is an outbreak of rash or blisters on the skin that is caused by the same virus that causes chickenpox the varicella-zoster virus. The first sign of shingles is often burning or tingling pain ...
2. `medquad_ans_0000062-1__dup5` (score=0.6806) - Shingles is a painful rash that develops on one side of the face or body. The rash forms blisters that typically scab over in 7 to 10 days and clear up within 2 to 4 weeks. Most commonly, the rash occurs in a single s...
3. `medquad_ans_0000062-2__dup3` (score=0.5667) - Caused By A Virus Shingles is caused by a virus called varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores healed and the other symptoms went a...
4. `medquad_ans_0000812-1__dup2` (score=0.5338) - Shingles is a disease caused by the varicella-zoster virus - the same virus that causes chickenpox. After you have chickenpox, the virus stays in your body. It may not cause problems for many years. As you get older, ...
5. `medquad_ans_0000149-1__dup1` (score=0.4378) - Herpes zoster oticus, also called Ramsay Hunt Syndrome or Ramsay Hunt Syndrome type II, is a common complication of shingles. Shingles is an infection caused by the varicella-zoster virus, which is the virus that caus...

**Answer**:

Shingles is caused by a reactivation of the chickenpox virus. The virus, known as the varicella-zoster virus, remains in the body for life after the initial chickenpox infection and stays inactive until the immune system weakens [medquad_ans_0000062-1__dup5].

**Supported citations**: `medquad_ans_0000062-1__dup5`
**Unsupported citations**: _none_

### Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.849, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002456-1` (score=0.7000) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the ta...
2. `medquad_ans_0000333-1` (score=0.6016) - Fabry disease is an inherited disorder that results from the buildup of a particular type of fat, called globotriaosylceramide, in the body's cells. Beginning in childhood, this buildup causes signs and symptoms that ...
3. `medquad_ans_0000392-1` (score=0.5386) - Fryns syndrome is a condition that affects the development of many parts of the body. The features of this disorder vary widely among affected individuals and overlap with the signs and symptoms of several other disor...
4. `medquad_ans_0002200-2` (score=0.5375) - What are the signs and symptoms of Fabry disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Fabry disease. If the information is available, the table below includes how often t...
5. `medquad_ans_0002458-2` (score=0.4878) - What are the signs and symptoms of Fryns syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns syndrome. If the information is available, the table below includes how often...

**Answer**:

The provided context does not describe "Fryns Hofkens Fabry syndrome" as a valid or recognized medical condition. Instead, the context separately describes Fryns syndrome and Fabry disease, which are distinct disorders.

- Fryns syndrome is a developmental disorder with features such as ulnar deviation of fingers, distal ulnar hypoplasia, dysplastic radii, and other skeletal abnormalities [medquad_ans_0002456-1].
- Fabry disease is an inherited disorder characterized by pain in hands and feet, angiokeratomas, hypohidrosis, corneal opacity, gastrointestinal issues, tinnitus, hearing loss, and complications like kidney damage, heart attack, and stroke [medquad_ans_0000333-1].

There is no evidence in the provided context that these two conditions are combined into a single entity called "Fryns Hofkens Fabry syndrome."

**Supported citations**: `medquad_ans_0002456-1`, `medquad_ans_0000333-1`
**Unsupported citations**: _none_

### Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Scores**: context_relevance=0.900, groundedness=0.919, answer_relevance=0.740

**Top docs**:

1. `medquad_ans_0000050-8` (score=0.7000) - Scientists are pursuing a wide range of basic and clinical studies on osteoporosis. Significant advances in preventing and treating osteoporosis continue to be made. Such advances are the direct result of research foc...
2. `medquad_ans_0000030_1-7` (score=0.7000) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000007_2-6` (score=0.6122) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000552-5` (score=0.6122) - These resources address the diagnosis or management of juvenile primary osteoporosis: - Lucile Packard Children's Hospital at Stanford: Juvenile Osteoporosis - MedlinePlus Encyclopedia: Bone Mineral Density Test - Mer...
5. `medquad_ans_0000050-34` (score=0.5272) - A comprehensive osteoporosis treatment program includes a focus on proper nutrition, exercise, and safety issues to prevent falls that may result in fractures. In addition, your doctor may prescribe a medication to sl...

**Answer**:

Research on osteoporosis is focused on determining the causes and consequences of bone loss at the cellular and tissue levels, assessing risk factors, developing new strategies to maintain and even enhance bone density, and reducing fracture risk. This research also explores the roles of genetics, hormones, calcium, vitamin D, drugs, and exercise on bone mass [medquad_ans_0000050-8]. Additionally, scientists are pursuing basic and clinical studies to advance prevention and treatment methods [medquad_ans_0000050-8]. For more information, the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH provides resources on ongoing research [medquad_ans_0000050-8].

Note: The provided context does not specifically mention clinical trials for osteoporosis, only that clinical trials are being conducted for other conditions such as multiple myeloma and plasma cell neop

**Supported citations**: `medquad_ans_0000050-8`
**Unsupported citations**: _none_

### Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Scores**: context_relevance=1.000, groundedness=0.888, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000430-1__dup1` (score=0.7000) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. ...
2. `medquad_ans_0000352-1__dup2` (score=0.5011) - Your family history includes health information about you and your close relatives. Families have many factors in common, including their genes, environment, and lifestyle. Looking at these factors can help you figure...
3. `medquad_ans_0000588-1__dup1` (score=0.3877) - Summary : The field of ethics studies principles of right and wrong. There is hardly an area in medicine that doesn't have an ethical aspect. For example, there are ethical issues relating to - End of life care: Shoul...
4. `medquad_ans_0000694-1__dup1` (score=0.3658) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of me...
5. `medquad_ans_0000693-1__dup2` (score=0.3411) - Summary : As a patient, you have certain rights. Some are guaranteed by federal law, such as the right to get a copy of your medical records, and the right to keep them private. Many states have additional laws protec...

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. It often promises quick fixes, miracle cures, ancient remedies, secret ingredients, or scientific breakthroughs for common health issues like weight loss, memory loss, sexual performance, and joint pain. These scams target individuals with serious conditions such as cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. Red flags include claims of a miracle cure or quick fix. Before taking unproven treatments, it is advised to consult a doctor or healthcare professional, especially when using prescription drugs. The Food and Drug Administration is mentioned as a source of information on this topic [medquad_ans_0000430-1__dup1].

**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

### Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Scores**: context_relevance=1.000, groundedness=0.940, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000054-1__dup4` (score=0.7000) - Diabetes problems are health problems that can happen when you have diabetes. If your diabetes is not under control, you will have too much glucose\*, also called sugar, in your blood. Having too much glucose in your b...
2. `medquad_ans_0000275-1__dup1` (score=0.5672) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
3. `medquad_ans_0000061-6__dup1` (score=0.5445) - People with diabetes can lower their risk of sexual and urologic problems by keeping their blood glucose, blood pressure, and cholesterol levels close to the target numbers their health care provider recommends. Being...
4. `medquad_ans_0000015-8` (score=0.5278) - Diabetes cannot be cured, but it can be managed. Managing blood glucose (blood sugar) as well as blood pressure and cholesterol is the best defense against the serious complications of diabetes. Know What To Do Every ...
5. `medquad_ans_0000009-3__dup3` (score=0.4598) - There are several things you can do to lower your risk for cataract. They include - having regular eye exams - quitting smoking - wearing sunglasses - taking care of other health problems - maintaining a healthy weigh...

**Answer**:

The symptoms of diabetic eye problems, such as diabetic retinopathy, can include blurry or double vision, rings, flashing lights, or blank spots, dark or floating spots, pain or pressure in one or both eyes, and trouble seeing things out of the corners of the eyes [medquad_ans_0000275-1__dup1]. These symptoms may not be noticeable at first, highlighting the importance of regular eye exams for people with diabetes to detect and treat issues early [medquad_ans_0000275-1__dup1].

**Supported citations**: `medquad_ans_0000275-1__dup1`
**Unsupported citations**: _none_

### Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.285, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0003743-4` (score=0.7000) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition w...
2. `medquad_ans_0000001-4` (score=0.7000) - Aarskog-Scott syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chro...
3. `medquad_ans_0000983-3__dup1` (score=0.6122) - Is Castleman disease inherited? Although the exact underlying cause of Castleman disease is unknown, it is thought to occur sporadically in people with no family history of the condition.
4. `medquad_ans_0000001-1` (score=0.6122) - Aarskog-Scott syndrome is a genetic disorder that affects the development of many parts of the body. This condition mainly affects males, although females may have mild features of the syndrome. People with Aarskog-Sc...
5. `medquad_ans_0000001-2` (score=0.5272) - Aarskog-Scott syndrome is believed to be a rare disorder; however, its prevalence is unknown because mildly affected people are often not diagnosed.

**Answer**:

Yes, Aarskog-Scott syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, and in males, one altered copy of the gene in each cell is sufficient to cause the condition. Females typically require mutations in both copies of the gene to be affected, making males more frequently affected than females. Females who carry one altered copy of the FGD1 gene may show mild signs of the condition [doc_id: medquad_ans_0000001-4].

**Supported citations**: _none_
**Unsupported citations**: `doc_id: medquad_ans_0000001-4`

### Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Scores**: context_relevance=0.900, groundedness=0.980, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0006525-1` (score=0.7000) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin. People with X-linked sideroblastic anemia have mature red blood cells that...
2. `medquad_ans_0001062-1` (score=0.6680) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin, which is the protein that carries oxygen in the blood. People with X-linke...
3. `medquad_ans_0001063-1` (score=0.6481) - X-linked sideroblastic anemia and ataxia is a rare condition characterized by a blood disorder called sideroblastic anemia and movement problems known as ataxia. This condition occurs only in males. Sideroblastic anem...
4. `medquad_ans_0001062-4` (score=0.5766) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
5. `medquad_ans_0001062-5` (score=0.5718) - These resources address the diagnosis or management of X-linked sideroblastic anemia: - Genetic Testing Registry: Hereditary sideroblastic anemia - MedlinePlus Encyclopedia: Anemia These resources from MedlinePlus off...

**Answer**:

The genetic changes related to X-linked sideroblastic anemia are mutations in the ALAS2 gene. In rare cases, mutations are also found in both the HFE gene and the ALAS2 gene, resulting in a more severe form of the disorder [medquad_ans_0006525-1].

**Supported citations**: `medquad_ans_0006525-1`
**Unsupported citations**: _none_

### Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Scores**: context_relevance=0.833, groundedness=0.509, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0003601-1` (score=0.7000) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
2. `medquad_ans_0003604-1` (score=0.6956) - Limb-girdle muscular dystrophy type 2A (LGMD2A) is an autosomal recessive limb-girdle muscular dystrophy characterized by progressive, symmetrical weakness of the proximal limb and girdle muscles (mainly those around ...
3. `medquad_ans_0000602-1` (score=0.6925) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifi...
4. `medquad_ans_0000602-3` (score=0.6167) - The various forms of limb-girdle muscular dystrophy are caused by mutations in many different genes. These genes provide instructions for making proteins that are involved in muscle maintenance and repair. Some of the...
5. `medquad_ans_0003610-1` (score=0.5671) - Limb-girdle muscular dystrophy type 2C (LGMD2C) is a condition that affects the muscles and is caused by mutations in the gamma-sarcoglycan gene. This condition belongs to a group of muscle disorders called limb-girdl...

**Answer**:

Limb-girdle muscular dystrophy is a group of disorders that affect the voluntary muscles around the hips and shoulders, leading to progressive loss of muscle strength and bulk over time. The conditions primarily impact proximal muscles, including those in the shoulders, upper arms, pelvic area, and thighs. Onset may occur in childhood, adolescence, young adulthood, or later in life, and the severity, age of onset, and specific features vary among the different subtypes. Most forms are inherited in an autosomal recessive manner, although several rare forms follow an autosomal dominant pattern. Males and females are affected equally. There are at least 20 different types of limb-girdle muscular dystrophy, with limb-girdle muscular dystrophy type 2A (LGMD2A) being the most common, accounting for about 30 percent of cases and caused by mutations in the CAPN3 gene [medquad_ans_00

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Scores**: context_relevance=1.000, groundedness=0.912, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000016-6` (score=0.7000) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
2. `medquad_ans_0000275-1__dup1` (score=0.6698) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
3. `medquad_ans_0000016-2__dup2` (score=0.6616) - Who Is at Risk? All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes ...
4. `medquad_ans_0000016-1__dup3` (score=0.6597) - Can Cause Vision Loss, Blindness Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina in the back of the eye. A ...
5. `medquad_ans_0000016-8` (score=0.5879) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a com...

**Answer**:

Diabetic eye problems include diabetic retinopathy, cataracts, and glaucoma. Diabetic retinopathy is a condition that occurs when diabetes damages the tiny blood vessels in the retina, leading to potential vision loss. It is a leading cause of blindness in American adults and may present with symptoms such as blurry or double vision, rings or flashing lights, dark or floating spots, pain or pressure in the eyes, and trouble seeing out of the corners of the eyes. Cataracts involve a clouding of the lens of the eye, which can be treated with surgery. Glaucoma occurs when pressure builds up in the eye, damaging the main nerve, and can be managed with eye drops or surgery. People with diabetes are at increased risk for all three conditions and should have a comprehensive dilated eye exam at least once a year [medquad_ans_0000275-1__dup1]medquad_ans_0

**Supported citations**: `medquad_ans_0000275-1__dup1`
**Unsupported citations**: _none_

### Example 88 (`0006449-1`)

**Question**: What is (are) Wilson disease ?

**Scores**: context_relevance=1.000, groundedness=0.425, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006449-1` (score=0.7000) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver,...
2. `medquad_ans_0006449-3` (score=0.6707) - What causes Wilson disease? Wilson disease is caused by changes (mutations) in the ATP7B gene. This gene encodes a protein that plays an important role in the transport of copper from the liver to the rest of the body...
3. `medquad_ans_0001037-2` (score=0.6638) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
4. `medquad_ans_0006449-5` (score=0.6342) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lif...
5. `medquad_ans_0000133-1__dup3` (score=0.6337) - Wilson disease is a genetic disease that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filt...

**Answer**:

Wilson disease is a rare inherited disorder characterized by the accumulation of copper in the body due to mutations in the ATP7B gene. This gene encodes a protein responsible for transporting copper from the liver to other parts of the body and removing excess copper from the body. When the protein does not function properly, copper builds up in tissues and organs such as the liver, brain, and eyes, leading to damage. The disorder can cause chronic liver disease, central nervous system abnormalities, and psychiatric disturbances. It is inherited in an autosomal recessive manner and affects approximately 1 in 30,000 individuals. Although there is no cure, treatments exist to reduce or control copper accumulation, often involving lifelong medication, dietary modifications, and, in some cases, liver transplantation if liver failure occurs [medquad*ans_0006449-1, medquad_ans_0006449-3, medquad_ans*

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 89 (`0001503-2`)

**Question**: What are the symptoms of Congenital dyserythropoietic anemia type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.971, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001504-1` (score=0.7000) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital dyserythropoietic anemia type 3. If the info...
2. `medquad_ans_0001503-2` (score=0.6903) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to abs...
3. `medquad_ans_0001503-1` (score=0.6502) - Congenital dyserythropoietic anemia type 2 (CDA II) is an inherited blood disorder characterized by mild to severe anemia. It is usually diagnosed in adolescence or early adulthood. Many affected individuals have yell...
4. `medquad_ans_0000307-1` (score=0.6050) - Dyserythropoietic anemia and thrombocytopenia is a condition that affects blood cells and primarily occurs in males. A main feature of this condition is a type of anemia called dyserythropoietic anemia, which is chara...
5. `medquad_ans_0003748-2` (score=0.5850) - What are the signs and symptoms of Majeed syndrome? Majeed syndrome is characterized by recurrent episodes of fever and inflammation in the bones and skin. There are two main features of Majeed syndrome: Chronic recur...

**Answer**:

The symptoms of Congenital dyserythropoietic anemia type 2 include jaundice, gallstones, an enlarged liver and spleen, and iron overload, which can lead to complications such as an abnormal heart rhythm (arrhythmia), congestive heart failure, diabetes, and chronic liver disease (cirrhosis). Rarely, mediastinal tumors may occur. Blood transfusions may be necessary during pregnancy or other special circumstances like anemic crisis, major surgery, or infections [medquad_ans_0001503-2].

**Supported citations**: `medquad_ans_0001503-2`
**Unsupported citations**: _none_

### Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.937, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000038-3__dup2` (score=0.7000) - Kidney Disease Kidney disease is often called a "silent" disease, because most people have no symptoms with early kidney disease. In fact, you might feel just fine until your kidneys have almost stopped working. Do NO...
2. `medquad_ans_0000863-1` (score=0.5411) - REN-related kidney disease is an inherited condition that affects kidney function. This condition causes slowly progressive kidney disease that usually becomes apparent during childhood. As this condition progresses, ...
3. `medquad_ans_0004939-1` (score=0.4841) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the ...
4. `medquad_ans_0000167-1__dup4` (score=0.4402) - Polycystic kidney disease is a genetic disorder that causes numerous cysts to grow in the kidneys. A kidney cyst is an abnormal sac filled with fluid. PKD cysts can greatly enlarge the kidneys while replacing much of ...
5. `medquad_ans_0000144-10` (score=0.4262) - Renal failure is any acute or chronic loss of kidney function and is the term used when some kidney function remains. Total kidney failure, sometimes called end-stage renal disease (ESRD), indicates permanent loss of ...

**Answer**:

Kidney disease is often called a "silent" disease because most people have no symptoms in the early stages, and individuals might feel fine until their kidneys have almost stopped working. It is characterized by damaged kidneys that filter less than 15% of the blood filtered by healthy kidneys, leading to a condition known as uremia, where waste products build up in the blood. Kidney disease can also result in hormonal imbalances and disruptions in mineral balance in the blood, causing complications such as itching, sleep problems, restless legs, weak bones, joint problems, and depression. Diagnosis is typically made through blood and urine tests, which assess kidney function using the glomerular filtration rate (GFR). Key risk factors include diabetes, high blood pressure, heart disease, and a family history of kidney failure [medquad_ans_0000038-3__dup2].

**Supported citations**: `medquad_ans_0000038-3__dup2`
**Unsupported citations**: _none_

### Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Scores**: context_relevance=0.857, groundedness=0.917, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000058-23` (score=0.7000) - There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But ...
2. `medquad_ans_0000036_1-6` (score=0.6671) - Key Points - There are different types of treatment for patients with prostate cancer. - Seven types of standard treatment are used: - Watchful waiting or active surveillance - Surgery - Radiation therapy and radiopha...
3. `medquad_ans_0000036_1-7` (score=0.5730) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000750-1__dup1` (score=0.5429) - The prostate is the gland below a man's bladder that produces fluid for semen. Prostate cancer is common among older men. It is rare in men younger than 40. Risk factors for developing prostate cancer include being ov...
5. `medquad_ans_0000058-6__dup1` (score=0.4356) - Choosing Treatment There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade...

**Answer**:

The treatments for prostate cancer include watchful waiting or active surveillance, surgery, radiation therapy and radiopharmaceutical therapy, hormone therapy, chemotherapy, biologic therapy, and bisphosphonate therapy. Treatments for bone pain caused by bone metastases or hormone therapy are also available. Additionally, new types of treatments such as cryosurgery, high-intensity focused ultrasound therapy, and proton beam radiation therapy are being used. Patients may also consider participating in clinical trials, which are research studies aimed at improving current treatments or obtaining information on new treatments [medquad_ans_0000036_1-6].

**Supported citations**: `medquad_ans_0000036_1-6`
**Unsupported citations**: _none_

### Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Scores**: context_relevance=0.857, groundedness=0.518, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000036-4__dup3` (score=0.7000) - Insurance companies sell private health insurance plans. Two types of private health insurance are - Group health insurance. People may be eligible to purchase group health insurance through their employer or union or...
2. `medquad_ans_0000036-2__dup3` (score=0.6828) - Health insurance helps pay for medical care, including the cost of diabetes care. Health insurance options include the following: - private health insurance, which includes group and individual health insurance - gove...
3. `medquad_ans_0000431-1__dup1` (score=0.6241) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...
4. `medquad_ans_0000362-1__dup2` (score=0.5446) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
5. `medquad_ans_0000045-9` (score=0.5209) - Medicare Part B is medical insurance. It helps cover medical services such as doctor's services, outpatient care and other medical services that Part A doesn't cover. Part B also covers some preventive services, such ...

**Answer**:

Yes, I have information about health insurance. Private health insurance includes group health insurance, which individuals may purchase through employers, unions, professional organizations, or alumni groups, and individual health insurance, which individuals can buy for themselves and their families. The HealthCare.gov website provides information about individual insurance plans and a search function called the Health Insurance Marketplace to find options by state. Depending on income and family size, some individuals may qualify for lower-cost premiums through the Marketplace. People can select or change individual plans during the annual open enrollment period, and certain life events may allow enrollment outside this period. Employers may have a waiting period of no longer than 90 days for employees to enroll in a company health plan, and HMOs may have an affiliation period of no longer than three months. The ACA expanded access to preventive services, such as diabetes screening for adults with high blood pressure, obesity screening, and counseling at no cost for adults and children. State

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 93 (`0003860-1`)

**Question**: What is (are) MECP2 duplication syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.935, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003860-1` (score=0.7000) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...
2. `medquad_ans_0003860-3` (score=0.7000) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
3. `medquad_ans_0000635-1` (score=0.6506) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infan...
4. `medquad_ans_0000635-3` (score=0.6316) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
5. `medquad_ans_0000635-2` (score=0.5894) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of...

**Answer**:

MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. It is caused by an extra copy (duplication) of the MECP2 gene in each cell, which is generally due to a duplication of genetic material on the long (q) arm of the X chromosome [medquad_ans_0003860-1]. The MECP2 gene encodes a protein important for normal brain functioning, particularly in regulating the expression of other genes in the brain. A duplication of this gene leads to excess protein production, disrupting proper gene regulation and resulting in irregular brain activity, which manifests as the signs and symptoms of the condition [medquad_ans_0003860-3]. Other signs and symptoms include infantile hypotonia, delayed motor milestones (such as sitting up or crawling), recurrent infections, poor or absent speech, seizures, and/or spastic

**Supported citations**: `medquad_ans_0003860-1`, `medquad_ans_0003860-3`
**Unsupported citations**: _none_

### Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Scores**: context_relevance=0.833, groundedness=0.759, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000028_2-1` (score=0.7000) - Key Points - Melanoma is a disease in which malignant (cancer) cells form in melanocytes (cells that color the skin). - There are different types of cancer that start in the skin. - Melanoma can occur anywhere on the ...
2. `medquad_ans_0000594-1__dup2` (score=0.6062) - Melanoma is the most serious type of skin cancer. Often the first sign of melanoma is a change in the size, shape, color, or feel of a mole. Most melanomas have a black or black-blue area. Melanoma may also appear as ...
3. `medquad_ans_0005046-1` (score=0.5743) - Primary melanoma of the gastrointestinal (GI) tract refers to a melanoma starting in the stomach, intestines, salivary glands, mouth, esophagus, liver, pancreas, gallbladder, or rectum. Melanoma is a disease in which ...
4. `medquad_ans_0005046-4` (score=0.5424) - How might primary melanoma of the small intestine be diagnosed? A variety of tests may be involved in the initial diagnosis of the tumor, including contrast radiography, endoscopy, and CT scan. The tumor is confirmed ...
5. `medquad_ans_0000017_1-1` (score=0.5211) - Key Points - Intraocular melanoma is a disease in which malignant (cancer) cells form in the tissues of the eye. - Being older and having fair skin may increase the risk of intraocular melanoma. - Signs of intraocular...

**Answer**:

The provided context does not contain specific information about the outlook for melanoma, such as prognosis or survival rates. While it mentions that certain factors affect prognosis and treatment options [medquad_ans_0000028_2-1], no details on the actual outlook are given. Therefore, based on the available information, the outlook for melanoma cannot be determined.

**Supported citations**: `medquad_ans_0000028_2-1`
**Unsupported citations**: _none_

### Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.664, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000132-2` (score=0.7000) - Boomerang dysplasia is a rare disorder; its exact prevalence is unknown. Approximately 10 affected individuals have been identified.
2. `medquad_ans_0000139-2__dup2` (score=0.7000) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...
3. `medquad_ans_0002274-3` (score=0.6122) - How is familial isolated hyperparathyroidism inherited? Familial isolated hyperparathyroidism (FIHP) is typically inherited in an autosomal dominant manner. This means that having only one changed (mutated) copy of th...
4. `medquad_ans_0000132-1` (score=0.6122) - Boomerang dysplasia is a disorder that affects the development of bones throughout the body. Affected individuals are born with inward- and upward-turning feet (clubfeet) and dislocations of the hips, knees, and elbow...
5. `medquad_ans_0000132-5` (score=0.5272) - These resources address the diagnosis or management of boomerang dysplasia: - Gene Review: Gene Review: FLNB-Related Disorders - Genetic Testing Registry: Boomerang dysplasia These resources from MedlinePlus offer inf...

**Answer**:

The provided context does not specify whether boomerang dysplasia is inherited. While it describes the physical features and prevalence of the disorder, there is no information on its mode of inheritance [medquad_ans_0000132-1][medquad_ans_0000132-2]. Therefore, based on the given context, it cannot be determined if boomerang dysplasia is inherited.

**Supported citations**: `medquad_ans_0000132-1`, `medquad_ans_0000132-2`
**Unsupported citations**: _none_

### Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Scores**: context_relevance=0.700, groundedness=0.897, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0000466-1` (score=0.7000) - Hereditary multiple osteochondromas is a condition in which people develop multiple benign (noncancerous) bone tumors called osteochondromas. The number of osteochondromas and the bones on which they are located vary ...
2. `medquad_ans_0002838-1` (score=0.6712) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called...
3. `medquad_ans_0000466-2` (score=0.6284) - The incidence of hereditary multiple osteochondromas is estimated to be 1 in 50,000 individuals. This condition occurs more frequently in some isolated populations: the incidence is approximately 1 in 1,000 in the Cha...
4. `medquad_ans_0000466-5` (score=0.6260) - These resources address the diagnosis or management of hereditary multiple osteochondromas: - Gene Review: Gene Review: Hereditary Multiple Osteochondromas - Genetic Testing Registry: Multiple congenital exostosis - G...
5. `medquad_ans_0000466-3` (score=0.6036) - Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and the EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. The two exosto...

**Answer**:

The genetic changes related to hereditary multiple osteochondromas are mutations in the EXT1 and EXT2 genes [medquad_ans_0002838-1].

**Supported citations**: `medquad_ans_0002838-1`
**Unsupported citations**: _none_

### Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.775, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000746-5` (score=0.7000) - These resources address the diagnosis or management of oculofaciocardiodental syndrome: - Genetic Testing Registry: Oculofaciocardiodental syndrome These resources from MedlinePlus offer information about the diagnosi...
2. `medquad_ans_0004510-1` (score=0.6709) - Oculofaciocardiodental syndrome is a genetic syndrome that affects the eyes, heart, face, and teeth. Common signs and symptoms include abnormally small deep-set eyes, cataracts, long narrow face, a broad nasal tip tha...
3. `medquad_ans_0004510-2` (score=0.6523) - What are the signs and symptoms of Oculofaciocardiodental syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Oculofaciocardiodental syndrome. If the information is available, ...
4. `medquad_ans_0000746-1` (score=0.5974) - Oculofaciocardiodental (OFCD) syndrome is a condition that affects the development of the eyes (oculo-), facial features (facio-), heart (cardio-) and teeth (dental). This condition occurs only in females. The eye abn...
5. `medquad_ans_0000344-2` (score=0.0643) - It is estimated that 750,000 people in the United States have dilated cardiomyopathy; roughly half of these cases are familial.

**Answer**:

The provided context does not specify the total number of people affected by oculofaciocardiodental syndrome. While it lists the frequency of various signs and symptoms in individuals with the condition, such as "90% with abnormality of the cardiac septa," it does not provide a population-wide prevalence or total number of affected individuals. Therefore, the exact number of people affected cannot be determined from the given information.

[medquad_ans_0004510-2]

**Supported citations**: `medquad_ans_0004510-2`
**Unsupported citations**: _none_

### Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000189-1` (score=0.7000) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused b...
2. `medquad_ans_0001350-5` (score=0.6899) - What happens when citrullinemia type I is not treated? Untreated individuals with the severe form of citrullinemia type I have hyperammonemia (plasma ammonia concentration 1000-3000 mol/L). Without prompt intervention...
3. `medquad_ans_0001350-1` (score=0.6809) - Citrullinemia type I is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. This condition, also known as classic citrullinemia, belongs to a class of genetic diseases call...
4. `medquad_ans_0000189-2` (score=0.6711) - Type I citrullinemia is the most common form of the disorder, affecting about 1 in 57,000 people worldwide. Type II citrullinemia is found primarily in the Japanese population, where it occurs in an estimated 1 in 100...
5. `medquad_ans_0000189-5` (score=0.6445) - These resources address the diagnosis or management of citrullinemia: - Baby's First Test: Citrullinemia, Type I - Baby's First Test: Citrullinemia, Type II - Gene Review: Gene Review: Citrin Deficiency - Gene Review:...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood [medquad_ans_0000189-1]. Two forms of citrullinemia have been described: Type I (also known as classic citrullinemia) and Type II. Type I usually becomes evident in the first few days of life, with symptoms including lethargy, poor feeding, vomiting, seizures, and loss of consciousness, which can be life-threatening. A milder, later-onset form of Type I may present with intense headaches, partial loss of vision, ataxia, and lethargy. Type II chiefly affects the nervous system, causing confusion, restlessness, memory loss, abnormal behaviors, seizures, and coma, and may appear during adulthood, often triggered by medications, infections, surgery, or alcohol intake. In some cases, features of adult-onset Type II develop in individuals

**Supported citations**: `medquad_ans_0000189-1`
**Unsupported citations**: _none_

### Example 99 (`0000718-1__dup1`)

**Question**: What is (are) Bilateral frontal polymicrogyria ?

**Scores**: context_relevance=1.000, groundedness=0.960, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000718-1__dup1` (score=0.7000) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
2. `medquad_ans_0000720-1__dup1` (score=0.6723) - Bilateral generalized polymicrogyria is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). This is the most widespread form of polymicrogyria and typically affects the enti...
3. `medquad_ans_0000719-1__dup1` (score=0.6459) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides ...
4. `medquad_ans_0000807-1` (score=0.5971) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. The surface of the brain normally has many ridges or folds, called gyri. In people with polymicrogyria, the brain develops...
5. `medquad_ans_0000722-3__dup1` (score=0.5962) - What causes bilateral perisylvian polymicrogyria? The exact underlying cause of bilateral perisylvian polymicrogyria (BPP) is unknown. The signs and symptoms associated with the condition are thought to be due to impr...

**Answer**:

Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form that affects only the frontal lobes of the brain, without involving the areas located behind the Sylvius fissure or the Rolando sulcus. Signs and symptoms include delayed motor and language milestones, spastic hemiparesis or quadriparesis, and mild to moderate intellectual disability. Seizures may also be present. The cause of this form is still not known, unlike the frontoparietal form, which is associated with mutations in the GPR56 gene. Treatment is based on the signs and symptoms present in each individual [medquad_ans_0000718-1__dup1].

**Supported citations**: `medquad_ans_0000718-1__dup1`
**Unsupported citations**: _none_

### Example 100 (`0000944-5`)

**Question**: What are the treatments for spondylocostal dysostosis ?

**Scores**: context_relevance=0.714, groundedness=0.295, answer_relevance=0.486

**Top docs**:

1. `medquad_ans_0005820-1` (score=0.7000) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
2. `medquad_ans_0005821-1` (score=0.6914) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
3. `medquad_ans_0005819-1` (score=0.6713) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
4. `medquad_ans_0005822-1` (score=0.6627) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
5. `medquad_ans_0000944-2` (score=0.6374) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.

**Answer**:

Treatment for spondylocostal dysostosis is symptomatic and supportive and may include respiratory support and surgery to correct inguinal hernia and scoliosis [medquad_ans_0005820-1, medquad_ans_0005821-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0005820-1, medquad_ans_0005821-1`

### Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.929, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000786-1` (score=0.7000) - Perry syndrome is a progressive brain disease that is characterized by four major features: a pattern of movement abnormalities known as parkinsonism, psychiatric changes, weight loss, and abnormally slow breathing (h...
2. `medquad_ans_0004825-1` (score=0.6903) - What are the signs and symptoms of Perry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Perry syndrome. If the information is available, the table below includes how often...
3. `medquad_ans_0000786-5` (score=0.0776) - These resources address the diagnosis or management of Perry syndrome: - Gene Review: Gene Review: Perry Syndrome - Genetic Testing Registry: Perry syndrome - MedlinePlus Encyclopedia: Major Depression - MedlinePlus E...
4. `medquad_ans_0002652-2` (score=0.0679) - What are the signs and symptoms of Gordon syndrome? Gordon syndrome belongs to a group of conditions known as the distal arthrogryposes, which are characterized by stiffness and impaired mobility of certain joints of ...
5. `medquad_ans_0000786-3` (score=0.0585) - Perry syndrome results from mutations in the DCTN1 gene. This gene provides instructions for making a protein called dynactin-1, which is involved in the transport of materials within cells. To move materials, dynacti...

**Answer**:

The symptoms of Perry syndrome include parkinsonism (characterized by bradykinesia, stiffness, and tremors), psychiatric changes such as depression, apathy, withdrawal from friends and family, and suicidal thoughts, weight loss, and abnormally slow breathing (hypoventilation) [medquad_ans_0000786-1]. Hypoventilation often occurs at night and can lead to respiratory failure. Other reported symptoms include sleep disturbance, tremor, rigidity, bradykinesia, mask-like facies, dysarthria, hypoventilation, inappropriate behavior, insomnia, and a short-stepped shuffling gait [medquad_ans_0004825-1]. Weight loss is significant and unexplained, and the disease is rapidly progressive, with most individuals surviving about 5 years after symptom onset, typically dying from respiratory failure or pneumonia, or by suicide [medquad*ans*

**Supported citations**: `medquad_ans_0000786-1`, `medquad_ans_0004825-1`
**Unsupported citations**: _none_

### Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Scores**: context_relevance=1.000, groundedness=0.954, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001028-2` (score=0.7000) - The prevalence of warfarin sensitivity is unknown. However, it appears to be more common in people who are older, those with lower body weights, and individuals of Asian ancestry. Of the approximately 2 million people...
2. `medquad_ans_0001028-1` (score=0.6628) - Warfarin sensitivity is a condition in which individuals have a low tolerance for the drug warfarin. Warfarin is an anticoagulant, which means that it thins the blood, preventing blood clots from forming. Warfarin is ...
3. `medquad_ans_0006398-1` (score=0.6620) - Warfarin sensitivity is a condition that is characterized by a reduced tolerance for a "blood-thinning" medication called warfarin. Warfarin is an anticoagulant that is often prescribed to people who are at an increas...
4. `medquad_ans_0006398-2` (score=0.6302) - What are the signs and symptoms of Warfarin sensitivity? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin sensitivity. If the information is available, the table below includ...
5. `medquad_ans_0006399-1` (score=0.6252) - What are the signs and symptoms of Warfarin syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin syndrome. If the information is available, the table below includes how...

**Answer**:

The most common symptom of warfarin sensitivity is excessive internal bleeding, which often occurs when individuals with warfarin sensitivity are given too much of the medication. This can manifest as abnormal bleeding in the brain, gastrointestinal tract, or other tissues [medquad_ans_0001028-2]. [medquad_ans_0006398-1]

**Supported citations**: `medquad_ans_0001028-2`, `medquad_ans_0006398-1`
**Unsupported citations**: _none_

### Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.506, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0004262-1` (score=0.7000) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, a...
2. `medquad_ans_0000701-2` (score=0.6821) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.
3. `medquad_ans_0000701-1` (score=0.6716) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most...
4. `medquad_ans_0004262-5` (score=0.6649) - How might Myhre syndrome be treated? Treatment of this condition is symptomatic and supportive. Children with Myhre syndrome may require management by a team of specialists, including pediatricians, speech pathologist...
5. `medquad_ans_0004262-3` (score=0.6539) - What causes Myhre syndrome? Myhre syndrome is caused by mutations in the SMAD4 gene. This gene provides instructions for making a protein involved in transmitting chemical signals from the cell surface to the nucleus....

**Answer**:

Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement of the muscles (muscle hypertrophy), and/or joint stiffness. It is caused by mutations in the SMAD4 gene and is inherited in an autosomal dominant pattern, with most cases resulting from a new mutation. The disorder affects multiple body systems and functions, with features such as delayed development of language and motor skills, hearing loss (sensorineural, conductive, or mixed), growth retardation starting before birth, low birth weight, skin stiffness, and muscular appearance. Skeletal abnormalities include thickened skull bones, flattened spine bones (platyspondyly), broad ribs, underdeveloped pelvic wings (hypoplastic iliac wings), and short fingers and toes (brachydactyly). Typical facial features include narrow eyelid openings,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.771, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004109-2` (score=0.7000) - What are the signs and symptoms of Mondini dysplasia? Mondini dysplasia is a congenital malformation (present at birth). It may occur either unilaterally (in one ear) or bilaterally (in both ears). Most affected indiv...
2. `medquad_ans_0006057-4` (score=0.1544) - Is thanatophoric dysplasia inherited? Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell causes the condition. However, almost all cases of tha...
3. `medquad_ans_0004451-2` (score=0.1544) - What are the signs and symptoms of Noonan syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome 1. If the information is available, the table below includes how...
4. `medquad_ans_0002523-1` (score=0.1351) - What are the signs and symptoms of Genitopatellar syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Genitopatellar syndrome. If the information is available, the table below ...
5. `medquad_ans_0006057-1` (score=0.1351) - Thanatophoric dysplasia is a severe skeletal disorder characterized by extremely short limbs and folds of extra skin on the arms and legs. Other features of this condition include a narrow chest, short ribs, underdeve...

**Answer**:

The provided context does not include information about the symptoms of thanatophoric dysplasia. Therefore, based on the given context, the symptoms cannot be determined. [medquad_ans_0006057-4] discusses inheritance patterns but not symptoms.

**Supported citations**: `medquad_ans_0006057-4`
**Unsupported citations**: _none_

### Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002417-2` (score=0.7000) - What are the signs and symptoms of Fragile X syndrome? Fragile X syndrome is characterized by developmental problems including intellectual disability and delayed speech and language development. Males are usually mor...
2. `medquad_ans_0000381-1` (score=0.6706) - Fragile X syndrome is a genetic condition that causes a range of developmental problems including learning disabilities and cognitive impairment. Usually, males are more severely affected by this disorder than females...
3. `medquad_ans_0000381-5` (score=0.6706) - These resources address the diagnosis or management of fragile X syndrome: - Gene Review: Gene Review: FMR1-Related Disorders - GeneFacts: Fragile X Syndrome: Diagnosis - GeneFacts: Fragile X Syndrome: Management - Ge...
4. `medquad_ans_0000381-1__dup1` (score=0.6154) - Fragile X syndrome is the most common form of inherited developmental disability. A problem with a specific gene causes the disease. Normally, the gene makes a protein you need for brain development. But the problem c...
5. `medquad_ans_0002417-6` (score=0.5751) - How might fragile X syndrome be treated? There is no specific treatment available for fragile X syndrome. Management of this condition is generally supportive and may include: recognizing the need for special educatio...

**Answer**:

The symptoms of Fragile X syndrome include cognitive impairment, joint hypermobility, macroorchidism, neurological speech impairment, otitis media, pes planus, abnormality of the pinna, attention deficit hyperactivity disorder, frontal bossing, intellectual disability, moderate, long face, macrocephaly, mandibular prognathia, muscular hypotonia, narrow face, sinusitis, abnormality of the mitral valve, autism, cerebral cortical atrophy, dilatation of the ascending aorta, seizures, self-injurious behavior, and strabismus [medquad_ans_0002417-2]. Additionally, males may have enlarged testicles (macroorchidism) after puberty, and physical features such as a long and narrow face, large ears, prominent jaw and forehead, unusually flexible fingers, and flat feet may become more apparent with age [medquad_ans_000241

**Supported citations**: `medquad_ans_0002417-2`
**Unsupported citations**: _none_

### Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Scores**: context_relevance=1.000, groundedness=0.502, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000166-2` (score=0.7000) - CADASIL is likely a rare condition; however, its prevalence is unknown.
2. `medquad_ans_0000166-1` (score=0.6519) - Cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy, usually called CADASIL, is an inherited condition that causes stroke and other impairments. This condition affects blood flow...
3. `medquad_ans_0000166-5` (score=0.6472) - These resources address the diagnosis or management of CADASIL: - Butler Hospital: Treatment and Management of CADASIL - Gene Review: Gene Review: CADASIL - Genetic Testing Registry: Cerebral autosomal dominant arteri...
4. `medquad_ans_0000889-2__dup1` (score=0.6357) - What are the signs and symptoms of CADASIL? Strokes are the main feature of CADASIL and often occur repeatedly. Strokes may lead to severe disability such as an inability to walk and urinary incontinence. The average ...
5. `medquad_ans_0000056-3__dup1` (score=0.6295) - Symptoms usually progress slowly. By age 65, the majority of persons with CADASIL have cognitive problems and dementia. Some will become dependent due to multiple strokes.

**Answer**:

CADASIL, or cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy, is an inherited condition that causes stroke and other impairments by affecting blood flow in small blood vessels, particularly in the brain. It results from abnormal muscle cells surrounding these blood vessels (vascular smooth muscle cells), which gradually die, leading to arteriopathy. This damage can cause migraines, often with visual sensations or auras, recurrent seizures (epilepsy), and areas of tissue death (infarcts), especially in the brain, which can lead to stroke. Strokes in CADASIL typically occur during mid-adulthood, with an average age of onset at 46 years, and may recur throughout a person's lifetime. Recurrent strokes can lead to progressive cognitive decline, dementia, and changes in mood and personality. Additionally, individuals often develop leukoencephalopathy, a white matter change

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Scores**: context_relevance=1.000, groundedness=0.992, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000688-1` (score=0.7000) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate...
2. `medquad_ans_0000688-2` (score=0.7000) - Multiple endocrine neoplasia type 1 affects about 1 in 30,000 people; multiple endocrine neoplasia type 2 affects an estimated 1 in 35,000 people. Among the subtypes of type 2, type 2A is the most common form, followe...
3. `medquad_ans_0004187-2` (score=0.6153) - What are the signs and symptoms of Multiple endocrine neoplasia type 2A? The Human Phenotype Ontology provides the following list of signs and symptoms for Multiple endocrine neoplasia type 2A. If the information is a...
4. `medquad_ans_0000688-3` (score=0.5961) - Mutations in the MEN1, RET, and CDKN1B genes can cause multiple endocrine neoplasia. Mutations in the MEN1 gene cause multiple endocrine neoplasia type 1. This gene provides instructions for producing a protein called...
5. `medquad_ans_0000688-5` (score=0.5670) - These resources address the diagnosis or management of multiple endocrine neoplasia: - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 1 - Gene Review: Gene Review: Multiple Endocrine Neoplasia Type 2 - Ge...

**Answer**:

Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cells and tissues throughout the body. Multiple endocrine neoplasia typically involves tumors (neoplasia) in at least two endocrine glands; tumors can also develop in other organs and tissues. These growths can be noncancerous (benign) or cancerous (malignant). If the tumors become cancerous, the condition can be life-threatening. The major forms of multiple endocrine neoplasia are called type 1, type 2, and type 4. These types are distinguished by the genes involved, the types of hormones made, and the characteristic signs and symptoms. [medquad_ans_0000688-1]

**Supported citations**: `medquad_ans_0000688-1`
**Unsupported citations**: _none_

### Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Scores**: context_relevance=1.000, groundedness=0.912, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004155-1` (score=0.7000) - Mucopolysaccharidosis I (MPS I) is a condition that affects many parts of the body. It is a progressively debilitating disorder; however, the rate of progression varies among affected individuals. MPS I is caused by m...
2. `medquad_ans_0004155-3` (score=0.6997) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycos...
3. `medquad_ans_0000678-5` (score=0.6839) - These resources address the diagnosis or management of mucopolysaccharidosis type I: - Baby's First Test - Gene Review: Gene Review: Mucopolysaccharidosis Type I - Genetic Testing Registry: Mucopolysaccharidosis type ...
4. `medquad_ans_0000682-5` (score=0.6209) - These resources address the diagnosis or management of mucopolysaccharidosis type VI: - Emory University Lysosomal Storage Disease Center - Genetic Testing Registry: Mucopolysaccharidosis type VI - MedlinePlus Encyclo...
5. `medquad_ans_0000683-5` (score=0.6023) - These resources address the diagnosis or management of mucopolysaccharidosis type VII: - Genetic Testing Registry: Mucopolysaccharidosis type VII - National MPS Society: A Guide to Understanding MPS VII These resource...

**Answer**:

Mutations in the IDUA gene cause mucopolysaccharidosis type I. These mutations lead to reduced levels or the complete lack of the IDUA enzyme, which is responsible for breaking down large sugar molecules called glycosaminoglycans (GAGs). The lack of IDUA enzyme activity results in the accumulation of GAGs within lysosomes, causing lysosomal enlargement and the symptoms associated with the disorder [medquad_ans_0004155-3].

**Supported citations**: `medquad_ans_0004155-3`
**Unsupported citations**: _none_

### Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Scores**: context_relevance=0.900, groundedness=0.772, answer_relevance=0.620

**Top docs**:

1. `medquad_ans_0000104-4__dup2` (score=0.7000) - Pleurisy The main symptom of pleurisy is a sharp or stabbing pain in your chest that gets worse when you breathe in deeply or cough or sneeze. The pain may stay in one place or it may spread to your shoulders or back....
2. `medquad_ans_0000167-2__dup3` (score=0.4852) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up...
3. `medquad_ans_0000825-1` (score=0.1203) - Primary spontaneous pneumothorax is an abnormal accumulation of air in the space between the lungs and the chest cavity (called the pleural space) that can result in the partial or complete collapse of a lung. This ty...
4. `medquad_ans_0004784-4` (score=0.1203) - How is Pelizaeus-Merzbacher disease inherited?
5. `medquad_ans_0005065-1` (score=0.1052) - Primary spontaneous pneumothorax is an abnormal accumulation of air in the pleural space (the space between the lungs and the chest cavity) that can result in the partial or complete collapse of a lung. It is called p...

**Answer**:

The provided context does not contain information about genetic changes related to primary spontaneous pneumothorax. [medquad_ans_0000104-4__dup2] discusses symptoms of pleurisy, pneumothorax, pleural effusion, and hemothorax, but does not mention genetic changes. [medquad_ans_0000167-2__dup3] discusses gene mutations in polycystic kidney disease and genetic disorders in general, but does not relate to pneumothorax. Therefore, the context is insufficient to answer the question.

**Supported citations**: `medquad_ans_0000104-4__dup2`, `medquad_ans_0000167-2__dup3`
**Unsupported citations**: _none_

### Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000066-13` (score=0.7000) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blo...
2. `medquad_ans_0000066-1__dup4` (score=0.6004) - Stroke -- A Serious Event A stroke is serious, just like a heart attack. Each year in the United States, approximately 795,000 people have a stroke. About 610,000 of these are first or new strokes. On average, one Ame...
3. `medquad_ans_0000857-1__dup2` (score=0.6004) - A stroke is a medical emergency. Strokes happen when blood flow to your brain stops. Within minutes, brain cells begin to die. There are two kinds of stroke. The more common kind, called ischemic stroke, is caused by ...
4. `medquad_ans_0000527-1__dup2` (score=0.5860) - A stroke is a medical emergency. There are two types - ischemic and hemorrhagic. Ischemic stroke is the most common type. It is usually caused by a blood clot that blocks or plugs a blood vessel in the brain. This kee...
5. `medquad_ans_0000261-3__dup2` (score=0.4888) - Although stroke is a disease of the brain, it can affect the entire body. A common disability that results from stroke is complete paralysis on one side of the body, called hemiplegia. A related disability that is not...

**Answer**:

A stroke is sometimes called a "brain attack." Most often, stroke occurs when blood flow to the brain stops because it is blocked by a clot. When this happens, the brain cells in the immediate area begin to die. Some brain cells die because they stop getting the oxygen and nutrients they need to function. Other brain cells die because they are damaged by sudden bleeding into or around the brain. The brain cells that don't die immediately remain at risk for death. These cells can linger in a compromised or weakened state for several hours. With timely treatment, these cells can be saved. New treatments are available that greatly reduce the damage caused by a stroke. But you need to arrive at the hospital as soon as possible after symptoms start to prevent disability and to greatly improve your chances for recovery. Knowing stroke symptoms, calling 911 immediately, and getting to a hospital as quickly as possible are critical. [medquad_ans_000006

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.791, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006276-5` (score=0.7000) - How is uncombable hair syndrome diagnosed? A diagnosis of uncombable hair syndrome (UHS) is made by observing the characteristic symptoms of the condition, as well observing the hair shaft under a special microscope. ...
2. `medquad_ans_0006276-2` (score=0.6908) - What are the signs and symptoms of Uncombable hair syndrome? Uncombable hair syndrome (UHS) may first become apparent any time between the ages of 3 months and 12 years. It only affects the scalp hair. The quantity of...
3. `medquad_ans_0006276-1` (score=0.6704) - Uncombable hair syndrome (UHS) is a rare disorder of the hair shaft of the scalp. It usually is characterized by silvery-blond or straw-colored hair that is disorderly; stands out from the scalp; and cannot be combed ...
4. `medquad_ans_0006276-6` (score=0.6593) - How might uncombable hair syndrome be treated? There is no definitive treatment for uncombable hair syndrome, but the condition usually improves or resolves on its own with the onset of puberty. Gentle hair care is ge...
5. `medquad_ans_0000247-3__dup1` (score=0.5773) - What causes alopecia universalis? The exact underlying cause of alopecia universalis (AU) is not currently known. AU is an advanced form of alopecia areata (AA), a condition that leads to round patches of hair loss. A...

**Answer**:

The provided context does not specify the cause of Uncombable hair syndrome. [medquad_ans_0006276-1] mentions it is a rare disorder of the hair, but no etiological information is given. Therefore, the cause cannot be determined from the available text.

**Supported citations**: `medquad_ans_0006276-1`
**Unsupported citations**: _none_

### Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.858, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000261-4` (score=0.7000) - In most cases of cyclic vomiting syndrome, affected people have no known history of the disorder in their family. However, many affected individuals have a family history of related conditions, such as migraines, irri...
2. `medquad_ans_0000887-3` (score=0.7000) - The genetic causes of Russell-Silver syndrome are complex. The disorder often results from the abnormal regulation of certain genes that control growth. Research has focused on genes located in particular regions of c...
3. `medquad_ans_0005440-4` (score=0.6122) - Is Russell-Silver syndrome inherited? Most cases of Russell-Silver syndrome (RSS) are sporadic (not inherited), which means they occur in people with no family history of RSS. Less commonly, Russell-Silver syndrome is...
4. `medquad_ans_0003743-4` (score=0.6122) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition w...
5. `medquad_ans_0000557-4` (score=0.5272) - This condition is generally not inherited but arises from mutations in the body's cells that occur after conception. This alteration is called a somatic mutation and is present only in certain cells. Rarely, this cond...

**Answer**:

Russell-Silver syndrome is not typically inherited in a simple Mendelian pattern. Instead, it is often caused by abnormalities in the regulation of certain genes, particularly those involved in growth, due to genomic imprinting and methylation changes. These abnormalities can occur due to changes in methylation involving genes such as H19 and IGF2 on chromosome 11, which are subject to genomic imprinting [medquad_ans_0000887-3]. While some cases may result from genetic changes inherited from either parent, the inheritance pattern is complex and not fully understood. In particular, the disorder is associated with parent-specific gene expression due to imprinting, and at least one third of cases are linked to methylation abnormalities. However, the context does not confirm a clear pattern of inheritance from either mother or father, nor does it state that the syndrome is inherited in the traditional sense. Therefore, based on the provided information, Russell-S

**Supported citations**: `medquad_ans_0000887-3`
**Unsupported citations**: _none_

### Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.7000) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0000095-7` (score=0.6158) - Successful weight-loss treatments include setting goals and making lifestyle changes, such as eating fewer calories and being physically active. Medicines and weight-loss surgery also are options for some people if li...
3. `medquad_ans_0000095-1__dup2` (score=0.3859) - Espaol The terms "overweight" and "obesity" refer to body weight thats greater than what is considered healthy for a certain height. The most useful measure of overweight and obesity is body mass index (BMI). BMI is c...
4. `medquad_ans_0000718-1__dup2` (score=0.0765) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts...
5. `medquad_ans_0000094-6` (score=0.0765) - Treatments for obesity hypoventilation syndrome (OHS) include breathing support, weight loss, and medicines. The goals of treating OHS may include: Supporting and aiding your breathing Achieving major weight loss Trea...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are different types of weight loss surgery, which often limit the amount of food you can take in. Some types also affect how you digest food and absorb nutrients. All types have risks and complications, such as infections, hernias, and blood clots. Many people who have the surgery lose weight quickly, but some regain weight later. Following diet and exercise recommendations can help keep most of the weight off, and medical follow-up is required for the rest of your life [medquad_ans_0000969-1__dup2].

**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

### Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.749, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000889-5` (score=0.7000) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Sae...
2. `medquad_ans_0000889-1` (score=0.6809) - Saethre-Chotzen syndrome is a genetic condition characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the ...
3. `medquad_ans_0000889-4` (score=0.4900) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
4. `medquad_ans_0000889-3` (score=0.4848) - Mutations in the TWIST1 gene cause Saethre-Chotzen syndrome. The TWIST1 gene provides instructions for making a protein that plays an important role in early development. This protein is a transcription factor, which ...
5. `medquad_ans_0000889-2` (score=0.0902) - Saethre-Chotzen syndrome has an estimated prevalence of 1 in 25,000 to 50,000 people.

**Answer**:

The treatments for Saethre-Chotzen syndrome include surgical intervention, genetic counseling, and rehabilitation. Surgical management is often necessary to address craniosynostosis and correct abnormalities in skull shape and facial structure. Additionally, rehabilitation may be required to manage associated physical and developmental issues. Genetic counseling is also provided to help families understand the condition and its inheritance pattern [medquad_ans_0000889-5]. However, specific details on the types of surgeries or rehabilitation protocols are not provided in the context.

**Supported citations**: `medquad_ans_0000889-5`
**Unsupported citations**: _none_

### Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=1.000, groundedness=0.881, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000512-1` (score=0.7000) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effect...
2. `medquad_ans_0000512-2` (score=0.6617) - Idiopathic pulmonary fibrosis has an estimated prevalence of 13 to 20 per 100,000 people worldwide. About 100,000 people are affected in the United States, and 30,000 to 40,000 new cases are diagnosed each year. Famil...
3. `medquad_ans_0000512-3` (score=0.6345) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of devel...
4. `medquad_ans_0000758-1__dup2` (score=0.6337) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
5. `medquad_ans_0000078-1__dup2` (score=0.6262) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...

**Answer**:

Idiopathic pulmonary fibrosis is a chronic, progressive lung disease characterized by the buildup of scar tissue (fibrosis) in the lungs, which impairs the lungs' ability to transport oxygen into the bloodstream effectively [medquad_ans_0000512-1]. The disease typically affects individuals between the ages of 50 and 70 and presents with common symptoms such as shortness of breath, a persistent dry, hacking cough, loss of appetite, and gradual weight loss. Some individuals may also develop clubbing of the fingers and toes due to oxygen shortage [medquad_ans_0000512-1]. The exact cause is unknown, but it is believed to result from a combination of genetic and environmental factors, where genetic changes may increase susceptibility and environmental exposures may trigger the disease [medquad_ans_0000512-3]. The condition has an estimated global prevalence of 13

**Supported citations**: `medquad_ans_0000512-1`, `medquad_ans_0000512-3`
**Unsupported citations**: _none_

### Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Scores**: context_relevance=0.917, groundedness=0.840, answer_relevance=0.783

**Top docs**:

1. `medquad_ans_0000213-3__dup1` (score=0.7000) - Early identification of and treatment for individuals with neuroleptic malignant syndrome improves outcome. If clinically indicated, a low potency neuroleptic can be reintroduced very slowly when the individual recove...
2. `medquad_ans_0004379-1` (score=0.6804) - Neuroleptic malignant syndrome is a rare neurological condition that is caused by an adverse reaction to neuroleptic (tranquilizer) or antipsychotic drugs. These drugs are commonly prescribed for the treatment of schi...
3. `medquad_ans_0000213-4__dup1` (score=0.6103) - The NINDS supports research on neurological disorders such as neuroleptic malignant syndrome. Much of this research focuses on finding ways to prevent and treat the disorder.
4. `medquad_ans_0000019_3-6` (score=0.0735) - Key Points - There are different types of treatment for patients with ovarian low malignant potential tumor. - Two types of standard treatment are used: - Surgery - Chemotherapy - New types of treatment are being test...
5. `medquad_ans_0000015_1-8` (score=0.0643) - Key Points - There are different types of treatment for children with extracranial germ cell tumors. - Children with extracranial germ cell tumors should have their treatment planned by a team of health care providers...

**Answer**:

The provided context does not specify any current research or clinical trials being conducted for neuroleptic malignant syndrome. While the NINDS supports research on neurological disorders such as neuroleptic malignant syndrome [medquad_ans_0000213-4__dup1], no details about specific studies, trials, or treatment investigations are given in the context. Therefore, based on the available information, the exact nature of ongoing research or clinical trials for neuroleptic malignant syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000213-4__dup1`
**Unsupported citations**: _none_

### Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Scores**: context_relevance=0.857, groundedness=0.860, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000203-11` (score=0.7000) - - A kidney stone is a solid piece of material that forms in a kidney when substances that are normally found in the urine become highly concentrated. - Kidney stones are one of the most common disorders of the urinary...
2. `medquad_ans_0000221-8` (score=0.6098) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other he...
3. `medquad_ans_0000221-12` (score=0.4883) - - A kidney stone is a solid piece of material that forms in a kidney when there are high levels of certain substances in the urine. These substances are normally found in the urine and do not cause problems at lower l...
4. `medquad_ans_0000204-11` (score=0.4490) - - A kidney stone is a solid piece of material that forms in a kidney when some substances that are normally found in the urine become highly concentrated. - Kidney stones occur in infants, children, and teenagers from...
5. `medquad_ans_0000203-10` (score=0.2918) - The first step in preventing kidney stones is to understand what is causing the stones to form. The health care provider may ask the person to try to catch the kidney stone as it passes, so it can be sent to a lab for...

**Answer**:

To diagnose kidney stones in adults, a healthcare provider will perform a physical exam and take a medical history, including questions about family history, diet, digestive problems, and other health issues. The provider may perform urine, blood, and imaging tests to complete the diagnosis. Urine tests can detect infections or substances in the urine that contribute to stone formation. Blood tests can identify underlying conditions that may lead to kidney stones. Imaging tests are used to locate the stones in the body and may also reveal factors that caused the stones to form [medquad_ans_0000203-11][medquad_ans_0000221-8][medquad_ans_0000221-12].

**Supported citations**: `medquad_ans_0000203-11`, `medquad_ans_0000221-8`, `medquad_ans_0000221-12`
**Unsupported citations**: _none_

### Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000041_1-1` (score=0.7000) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of ...
2. `medquad_ans_0000041_1-5` (score=0.6345) - Key Points - After urethral cancer has been diagnosed, tests are done to find out if cancer cells have spread within the urethra or to other parts of the body. - There are three ways that cancer spreads in the body. -...
3. `medquad_ans_0000041_1-2` (score=0.6239) - Signs of urethral cancer include bleeding or trouble with urination. These and other signs and symptoms may be caused by urethral cancer or by other conditions. There may be no signs or symptoms in the early stages. C...
4. `medquad_ans_0000041_1-6` (score=0.5579) - Key Points - There are different types of treatment for patients with urethral cancer. - Four types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Active surveillance - New types of tre...
5. `medquad_ans_0000929-1__dup1` (score=0.5145) - The urethra is the tube that allows urine to pass out of the body. In men, it's a long tube that runs through the penis. It also carries semen in men. In women, it's short and is just above the vagina. Urethral proble...

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. The urethra is the tube that carries urine from the bladder to outside the body. In women, the urethra is about 1 inch long and is just above the vagina. In men, the urethra is about 8 inches long, and goes through the prostate gland and the penis to the outside of the body. In men, the urethra also carries semen. Urethral cancer is a rare cancer that occurs more often in men than in women. There are different types of urethral cancer that begin in cells that line the urethra, named for the types of cells that become malignant: squamous cell carcinoma, transitional cell carcinoma, and adenocarcinoma. Squamous cell carcinoma is the most common type and forms in cells near the bladder in women and in the lining

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004671-4` (score=0.7000) - How is pachyonychia congenita inherited? Pachyonychia congenita (PC) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of the responsible...
2. `medquad_ans_0004725-1` (score=0.0938) - Paramyotonia congenita is an inherited condition that affects muscles used for movement (skeletal muscles), mainly in the face, neck, arms, and hands. Symptoms begin in infancy or early childhood and include episodes ...
3. `medquad_ans_0006446-4` (score=0.0830) - Is Williams syndrome inherited?
4. `medquad_ans_0000771-1` (score=0.0830) - Paramyotonia congenita is a disorder that affects muscles used for movement (skeletal muscles). Beginning in infancy or early childhood, people with this condition experience bouts of sustained muscle tensing (myotoni...
5. `medquad_ans_0004750-4` (score=0.0726) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is pass...

**Answer**:

Yes, paramyotonia congenita is inherited in an autosomal dominant pattern. This means that a person only needs a mutation in one copy of the responsible gene to be affected. The condition is caused by mutations in the SCN4A gene and is inherited in this manner [medquad_ans_0004725-1].

**Supported citations**: `medquad_ans_0004725-1`
**Unsupported citations**: _none_

### Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.540, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000735-2__dup1` (score=0.7000) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound bio...
2. `medquad_ans_0000735-1__dup1` (score=0.6803) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. The disorder may become apparent in the first few months of life, or later in childhood. The more severe form ...
3. `medquad_ans_0000125-1` (score=0.6612) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. If this condition is not recognized and treated, its signs and symptoms typically appear within the first few ...
4. `medquad_ans_0000125-3` (score=0.5314) - Mutations in the BTD gene cause biotinidase deficiency. The BTD gene provides instructions for making an enzyme called biotinidase. This enzyme recycles biotin, a B vitamin found in foods such as liver, egg yolks, and...
5. `medquad_ans_0000125-4` (score=0.5223) - This condition is inherited in an autosomal recessive pattern, which means both copies of the BTD gene in each cell have mutations. The parents of an individual with biotinidase deficiency each carry one copy of the m...

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), reduced consciousness/confusion (7.5%), respiratory insufficiency (7.5%), skin ulcer (7.5%), and visual field defect (7.5%). Other associated symptoms may include feeding difficulties in infancy, recurrent skin infections, seborrheic dermatitis, sensorineural hearing impairment, hepatomegaly, hyperammonemia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Scores**: context_relevance=0.714, groundedness=0.785, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000019-1__dup3` (score=0.7000) - Summary : Air pollution is a mixture of solid particles and gases in the air. Car emissions, chemicals from factories, dust, pollen and mold spores may be suspended as particles. Ozone, a gas, is a major part of air p...
2. `medquad_ans_0000505-1__dup1` (score=0.6234) - Summary : We usually think of air pollution as being outdoors, but the air in your house or office could also be polluted. Sources of indoor pollution include - Mold and pollen - Tobacco smoke - Household products and...
3. `medquad_ans_0000298-1__dup1` (score=0.5427) - Summary : We all need to drink water. How much you need depends on your size, activity level, and the weather where you live. The water you drink is a combination of surface water and groundwater. Surface water includ...
4. `medquad_ans_0000366-7` (score=0.4677) - Prevention No vaccine is available. The best way to prevent schistosomiasis is to take the following steps if you are visiting or live in an area where schistosomiasis is transmitted: - Avoid swimming or wading in fre...
5. `medquad_ans_0000368-1__dup1` (score=0.3396) - Floods are common in the United States. Weather such as heavy rain, thunderstorms, hurricanes, or tsunamis can cause flooding. Flooding can also happen when a river or stream overflows its bank, when a levee is breach...

**Answer**:

No, there is no information about water pollution in the provided context. The context discusses drinking water sources, treatment, and quality, but does not address water pollution specifically [medquad_ans_0000298-1__dup1].

**Supported citations**: `medquad_ans_0000298-1__dup1`
**Unsupported citations**: _none_

### Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000132-3__dup1` (score=0.7000) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is...
2. `medquad_ans_0000132-1__dup1` (score=0.6798) - Foot drop describes the inability to raise the front part of the foot due to weakness or paralysis of the muscles that lift the foot. As a result, individuals with foot drop scuff their toes along the ground or bend t...
3. `medquad_ans_0000132-4__dup1` (score=0.5986) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to the neurological conditions that cause foot drop in its laboratories at the National Institutes of Health (NIH), and als...
4. `medquad_ans_0000132-2__dup1` (score=0.5652) - Treatment depends on the specific cause of foot drop. The most common treatment is to support the foot with light-weight leg braces and shoe inserts, called ankle-foot orthotics. Exercise therapy to strengthen the mus...
5. `medquad_ans_0005812-1` (score=0.5214) - Split hand split foot nystagmus is a rare congenital syndrome characterized by split hand and split foot deformity and eye abnormalities, especially nystagmus. It is thought to have an autosomal dominant mode of inher...

**Answer**:

The outlook for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1].

**Supported citations**: `medquad_ans_0000132-3__dup1`
**Unsupported citations**: _none_

### Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000916-5__dup1` (score=0.7000) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene...
2. `medquad_ans_0000145-2` (score=0.6557) - The prevalence of Camurati-Engelmann disease is unknown. Approximately 200 cases have been reported worldwide.
3. `medquad_ans_0000916-3__dup1` (score=0.6543) - What causes Camurati-Engelmann disease? Mutations in the TGFB1 gene cause Camurati-Engelmann disease. The TGFB1 gene provides instructions for producing a protein called transforming growth factor beta-1 (TGF-1). The ...
4. `medquad_ans_0000145-1` (score=0.6193) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and h...
5. `medquad_ans_0000145-5` (score=0.6130) - These resources address the diagnosis or management of Camurati-Engelmann disease: - Gene Review: Gene Review: Camurati-Engelmann Disease - Genetic Testing Registry: Diaphyseal dysplasia These resources from MedlinePl...

**Answer**:

Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are also affected. The thickened bones can lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. An increase in the density of the skull results in increased pressure on the brain [medquad_ans_0000145-1]. The disease is caused by mutations in the TGFB1 gene, which result in the production of a transforming growth factor beta-1 (TGF-1) protein that is always active, leading to increased bone density and decreased body fat and muscle tissue [medquad_ans_0000916-3__dup1]. Diagnosis is based on physical examination, radiographic findings, and can be confirmed by

**Supported citations**: `medquad_ans_0000145-1`, `medquad_ans_0000916-3__dup1`
**Unsupported citations**: _none_

### Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Scores**: context_relevance=1.000, groundedness=0.978, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002080-2` (score=0.7000) - What are the signs and symptoms of Emery-Dreifuss muscular dystrophy, X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Emery-Dreifuss muscular dystrophy, X-linked. If the in...
2. `medquad_ans_0000316-2` (score=0.6716) - X-linked Emery-Dreifuss muscular dystrophy is the most common form of this condition, affecting an estimated 1 in 100,000 people. The autosomal recessive type of this disorder appears to be very rare; only a few cases...
3. `medquad_ans_0000316-5` (score=0.6697) - These resources address the diagnosis or management of Emery-Dreifuss muscular dystrophy: - Gene Review: Gene Review: Emery-Dreifuss Muscular Dystrophy - Genetic Testing Registry: Emery-Dreifuss muscular dystrophy - G...
4. `medquad_ans_0000316-1` (score=0.6601) - Emery-Dreifuss muscular dystrophy is a condition that chiefly affects muscles used for movement (skeletal muscles) and heart (cardiac) muscle. Among the earliest features of this disorder are joint deformities called ...
5. `medquad_ans_0003610-2` (score=0.4756) - What are the signs and symptoms of Limb-girdle muscular dystrophy, type 2C? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy, type 2C. If the informatio...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked include Achilles tendon contracture, elbow flexion contracture, decreased cervical spine flexion due to contractures of posterior cervical muscles, pectus excavatum, elevated serum creatine phosphokinase, primary atrial arrhythmia, atrioventricular block, slow progression, sudden cardiac death, type 1 muscle fiber atrophy, waddling gait, and childhood or juvenile onset [medquad_ans_0002080-2].

**Supported citations**: `medquad_ans_0002080-2`
**Unsupported citations**: _none_

### Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.717, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000072-2` (score=0.7000) - Arterial tortuosity syndrome is a rare disorder; its prevalence is unknown. About 100 cases have been reported in the medical literature.
2. `medquad_ans_0000072-1` (score=0.6993) - Arterial tortuosity syndrome is a disorder that affects connective tissue. Connective tissue provides strength and flexibility to structures throughout the body, including blood vessels, skin, joints, and the gastroin...
3. `medquad_ans_0000072-5` (score=0.6993) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins Mc...
4. `medquad_ans_0000072-3` (score=0.6721) - Arterial tortuosity syndrome is caused by mutations in the SLC2A10 gene. This gene provides instructions for making a protein called GLUT10. The level of GLUT10 appears to be involved in the regulation of a process ca...
5. `medquad_ans_0004454-3` (score=0.2810) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not mention any treatments for arterial tortuosity syndrome. [medquad_ans_0000072-1] and [medquad_ans_0000072-2] describe the disorder's characteristics, prevalence, and complications but do not address therapeutic interventions. Therefore, based on the given information, treatments cannot be determined.

**Supported citations**: `medquad_ans_0000072-1`, `medquad_ans_0000072-2`
**Unsupported citations**: _none_

### Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Scores**: context_relevance=1.000, groundedness=0.756, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001093-1__dup1` (score=0.7000) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is ava...
2. `medquad_ans_0001100-1__dup1` (score=0.6301) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is a...
3. `medquad_ans_0001092-1__dup1` (score=0.5798) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
4. `medquad_ans_0000559-1__dup1` (score=0.5530) - What are the signs and symptoms of Autosomal dominant intermediate Charcot-Marie-Tooth disease type C? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant intermediate...
5. `medquad_ans_0000173-1` (score=0.5385) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...

**Answer**:

The provided context does not contain information about the symptoms of Charcot-Marie-Tooth disease type 1C. [medquad_ans_0001093-1__dup1] discusses symptoms of Charcot-Marie-Tooth disease type 1D, and [medquad_ans_0001100-1__dup1] discusses symptoms of Charcot-Marie-Tooth disease type 2B2. There is no mention of type 1C. Therefore, the symptoms of Charcot-Marie-Tooth disease type 1C cannot be determined from the given context.

**Supported citations**: `medquad_ans_0001093-1__dup1`, `medquad_ans_0001100-1__dup1`
**Unsupported citations**: _none_

### Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.886, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0005321-1` (score=0.7000) - What are the signs and symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal arterial macroaneurysm...
2. `medquad_ans_0000870-5` (score=0.6904) - These resources address the diagnosis or management of RAMSVPS: - Calgary Retina Consultants: Retinal Arterial Macroaneurysm - Genetic Testing Registry: Retinal arterial macroaneurysm with supravalvular pulmonic steno...
3. `medquad_ans_0000870-1` (score=0.6804) - Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) is a disorder that affects blood vessels in the eyes and heart. The condition generally becomes apparent in infancy or childhood. RAMSVPS d...
4. `medquad_ans_0005197-1` (score=0.4894) - What are the signs and symptoms of Pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pulmonic stenosis. If the information is available, the table below includes how...
5. `medquad_ans_0005330-2` (score=0.4409) - What are the signs and symptoms of Retinal vasculopathy with cerebral leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal vasculopathy with cerebral leukodystroph...

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) include exudative retinal detachment and pulmonic stenosis [medquad_ans_0005321-1]. These symptoms are part of the clinical features associated with the condition, as documented in the Human Phenotype Ontology [medquad_ans_0005321-1].

**Supported citations**: `medquad_ans_0005321-1`
**Unsupported citations**: _none_

### Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Scores**: context_relevance=1.000, groundedness=0.850, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001838-2` (score=0.7000) - What are the signs and symptoms of Dicarboxylic aminoaciduria? There are no common signs or symptoms of dicarboxylic aminoaciduria. Hypoglycemia, developmental and neurological abnormalities, and obsessive compulsive ...
2. `medquad_ans_0002894-1` (score=0.4686) - What are the signs and symptoms of Histidinuria renal tubular defect? The Human Phenotype Ontology provides the following list of signs and symptoms for Histidinuria renal tubular defect. If the information is availab...
3. `medquad_ans_0000286-1__dup1` (score=0.3041) - What are the signs and symptoms of Amino aciduria with mental deficiency, dwarfism, muscular dystrophy, osteoporosis and acidosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Ami...
4. `medquad_ans_0000434-5` (score=0.1904) - These resources address the diagnosis or management of GRACILE syndrome: - Genetic Testing Registry: GRACILE syndrome - MedlinePlus Encyclopedia: Aminoaciduria - MedlinePlus Encyclopedia: Cholestasis These resources f...
5. `medquad_ans_0001838-1` (score=0.0809) - Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. Symptoms have varied greatly among the few reported cases. Dicarboxylic aminoaciduria is...

**Answer**:

Dicarboxylic aminoaciduria is a rare genetic disorder characterized by the presence of dicarboxylic amino acids in the urine, typically due to autosomal recessive inheritance. It is associated with a range of signs and symptoms, though many individuals have no noticeable symptoms. Reported signs and symptoms include fasting hypoglycemia, intellectual disability, and in rare cases, developmental and neurological abnormalities or obsessive compulsive tendencies. However, in some diagnosed individuals, there are virtually no signs or symptoms. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which compiles information from sources such as Orphanet, and may vary depending on the study population [medquad_ans_0001838-2].

**Supported citations**: `medquad_ans_0001838-2`
**Unsupported citations**: _none_

### Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Scores**: context_relevance=1.000, groundedness=0.749, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005819-1` (score=0.7000) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
2. `medquad_ans_0000949-2` (score=0.6983) - Spondylothoracic dysostosis affects about one in 200,000 people worldwide. However, it is much more common in people of Puerto Rican ancestry, affecting approximately one in 12,000 people.
3. `medquad_ans_0005860-2` (score=0.6703) - What are the signs and symptoms of spondylothoracic dysostosis? Signs and symptoms of spondylothoracic dysostosis, include spine and vertebral abnormalities which result in a shortened spine, neck, and trunk, as well ...
4. `medquad_ans_0005821-1` (score=0.6573) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an...
5. `medquad_ans_0005859-1` (score=0.6537) - Spondylospinal thoracic dysostosis is an extremely rare skeletal disorder characterized by a short, curved spine and fusion of the spinous processes, short thorax with 'crab-like' configuration of the ribs, underdevel...

**Answer**:

The provided context does not specify whether spondylothoracic dysostosis is inherited. While it mentions that spondylocostal dysostosis is inherited in an autosomal recessive manner due to mutations in the DLL3 gene, there is no information in the context about the inheritance pattern of spondylothoracic dysostosis [medquad_ans_0005819-1][medquad_ans_0005860-2]. Therefore, based on the available information, the inheritance of spondylothoracic dysostosis cannot be determined.

Answer: Insufficient context to determine if spondylothoracic dysostosis is inherited.

**Supported citations**: `medquad_ans_0005819-1`, `medquad_ans_0005860-2`
**Unsupported citations**: _none_

### Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.493, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0003296-1` (score=0.7000) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. I...
2. `medquad_ans_0000542-5` (score=0.6617) - These resources address the diagnosis or management of Jervell and Lange-Nielsen syndrome: - Gene Review: Gene Review: Jervell and Lange-Nielsen Syndrome - Genetic Testing Registry: Jervell and Lange-Nielsen syndrome ...
3. `medquad_ans_0000542-1` (score=0.6528) - Jervell and Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). This disorder is a form of long QT syndrome, which is a heart ...
4. `medquad_ans_0000542-2` (score=0.6523) - Jervell and Lange-Nielsen syndrome is uncommon; it affects an estimated 1.6 to 6 per 1 million people worldwide. This condition has a higher prevalence in Denmark, where it affects at least 1 in 200,000 people.
5. `medquad_ans_0000542-3` (score=0.4476) - Mutations in the KCNE1 and KCNQ1 genes cause Jervell and Lange-Nielsen syndrome. The KCNE1 and KCNQ1 genes provide instructions for making proteins that work together to form a channel across cell membranes. These cha...

**Answer**:

Jervell Lange-Nielsen syndrome is a form of long QT syndrome that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). It is inherited in an autosomal recessive fashion and is caused by mutations in the KCNE1 and KCNQ1 genes. These genes provide instructions for making proteins that form potassium channels, which are essential for maintaining normal functions in the inner ear and cardiac muscle. About 90 percent of cases are due to mutations in the KCNQ1 gene, while the remaining cases are caused by mutations in the KCNE1 gene. The altered function or structure of these potassium channels disrupts potassium ion flow, leading to hearing loss and irregular heart rhythms. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. The condition is uncommon, affecting an estimated 1.6 to 6 per 1 million people worldwide, with a higher prevalence

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Scores**: context_relevance=1.000, groundedness=0.774, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000218-5` (score=0.7000) - These resources address the diagnosis or management of CCFDN: - Gene Review: Gene Review: Congenital Cataracts, Facial Dysmorphism, and Neuropathy - Genetic Testing Registry: Congenital Cataracts, Facial Dysmorphism, ...
2. `medquad_ans_0000218-1` (score=0.6798) - Congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is a rare disorder that affects several parts of the body. It is characterized by a clouding of the lens of the eyes at birth (congenital cataracts) and...
3. `medquad_ans_0001008-1__dup1` (score=0.3873) - What are the signs and symptoms of Cataracts, ataxia, short stature, and mental retardation? The Human Phenotype Ontology provides the following list of signs and symptoms for Cataracts, ataxia, short stature, and men...
4. `medquad_ans_0004217-1` (score=0.3358) - Muscle eye brain disease is a rare form of congenital muscular dystrophy. Individuals with this condition are born with muscle weakness (hypotonia), severe nearsightedness (myopia), glaucoma, and brain abnormalities. ...
5. `medquad_ans_0001075-1__dup1` (score=0.2930) - Cerulean cataracts are opaque areas that develop in the lens of the eye that often have a bluish or whitish color. They may be present at birth or develop in very early childhood, but may not be diagnosed until adulth...

**Answer**:

The provided context does not specify the number of people affected by congenital cataracts, facial dysmorphism, and neuropathy (CCFDN). [medquad_ans_0000218-1] describes the condition's features and complications but does not include population statistics or prevalence data. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000218-1`
**Unsupported citations**: _none_

### Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Scores**: context_relevance=0.923, groundedness=0.762, answer_relevance=0.862

**Top docs**:

1. `medquad_ans_0000127-1__dup2` (score=0.7000) - Thalassemias (thal-a-SE-me-ahs) are inherited blood disorders. "Inherited" means that the disorder is passed from parents to children through genes. Thalassemias cause the body to make fewer healthy red blood cells an...
2. `medquad_ans_0001664-3` (score=0.5492) - What causes cri du chat syndrome? Cri du chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected...
3. `medquad_ans_0000127-6` (score=0.5317) - Doctors diagnose thalassemias using blood tests, including a complete blood count (CBC) and special hemoglobin tests. A CBC measures the amount of hemoglobin and the different kinds of blood cells, such as red blood c...
4. `medquad_ans_0003225-3` (score=0.4791) - How is intellectual disability-developmental delay-contractures syndrome inherited? Intellectual disability-developmental delay-contractures syndrome syndrome is inherited in an X-linked recessive manner and is caused...
5. `medquad_ans_0000127-5__dup1` (score=0.4278) - A lack of oxygen in the bloodstream causes the signs and symptoms of thalassemias. The lack of oxygen occurs because the body doesn't make enough healthy red blood cells and hemoglobin. The severity of symptoms depend...

**Answer**:

The provided context does not contain information about the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome. Therefore, a definitive answer cannot be provided based on the given context. [medquad_ans_0000127-1__dup2] discusses thalassemias, including alpha and beta thalassemia, but does not mention X-linked intellectual disability or its genetic basis.

**Supported citations**: `medquad_ans_0000127-1__dup2`
**Unsupported citations**: _none_

### Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002622-1` (score=0.7000) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepa...
2. `medquad_ans_0002615-2` (score=0.6030) - What are the signs and symptoms of Glycogen storage disease type 13? Glycogen storage disease type 13 causes muscle pain (myalgia). Individuals with GSD13 also experience exercise intolerance, which means they have di...
3. `medquad_ans_0002622-2` (score=0.5181) - What are the signs and symptoms of Glycogen storage disease type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 6. If the information is available, ...
4. `medquad_ans_0000423-5` (score=0.3726) - These resources address the diagnosis or management of glycogen storage disease type III: - Gene Review: Gene Review: Glycogen Storage Disease Type III - Genetic Testing Registry: Glycogen storage disease type III The...
5. `medquad_ans_0000425-5` (score=0.3432) - These resources address the diagnosis or management of glycogen storage disease type IX: - Gene Review: Gene Review: Phosphorylase Kinase Deficiency - Genetic Testing Registry: Glycogen storage disease IXb - Genetic T...

**Answer**:

The symptoms of Glycogen storage disease type 6 include low blood sugar (hypoglycemia), an enlarged liver (hepatomegaly), and an increase in the amount of lactic acid in the blood (lactic acidosis), particularly when an individual does not eat for a long time. Symptoms improve significantly as individuals with this condition get older. [medquad_ans_0002622-1]

**Supported citations**: `medquad_ans_0002622-1`
**Unsupported citations**: _none_

### Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Scores**: context_relevance=1.000, groundedness=0.924, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0004342-3` (score=0.7000) - What causes nephrogenic diabetes insipidus? Nephrogenic diabetes insipidus can be either acquired or hereditary. The acquired form can result from chronic kidney disease, certain medications (such as lithium), low lev...
2. `medquad_ans_0000015-2__dup2` (score=0.6696) - Diabetes is a serious, life-long disease. It can lead to problems such as heart disease, stroke, vision loss, kidney disease, and nerve damage. More than 8 million people in the United States have type 2 diabetes and ...
3. `medquad_ans_0000035-1__dup4` (score=0.6290) - Diabetes is a complex group of diseases with a variety of causes. People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia. Diabetes is a disorder of metabolismthe way the body uses ...
4. `medquad_ans_0000057-1__dup5` (score=0.5428) - Too much glucose in the blood for a long time can cause diabetes problems. This high blood glucose, also called blood sugar, can damage many parts of the body, such as the heart, blood vessels, eyes, and kidneys. Hear...
5. `medquad_ans_0000032-1__dup4` (score=0.5281) - Diabetes is a disorder of metabolismthe way our bodies use digested food for energy. Most of the food we eat is broken down into glucose, the form of sugar in the blood. Glucose is the body's main source of fuel. Afte...

**Answer**:

The causes of diabetes include both genetic and environmental factors. Type 1 diabetes is an autoimmune disease where the body's immune system mistakenly attacks its own healthy tissues, particularly the insulin-producing cells in the pancreas. Scientists believe both genetic factors and environmental triggers, such as viruses, are involved in this process [medquad_ans_0000015-2__dup2]. Type 2 diabetes, the most common form, is closely linked to overweight and obesity, high blood pressure, and abnormal cholesterol levels. Being overweight can impair the body's ability to use insulin properly. Genetic factors also play a significant role, as certain genes or combinations of genes can increase or decrease a person's risk for developing type 2 diabetes [medquad_ans_0000015-2__dup2]. Additional risk factors include being over 45 years of age, having a first-degree relative with diabetes, and belonging to certain ethnic groups

**Supported citations**: `medquad_ans_0000015-2__dup2`
**Unsupported citations**: _none_

### Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Scores**: context_relevance=0.636, groundedness=0.707, answer_relevance=0.782

**Top docs**:

1. `medquad_ans_0000016-6` (score=0.7000) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
2. `medquad_ans_0000016-4__dup2` (score=0.6516) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...
3. `medquad_ans_0000016-1__dup3` (score=0.6367) - Can Cause Vision Loss, Blindness Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina in the back of the eye. A ...
4. `medquad_ans_0000016-2__dup2` (score=0.6111) - Who Is at Risk? All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes ...
5. `medquad_ans_0000016-24` (score=0.3804) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even ...

**Answer**:

The provided context does not mention any specific research or clinical trials being conducted for diabetic retinopathy. Therefore, based on the given information, it is not possible to identify ongoing research or clinical trials. [medquad_ans_0000016-6] [medquad_ans_0000016-4__dup2]

**Supported citations**: `medquad_ans_0000016-6`, `medquad_ans_0000016-4__dup2`
**Unsupported citations**: _none_

### Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Scores**: context_relevance=0.800, groundedness=0.937, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.7000) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart an...
2. `medquad_ans_0000977-1__dup1` (score=0.5283) - An injury is damage to your body. It is a general term that refers to harm caused by accidents, falls, hits, weapons, and more. In the U.S., millions of people injure themselves every year. These injuries range from m...
3. `medquad_ans_0000853-1__dup1` (score=0.4831) - Bronchiolitis obliterans is an inflammatory obstruction of the lung's tiniest airways, the bronchioles. The bronchioles may become damaged and inflamed after inhalation of toxic fumes, as a result of respiratory infec...
4. `medquad_ans_0000252-1__dup1` (score=0.3973) - Shaken baby syndrome is a type of inflicted traumatic brain injury that happens when a baby is violently shaken. A baby has weak neck muscles and a large, heavy head. Shaking makes the fragile brain bounce back and fo...
5. `medquad_ans_0000049-4__dup1` (score=0.3333) - The NINDS conducts and supports research on injuries to the nervous system such as brachial plexus injuries. Much of this research is aimed at finding ways to prevent and treat these disorders.

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles in the air from fires or toxic fumes. These substances can damage the eyes and respiratory system and may worsen chronic heart and lung diseases. Symptoms may include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, it may worsen. The best prevention is to limit exposure, especially when smoke is detected or fires are nearby [medquad_ans_0000516-1__dup1].

**Supported citations**: `medquad_ans_0000516-1__dup1`
**Unsupported citations**: _none_

### Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Scores**: context_relevance=0.625, groundedness=0.961, answer_relevance=0.425

**Top docs**:

1. `medquad_ans_0000134-4__dup2` (score=0.7000) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has ...
2. `medquad_ans_0000134-10` (score=0.6819) - - Zollinger-Ellison syndrome is a rare disorder that occurs when one or more tumors form in the pancreas and duodenum. - Experts do not know the exact cause of Zollinger-Ellison syndrome. - About 25 to 30 percent of g...
3. `medquad_ans_0000134-3__dup2` (score=0.6525) - Experts do not know the exact cause of Zollinger-Ellison syndrome. About 25 to 30 percent of gastrinomas are caused by an inherited genetic disorder called multiple endocrine neoplasia type 1 (MEN1).1 MEN1 causes horm...
4. `medquad_ans_0000134-9` (score=0.6466) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing Zollinger-Ellison syndrome.
5. `medquad_ans_0000134-1__dup3` (score=0.6138) - Zollinger-Ellison syndrome is a rare disorder that occurs when one or more tumors form in the pancreas and duodenum. The tumors, called gastrinomas, release large amounts of gastrin that cause the stomach to produce l...

**Answer**:

Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people [medquad_ans_0000134-4__dup2].

**Supported citations**: `medquad_ans_0000134-4__dup2`
**Unsupported citations**: _none_

### Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Scores**: context_relevance=0.857, groundedness=0.064, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000974-1` (score=0.7000) - Tetrahydrobiopterin deficiency is a rare disorder characterized by a shortage (deficiency) of a molecule called tetrahydrobiopterin or BH4. This condition alters the levels of several substances in the body, including...
2. `medquad_ans_0000974-3` (score=0.6907) - Tetrahydrobiopterin deficiency can be caused by mutations in one of several genes, including GCH1, PCBD1, PTS, and QDPR. These genes provide instructions for making enzymes that help produce and recycle tetrahydrobiop...
3. `medquad_ans_0006044-2` (score=0.6715) - What are the signs and symptoms of Tetrahydrobiopterin deficiency? Infants with tetrahydrobiopterin (BH4) deficiency typically appear normal and healthy at birth. Neurological signs and symptoms usually become apparen...
4. `medquad_ans_0001860-1` (score=0.6428) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...
5. `medquad_ans_0000974-5` (score=0.5685) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Ge...

**Answer**:

The provided context does not mention any treatments for tetrahydrobiopterin deficiency. Therefore, based on the given information, it is not possible to answer the question about treatments. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Scores**: context_relevance=1.000, groundedness=0.771, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000736-1` (score=0.7000) - Nonsyndromic holoprosencephaly is an abnormality of brain development that also affects the head and face. Normally, the brain divides into two halves (hemispheres) during early development. Holoprosencephaly occurs w...
2. `medquad_ans_0000736-2` (score=0.6903) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
3. `medquad_ans_0000736-3` (score=0.4949) - Mutations in 11 genes have been found to cause nonsyndromic holoprosencephaly. These genes provide instructions for making proteins that are important for normal embryonic development, particularly for determining the...
4. `medquad_ans_0000150-1__dup1` (score=0.4236) - Holoprosencephaly is a disorder caused by the failure of the prosencephalon (the embryonic forebrain) to sufficiently divide into the double lobes of the cerebral hemispheres. The result is a single-lobed brain struct...
5. `medquad_ans_0004011-1` (score=0.2452) - What are the signs and symptoms of Microcephaly brain defect spasticity hypernatremia? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly brain defect spasticity hypernatre...

**Answer**:

The provided context does not specify the number of people affected by nonsyndromic holoprosencephaly. [medquad_ans_0000736-1] describes the condition, its features, types, and variability but does not include epidemiological data or prevalence statistics. Therefore, the number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000736-1`
**Unsupported citations**: _none_

### Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Scores**: context_relevance=0.818, groundedness=0.880, answer_relevance=0.891

**Top docs**:

1. `medquad_ans_0000829-5` (score=0.7000) - These resources address the diagnosis or management of progressive familial intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Progressive familial intrahepatic cholest...
2. `medquad_ans_0000525-4` (score=0.5914) - Susceptibility to intrahepatic cholestasis of pregnancy is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to increase the risk of developing the disorde...
3. `medquad_ans_0005084-1` (score=0.5569) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching...
4. `medquad_ans_0000829-1` (score=0.5549) - Progressive familial intrahepatic cholestasis (PFIC) is a disorder that causes progressive liver disease, which typically leads to liver failure. In people with PFIC, liver cells are less able to secrete a digestive f...
5. `medquad_ans_0003240-4` (score=0.5537) - How is intrahepatic cholestasis of pregnancy diagnosed? Intrahepatic cholestasis of pregnancy (ICP) is suspected during pregnancy when symptoms of itching (pruritis) present after 25 weeks of gestation with absence of...

**Answer**:

Genetic changes related to progressive familial intrahepatic cholestasis include mutations in the ABCB11 gene, which cause progressive familial intrahepatic cholestasis type 2 (PFIC2) and are inherited in an autosomal recessive manner [medquad_ans_0005084-1]. Additionally, ATP8B1 deficiency is associated with a form of progressive familial intrahepatic cholestasis [medquad_ans_0000829-5]. Intrahepatic cholestasis of pregnancy has a susceptibility pattern linked to an autosomal dominant inheritance, where one copy of the altered gene increases risk, though not all carriers develop the disorder [medquad_ans_0000525-4].

**Supported citations**: `medquad_ans_0005084-1`, `medquad_ans_0000829-5`, `medquad_ans_0000525-4`
**Unsupported citations**: _none_

### Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.990, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.7000) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below in...
2. `medquad_ans_0002374-2` (score=0.4473) - What are the symptoms of fibrous dysplasia? Fibrous dysplasia may cause no symptoms, mild symptoms, or severe symptoms. The most common symptoms are bone pain, bone deformities, fractures, and skin pigmentation differ...
3. `medquad_ans_0003814-2` (score=0.0765) - What are the signs and symptoms of Marshall-Smith syndrome? Marshall-Smith syndrome is characterized by accelerated skeletal maturation, relative failure to thrive, respiratory difficulties, mental retardation, and un...
4. `medquad_ans_0003934-1` (score=0.0765) - What are the signs and symptoms of Mental retardation Smith Fineman Myers type? The Human Phenotype Ontology provides the following list of signs and symptoms for Mental retardation Smith Fineman Myers type. If the in...
5. `medquad_ans_0003845-2` (score=0.0669) - What are the signs and symptoms of McCune Albright syndrome? People with McCune Albright syndrome (MAS) may have symptoms related to bones, the endocrine system, and/or skin. The symptoms can range from mild to severe...

**Answer**:

The symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

### Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000453-1` (score=0.7000) - Hepatic veno-occlusive disease with immunodeficiency (also called VODI) is a hereditary disorder of the liver and immune system. Its signs and symptoms appear after the first few months of life. Hepatic veno-occlusive...
2. `medquad_ans_0000453-5` (score=0.6798) - These resources address the diagnosis or management of VODI: - Gene Review: Gene Review: Hepatic Veno-Occlusive Disease with Immunodeficiency - Genetic Testing Registry: Hepatic venoocclusive disease with immunodefici...
3. `medquad_ans_0002804-1` (score=0.6415) - What are the signs and symptoms of Hepatic venoocclusive disease with immunodeficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Hepatic venoocclusive disease with immunodefic...
4. `medquad_ans_0000850-5` (score=0.5085) - These resources address the diagnosis or management of pulmonary veno-occlusive disease: - Genetic Testing Registry: Pulmonary veno-occlusive disease These resources from MedlinePlus offer information about the diagno...
5. `medquad_ans_0001446-3` (score=0.4627) - What causes common variable immunodeficiency (CVID)? Common variable immunodeficiency (CVID) is usually sporadic and thought to result from a combination of genetic and environmental factors. In most cases, the exact ...

**Answer**:

Yes, hepatic veno-occlusive disease with immunodeficiency is inherited in an autosomal recessive manner. [medquad_ans_0002804-1]

**Supported citations**: `medquad_ans_0002804-1`
**Unsupported citations**: _none_

### Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Scores**: context_relevance=1.000, groundedness=0.954, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005902-1` (score=0.7000) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how o...
2. `medquad_ans_0000057-22` (score=0.0765) - You can help your doctor make a diagnosis by writing down important information about your problem beforehand and giving the information to your doctor during your visit. Write down answers to the following questions....
3. `medquad_ans_0001768-1` (score=0.0765) - What are the signs and symptoms of Dehydrated hereditary stomatocytosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Dehydrated hereditary stomatocytosis. If the information is a...
4. `medquad_ans_0000013_3-2__dup4` (score=0.0669) - Signs and symptoms of juvenile myelomonocytic leukemia include fever, weight loss, and feeling very tired. These and other signs and symptoms may be caused by JMML or by other conditions. Check with your doctor if you...
5. `medquad_ans_0000036-19` (score=0.0669) - If you are diagnosed with high blood pressure, here are questions to ask your doctor. - Your Blood Pressure Numbers - What is my blood pressure reading in numbers? - What is my goal blood pressure? - Is my blood press...

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. These signs and symptoms are based on the Human Phenotype Ontology, which compiles information from sources such as Orphanet, a European rare disease database [medquad_ans_0005902-1].

**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

### Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Scores**: context_relevance=0.667, groundedness=0.444, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0003688-1` (score=0.7000) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, h...
2. `medquad_ans_0000159-8` (score=0.6904) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed...
3. `medquad_ans_0000159-6` (score=0.6804) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
4. `medquad_ans_0000159-7` (score=0.6522) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may n...
5. `medquad_ans_0000159-1__dup3` (score=0.6254) - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). SLE is an autoimmune diseasea disorder in which the bodys immune system attacks the bodys own cells and organs. Up to 60 pe...

**Answer**:

Treatment for lupus nephritis involves medications to suppress the immune system to prevent further damage to the kidneys. Standard treatment typically includes a corticosteroid, such as prednisone, to reduce inflammation in the kidneys, and an immunosuppressive medication like cyclophosphamide or mycophenolate mofetil, used in combination with prednisone. If high blood pressure is present, medications to lower blood pressure may also be prescribed, as this can slow the progression of kidney disease. In some cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few or no further complications. However, 10 to 30 percent of people may develop kidney failure, which may require dialysis or a kidney transplant. Early diagnosis and treatment are crucial to prevent chronic kidney damage. Additionally, people with lupus nephritis are at a high risk for certain cancers, including B-cell lymphoma, and cardiovascular

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.700, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0005317-1` (score=0.7000) - Restless legs syndrome is a neurological condition that causes an irresistible urge to move the legs. The movement is triggered by strange or uncomfortable feelings, which occur mostly while the affected person is sit...
2. `medquad_ans_0000869-1` (score=0.6633) - Restless legs syndrome is a neurological condition that causes an irresistible urge to move the legs. The movement is triggered by strange or uncomfortable feelings, often described as crawling, pulling, or itching, d...
3. `medquad_ans_0000776-1__dup2` (score=0.6468) - Restless legs syndrome (RLS) causes a powerful urge to move your legs. Your legs become uncomfortable when you are lying down or sitting. Some people describe it as a creeping, crawling, tingling, or burning sensation...
4. `medquad_ans_0000246-1__dup2` (score=0.6441) - Restless legs syndrome (RLS) is a neurological disorder characterized by unpleasant sensations in the legs and an uncontrollable, and sometimes overwhelming, urge to move them for relief. Individuals affected with the...
5. `medquad_ans_0000869-2` (score=0.6435) - Restless legs syndrome is one of the most common sleep and movement disorders. It affects an estimated 5 to 10 percent of adults and 2 to 4 percent of children in the United States. For unknown reasons, the disorder a...

**Answer**:

The provided context does not specify the treatments for restless legs syndrome. It describes the symptoms, causes, and associated conditions but does not mention any treatment options [medquad_ans_0005317-1][medquad_ans_0000869-1]. Therefore, based on the given information, the treatments cannot be determined.

**Supported citations**: `medquad_ans_0005317-1`, `medquad_ans_0000869-1`
**Unsupported citations**: _none_

### Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002575-1` (score=0.7000) - Gliomatosis cerebri is a type of brain cancer. It is a variant form of glioblastoma multiforme. It is characterized by scattered and widespread tumor cells that can cause the cerebrum, cerebellum, or brain stem to enl...
2. `medquad_ans_0002573-5` (score=0.6724) - Is genetic testing available for glioblastoma? Genetic testing is not available for many people with glioblastoma since most of these tumors occur sporadically (by chance) and are not caused by a genetic mutation. How...
3. `medquad_ans_0002573-2` (score=0.5379) - What are the signs and symptoms of glioblastoma? Signs and symptoms of glioblastoma vary depending on the size and location of the tumor but may include: Headache Nausea and vomiting Drowsiness Changes in personality ...
4. `medquad_ans_0000006_3-7` (score=0.4717) - Key Points - The grade of the tumor is used to plan cancer treatment. - Low-grade astrocytomas - High-grade astrocytomas - An MRI is done after surgery. The grade of the tumor is used to plan cancer treatment. Staging...
5. `medquad_ans_0000006_1-1` (score=0.3496) - Key Points - An adult central nervous system tumor is a disease in which abnormal cells form in the tissues of the brain and/or spinal cord. - A tumor that starts in another part of the body and spreads to the brain i...

**Answer**:

Glioblastoma is a type of brain cancer that is typically diagnosed based on a physical exam identifying characteristic symptoms and various imaging studies such as computed tomography (CT) and/or magnetic resonance imaging (MRI). It is characterized by tumor cells that can cause enlargement of the cerebrum, cerebellum, or brain stem. Signs and symptoms may include headache, nausea and vomiting, drowsiness, changes in personality, weakness on one side of the body, memory loss, speech difficulty, changes in vision, and seizures. The grade of the tumor is used to plan cancer treatment, with high-grade astrocytomas being a category that includes glioblastoma [medquad_ans_0002573-5]. Glioblastoma is often sporadic and not caused by a genetic mutation, though genetic testing may be available for individuals with inherited conditions such as neurofibromatosis type 1, Turcot syndrome

**Supported citations**: `medquad_ans_0002573-5`
**Unsupported citations**: _none_

### Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Scores**: context_relevance=0.667, groundedness=0.971, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0005496-1` (score=0.7000) - Schizencephaly is a rare congenital (present from birth) brain malformation in which abnormal slits or clefts form in the cerebral hemispheres of the brain. The signs and symptoms of this condition may include develop...
2. `medquad_ans_0000251-1__dup2` (score=0.6982) - Schizencephaly is an extremely rare developmental birth defect characterized by abnormal slits, or clefts, in the cerebral hemispheres of the brain. Babies with clefts in both hemispheres (called bilateral clefts) com...
3. `medquad_ans_0000251-3__dup1` (score=0.6970) - The prognosis for individuals with schizencephaly varies depending on the size of the clefts and the extent of neurological disabilities.
4. `medquad_ans_0005496-6` (score=0.6425) - How might schizencephaly be treated? The best treatment options for people with schizencephaly depend on many factors, including the severity of the condition and the signs and symptoms present. For example, people wi...
5. `medquad_ans_0005496-2` (score=0.6324) - What are the signs and symptoms of Schizencephaly? Signs and symptoms of schizencephaly may include: Developmental delay Seizures Abnormally small head (microcephaly) Intellectual disability Partial or complete paraly...

**Answer**:

The exact cause of schizencephaly is unknown, but it has been linked to a variety of genetic and non-genetic factors [medquad_ans_0005496-1].

**Supported citations**: `medquad_ans_0005496-1`
**Unsupported citations**: _none_

### Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Scores**: context_relevance=0.800, groundedness=0.950, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000052-5` (score=0.7000) - These resources address the diagnosis or management of amyotrophic lateral sclerosis: - Gene Review: Gene Review: ALS2-Related Disorders - Gene Review: Gene Review: Amyotrophic Lateral Sclerosis Overview - Gene Review...
2. `medquad_ans_0000309-3__dup1` (score=0.6889) - What causes amyotrophic lateral sclerosis? In approximately 90-95% of cases the cause of amyotrophic lateral sclerosis (ALS) is unknown and is sporadic (occurring in individuals with no history of the condition in the...
3. `medquad_ans_0000309-1__dup1` (score=0.6332) - Amyotrophic lateral sclerosis (ALS), also referred to as "Lou Gehrig's disease," is a progressive motor neuron disease which leads to problems with muscle control and movement. There are various types of ALS, which ar...
4. `medquad_ans_0000316-1__dup1` (score=0.6221) - What are the signs and symptoms of Amyotrophic lateral sclerosis type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyotrophic lateral sclerosis type 6. If the information is a...
5. `medquad_ans_0000309-2__dup1` (score=0.5583) - What are the signs and symptoms of Amyotrophic lateral sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyotrophic lateral sclerosis. If the information is available, the ...

**Answer**:

Genetic changes related to amyotrophic lateral sclerosis (ALS) include mutations in several genes, such as C9orf72, SOD1, TARDBP, FUS, ANG, ALS2, SETX, and VAPB. These mutations can cause familial ALS and may also contribute to the development of sporadic ALS. Approximately 60% of individuals with familial ALS have an identifiable genetic mutation [medquad_ans_0000309-3__dup1]. The genes associated with ALS are thought to play a role in neuronal function or in regulating the production of various proteins [medquad_ans_0000309-3__dup1].

**Supported citations**: `medquad_ans_0000309-3__dup1`
**Unsupported citations**: _none_

### Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.967, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0003473-5` (score=0.7000) - How is Laron syndrome diagnosed? A diagnosis of Laron syndrome is often suspected based on the presence of characteristic signs and symptoms. Additional testing can then be ordered to confirm the diagnosis and rule ou...
2. `medquad_ans_0003473-6` (score=0.6821) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injectio...
3. `medquad_ans_0000581-1` (score=0.6444) - Laron syndrome is a rare form of short stature that results from the body's inability to use growth hormone, a substance produced by the brain's pituitary gland that helps promote growth. Affected individuals are clos...
4. `medquad_ans_0003473-2` (score=0.5831) - What are the signs and symptoms of Laron syndrome? Laron syndrome is a rare condition in which the body is unable to use growth hormone. The primary symptom is short stature. Although affected people are generally clo...
5. `medquad_ans_0003473-3` (score=0.5226) - What causes Laron syndrome? Laron syndrome is caused by changes (mutations) in the GHR gene. This gene encodes growth hormone receptor, which is a protein found on the outer membrane of cells throughout the body. Grow...

**Answer**:

The only specific treatment available for Laron syndrome is subcutaneous injections of insulin-like growth factor 1 (IGF-1), a growth-promoting hormone. IGF-1 stimulates linear growth (height), improves brain growth, and addresses metabolic abnormalities caused by long-term IGF-1 deficiency. It has also been shown to raise blood glucose levels, reduce cholesterol, and increase muscle growth. IGF-1 and GH levels should be closely monitored during treatment because overdosage of IGF-1 can cause various health problems [medquad_ans_0003473-6]. There is currently no cure for Laron syndrome, and treatment is primarily focused on improving growth [medquad_ans_0003473-6].

**Supported citations**: `medquad_ans_0003473-6`
**Unsupported citations**: _none_

### Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Scores**: context_relevance=0.857, groundedness=0.932, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000016-4__dup2` (score=0.7000) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...
2. `medquad_ans_0000016-6` (score=0.7000) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healt...
3. `medquad_ans_0000016-1__dup3` (score=0.6758) - Can Cause Vision Loss, Blindness Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina in the back of the eye. A ...
4. `medquad_ans_0000016-24` (score=0.6609) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even ...
5. `medquad_ans_0000016-15` (score=0.6598) - During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels of blood s...

**Answer**:

The treatments for diabetic retinopathy include:

- For macular edema: Prompt treatment with anti-VEGF drugs, with or without laser treatment, results in better vision than laser treatment alone or steroid injections. These drugs reduce fluid leakage and interfere with the growth of new blood vessels in the retina. In some cases, focal laser treatment is used along with eye injections. Focal laser treatment involves placing up to several hundred small laser burns around the macula to slow fluid leakage and reduce fluid in the retina; the surgery is usually completed in one session, though further treatment may be needed [medquad_ans_0000016-4__dup2].

- For proliferative retinopathy: Treatment involves scatter laser treatment, where 1,000 to 2,000 laser burns are placed in areas of the retina away from the macula to shrink abnormal blood vessels. This procedure usually requires two or more sessions

**Supported citations**: `medquad_ans_0000016-4__dup2`
**Unsupported citations**: _none_

### Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Scores**: context_relevance=0.857, groundedness=0.533, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0001421-3` (score=0.7000) - How might collagenous colitis be treated? Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases the condition may resolve on its own (spontaneous re...
2. `medquad_ans_0001421-1` (score=0.6985) - Collagenous colitis is a type of inflammatory bowel disease that affects the colon. It is a form of microscopic colitis, which means that the inflammation is only visible when a biopsy is examined under a microscope; ...
3. `medquad_ans_0001421-2` (score=0.6723) - What are the signs and symptoms of collagenous colitis? All individuals with collagenous colitis experience chronic, watery, non-bloody diarrhea which is what typically prompts individuals to seek medical attention. O...
4. `medquad_ans_0000122-1__dup3` (score=0.6459) - Ulcerative colitis is a chronic, or long lasting, disease that causes inflammationirritation or swellingand sores called ulcers on the inner lining of the large intestine. Ulcerative colitis is a chronic inflammatory ...
5. `medquad_ans_0000116-8` (score=0.6361) - To help reduce symptoms, a health care provider may recommend the following dietary changes: - avoid foods and drinks that contain caffeine or artificial sugars - drink plenty of liquids to prevent dehydration during ...

**Answer**:

Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases, the condition may resolve on its own (spontaneous remission), although most people continue to have ongoing or occasional diarrhea. Dietary changes are usually tried first to alleviate symptoms, including a reduced-fat diet, eliminating foods that contain caffeine and lactose, and avoiding over-the-counter pain relievers such as ibuprofen or aspirin. If these changes are not sufficient, medications can be used to control symptoms. Prescription anti-inflammatory medications such as mesalamine and sulfasalazine may help reduce swelling. Steroids including budesonide and prednisone can be used to reduce inflammation, but are usually only used for sudden attacks of diarrhea and long-term use is typically avoided due to side effects. Anti-diarrheal medications such as bismuth subsalicylate, diphenoxylate with atropine, and loper

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Scores**: context_relevance=0.667, groundedness=0.974, answer_relevance=0.756

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=0.7000) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
2. `medquad_ans_0000418-7` (score=0.6609) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause ...
3. `medquad_ans_0000038-2__dup3` (score=0.4532) - People can get infected with Babesia parasites in several ways: - The main way is through the bite of an infected tick—during outdoor activities in areas where babesiosis is found (see below). - A less common way is b...
4. `medquad_ans_0000689-1__dup2` (score=0.4062) - Parasites are living things that use other living things - like your body - for food and a place to live. You can get them from contaminated food or water, a bug bite, or sexual contact. Some parasitic diseases are ea...
5. `medquad_ans_0000038-1__dup5` (score=0.3856) - Babesiosis is caused by microscopic parasites that infect red blood cells. Most human cases of Babesia infection in the United States are caused by the parasite Babesia microti. Occasional cases caused by other specie...

**Answer**:

To prevent Trichinellosis (also known as Trichinosis):

- Wash your hands with warm water and soap after handling raw meat [medquad_ans_0000418-7].
- Freeze pork less than 6 inches thick for 20 days at 5°F (-15°C) to kill any worms [medquad_ans_0000418-7].
- Freezing wild game meats may not effectively kill all worms because some worm species that infect wild game animals are freeze-resistant [medquad_ans_0000418-7].
- Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms [medquad_ans_0000418-7].
- Clean meat grinders thoroughly after each use [medquad_ans_0000418-7].

**Supported citations**: `medquad_ans_0000418-7`
**Unsupported citations**: _none_

### Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Scores**: context_relevance=0.750, groundedness=0.893, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000182-1__dup3` (score=0.7000) - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the normal components, or cells, of the liver and causes inflammation and liver damage. The immune system normally prot...
2. `medquad_ans_0000182-5__dup1` (score=0.6985) - Autoimmune hepatitis is classified into several types. Type 1 autoimmune hepatitis is the most common form in North America. Type 1 can occur at any age; however, it most often starts in adolescence or young adulthood...
3. `medquad_ans_0000182-11` (score=0.6880) - - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the liver and causes inflammation and damage. - Autoimmune hepatitis is a serious condition that may worsen over time...
4. `medquad_ans_0000182-6` (score=0.5552) - The most common symptoms of autoimmune hepatitis are - fatigue - joint pain - nausea - loss of appetite - pain or discomfort over the liver - skin rashes - dark yellow urine - light-colored stools - jaundice, or yello...
5. `medquad_ans_0000182-8` (score=0.4676) - Treatment for autoimmune hepatitis includes medication to suppress, or slow down, an overactive immune system. Treatment may also include a liver transplant. Treatment works best when autoimmune hepatitis is diagnosed...

**Answer**:

Autoimmune hepatitis is caused by the body's immune system attacking the normal components or cells of the liver, leading to inflammation and liver damage [medquad_ans_0000182-1__dup3]. The immune system, which normally protects against infections by identifying and destroying foreign substances, mistakenly targets the liver in this condition [medquad_ans_0000182-1__dup3].

**Supported citations**: `medquad_ans_0000182-1__dup3`
**Unsupported citations**: _none_

### Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Scores**: context_relevance=0.800, groundedness=0.958, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000006_3-1` (score=0.7000) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
2. `medquad_ans_0000006_3-4` (score=0.6538) - The signs and symptoms of astrocytomas are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain or spinal cord. - The size of the tumor. - How fast the tumor gr...
3. `medquad_ans_0000006_3-7` (score=0.6315) - Key Points - The grade of the tumor is used to plan cancer treatment. - Low-grade astrocytomas - High-grade astrocytomas - An MRI is done after surgery. The grade of the tumor is used to plan cancer treatment. Staging...
4. `medquad_ans_0004883-4` (score=0.6267) - Are pilocytic astrocytomas inherited? Pilocytic astrocytomas are typically sporadic, occurring by chance in individuals with no history of the condition in the family. Sporadic abnormalities are not inherited from a p...
5. `medquad_ans_0000006_3-9` (score=0.6261) - Key Points - There are different types of treatment for patients with childhood astrocytoma. - Children with astrocytomas should have their treatment planned by a team of health care providers who are experts in treat...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They are tumors that start in star-shaped brain cells called astrocytes, which are a type of glial cell that holds nerve cells in place, brings food and oxygen to them, and helps protect them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

### Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Scores**: context_relevance=0.857, groundedness=0.980, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000013_3-1__dup3` (score=0.7000) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia in...
2. `medquad_ans_0000013_3-2__dup3` (score=0.5901) - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising or bleeding and feeling tired and weak. These and other signs and symptoms may be caused by atypical CML or by other conditions. Check ...
3. `medquad_ans_0000001_4-1` (score=0.5843) - Key Points - Chronic myelogenous leukemia is a disease in which the bone marrow makes too many white blood cells. - Leukemia may affect red blood cells, white blood cells, and platelets. - Signs and symptoms of chroni...
4. `medquad_ans_0000013_2-1__dup3` (score=0.5832) - Chronic myelogenous leukemia is a disease in which too many white blood cells are made in the bone marrow. See the PDQ summary on Chronic Myelogenous Leukemia Treatment for information on diagnosis, staging, and treat...
5. `medquad_ans_0000013_2-2__dup3` (score=0.4491) - See the PDQ summary about Chronic Myelogenous Leukemia Treatment for information.

**Answer**:

Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. In atypical chronic myelogenous leukemia (CML), the body tells too many blood stem cells to become a type of white blood cell called granulocytes. Some of these blood stem cells never become mature white blood cells; these immature white blood cells are called blasts. Over time, the granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. The leukemia cells in atypical CML and CML look alike under a microscope. However, in atypical CML, a certain chromosome change called the "Philadelphia chromosome" is not present [medquad_ans_0000013_3-1__dup3].

**Supported citations**: `medquad_ans_0000013_3-1__dup3`
**Unsupported citations**: _none_

### Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Scores**: context_relevance=0.625, groundedness=0.903, answer_relevance=0.725

**Top docs**:

1. `medquad_ans_0000770-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000996-3` (score=0.7000) - Most cases of trisomy 13 result from having three copies of chromosome 13 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the cha...
3. `medquad_ans_0000996-4` (score=0.6122) - Most cases of trisomy 13 are not inherited and result from random events during the formation of eggs and sperm in healthy parents. An error in cell division called nondisjunction results in a reproductive cell with a...
4. `medquad_ans_0001083-4` (score=0.6122) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000784-4` (score=0.5272) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The genetic changes related to trisomy 13 include having three copies of chromosome 13 in each cell instead of the usual two copies, which results from an error in cell division called nondisjunction during the formation of reproductive cells (eggs and sperm) [medquad_ans_0000996-3]. This extra genetic material disrupts normal development and causes the characteristic features of trisomy 13. In some cases, trisomy 13 occurs due to a translocation, where a part of chromosome 13 becomes attached to another chromosome during the formation of reproductive cells or very early in fetal development [medquad_ans_0000996-3]. In these translocation cases, individuals have two normal copies of chromosome 13 and an extra copy attached to another chromosome. Rarely, only part of chromosome 13 is present in three copies, leading to potentially different

**Supported citations**: `medquad_ans_0000996-3`
**Unsupported citations**: _none_

### Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Scores**: context_relevance=0.714, groundedness=0.892, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004415-1` (score=0.7000) - Niemann-Pick disease is an inherited condition involving lipid metabolism, which is the breakdown, transport, and use of fats and cholesterol in the body. In people with this condition, abnormal lipid metabolism cause...
2. `medquad_ans_0000731-2` (score=0.6752) - Niemann-Pick disease types A and B is estimated to affect 1 in 250,000 individuals. Niemann-Pick disease type A occurs more frequently among individuals of Ashkenazi (eastern and central European) Jewish descent than ...
3. `medquad_ans_0004415-3` (score=0.6752) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respirato...
4. `medquad_ans_0000218-4__dup1` (score=0.6628) - The National Institute of Neurological Disorders and Stroke (NINDS), a part of the National Institutes of Health (NIH), conducts and supports research about Niemann-Pick disease through research grants to research ins...
5. `medquad_ans_0000731-1` (score=0.6625) - Niemann-Pick disease is a condition that affects many body systems. It has a wide range of symptoms that vary in severity. Niemann-Pick disease is divided into four main types: type A, type B, type C1, and type C2. Th...

**Answer**:

There is no specific treatment for Niemann-Pick disease type A. Supportive care from various specialists may help manage symptoms, including pulmonologists for respiratory problems, cardiologists for heart problems, liver and spleen specialists, nutritionists, physical therapists, and gastroenterologists. Learning specialists may also be involved [medquad_ans_0004415-3]. However, there is currently no effective treatment for the condition, and those affected generally do not survive past early childhood [medquad_ans_0004415-1].

**Supported citations**: `medquad_ans_0004415-3`, `medquad_ans_0004415-1`
**Unsupported citations**: _none_

### Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001714-1` (score=0.7000) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial poly...
2. `medquad_ans_0000270-4` (score=0.6303) - Most cases of Dandy-Walker malformation are sporadic, which means they occur in people with no history of the disorder in their family. A small percentage of cases seem to run in families; however, Dandy-Walker malfor...
3. `medquad_ans_0001710-1` (score=0.6200) - What are the signs and symptoms of Dandy-Walker malformation with facial hemangioma? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with facial hemangioma....
4. `medquad_ans_0001708-1` (score=0.5663) - What are the signs and symptoms of Dandy-Walker like malformation with atrioventricular septal defect? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker like malformation ...
5. `medquad_ans_0000270-3` (score=0.5389) - Researchers have found mutations in a few genes that are thought to cause Dandy-Walker malformation, but these mutations account for only a small number of cases. Dandy-Walker malformation has also been associated wit...

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, autosomal recessive inheritance, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, and wide nasal bridge [

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Scores**: context_relevance=0.800, groundedness=0.752, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0003789-1` (score=0.7000) - Mantle cell lymphoma (MCL) belongs to a group of diseases known as non-Hodgkins lymphomas (NHL), which are cancers that affect the the lymphatic system (part of the immune system). MCL accounts for 6% of all non-Hodgk...
2. `medquad_ans_0003789-4` (score=0.6902) - How is Mantle cell lymphoma diagnosed? Mantle cell lymphoma is diagnosed by a biopsy (surgical removal) of the lymph nodes. If lymph nodes are not easily accessible to be biopsied, a fine needle aspiration may be perf...
3. `medquad_ans_0003789-5` (score=0.6613) - How might Mantle cell lympoma be treated? Various treatmentsare currently available for Mantle cell lymphomas. Rare cases of early stage mantle cell lymphomas may be treated with radiation therapy. For more advance st...
4. `medquad_ans_0003789-2` (score=0.6512) - What are the signs and symptoms of Mantle cell lymphoma? Common symptoms of Mantle cell lymphoma include fatigue, loss of appetite, and enlarged lymph nodes, spleen, and/or liver. Other symptoms may include night swea...
5. `medquad_ans_0004916-1` (score=0.5406) - Plasmablastic lymphoma is an aggressive form of non-Hodgkin lymphoma. Although the condition most commonly occurs in the oral cavity, it can be diagnosed in many other parts of the body such as the gastrointestinal tr...

**Answer**:

The provided context does not specify the cause of Mantle cell lymphoma. [medquad_ans_0003789-1], [medquad_ans_0003789-4], and [medquad_ans_0003789-2] describe the diagnosis, symptoms, and treatment of Mantle cell lymphoma, but none mention the underlying cause. Therefore, based on the given information, the cause cannot be determined.

**Supported citations**: `medquad_ans_0003789-1`, `medquad_ans_0003789-4`, `medquad_ans_0003789-2`
**Unsupported citations**: _none_

### Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.834, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000589-3` (score=0.7000) - Legg-Calv-Perthes disease is usually not caused by genetic factors. The cause in these cases is unknown. In a small percentage of cases, mutations in the COL2A1 gene cause the bone abnormalities characteristic of Legg...
2. `medquad_ans_0000589-1` (score=0.6997) - Legg-Calv-Perthes disease is a bone disorder that affects the hips. Usually, only one hip is involved, but in about 10 percent of cases, both hips are affected. Legg-Calv-Perthes disease begins in childhood, typically...
3. `medquad_ans_0000589-5` (score=0.6812) - These resources address the diagnosis or management of Legg-Calv-Perthes disease: - National Osteonecrosis Foundation - Seattle Children's Hospital These resources from MedlinePlus offer information about the diagnosi...
4. `medquad_ans_0003718-4` (score=0.3401) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
5. `medquad_ans_0000589-2` (score=0.0689) - The incidence of Legg-Calv-Perthes disease varies by population. The condition is most common in white populations, in which it affects an estimated 1 to 3 in 20,000 children under age 15.

**Answer**:

Legg-Calv-Perthes disease is usually not caused by genetic factors, and the cause in most cases is unknown. However, in a small percentage of cases, mutations in the COL2A1 gene may be involved, which suggests a possible genetic component in those specific instances [medquad_ans_0000589-3]. Therefore, while Legg-Calv-Perthes disease is generally not inherited, rare cases may have a genetic basis due to COL2A1 gene mutations.

**Supported citations**: `medquad_ans_0000589-3`
**Unsupported citations**: _none_

### Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Scores**: context_relevance=0.800, groundedness=0.527, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000647-2` (score=0.7000) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigra...
2. `medquad_ans_0000647-1` (score=0.6994) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the sub...
3. `medquad_ans_0000647-5` (score=0.6805) - These resources address the diagnosis or management of metachromatic leukodystrophy: - Gene Review: Gene Review: Arylsulfatase A Deficiency - Genetic Testing Registry: Metachromatic leukodystrophy - Genetic Testing Re...
4. `medquad_ans_0003566-1` (score=0.6433) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to def...
5. `medquad_ans_0000181-1__dup2` (score=0.6003) - Leukodystrophy refers to progressive degeneration of the white matter of the brain due to imperfect growth or development of the myelin sheath, the fatty covering that acts as an insulator around nerve fiber. Myelin, ...

**Answer**:

Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells, particularly in the nervous system cells that produce myelin, the substance that insulates and protects nerves. This accumulation leads to progressive destruction of white matter throughout the nervous system, including the brain, spinal cord, and peripheral nerves. The damage results in progressive deterioration of intellectual functions and motor skills, such as the ability to walk, loss of sensation in the extremities (peripheral neuropathy), incontinence, seizures, paralysis, inability to speak, blindness, and hearing loss. Eventually, individuals may lose awareness of their surroundings and become unresponsive. Effects of sulfatide accumulation have also been reported on other organs, most commonly the gallbladder. The most common form, affecting about 50 to 60 percent of individuals, is the late infantile form, which typically appears in the second year of life and leads to

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.941, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005135-1` (score=0.7000) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
2. `medquad_ans_0002374-2` (score=0.3831) - What are the symptoms of fibrous dysplasia? Fibrous dysplasia may cause no symptoms, mild symptoms, or severe symptoms. The most common symptoms are bone pain, bone deformities, fractures, and skin pigmentation differ...
3. `medquad_ans_0003558-1` (score=0.2508) - What are the signs and symptoms of Lethal short limb skeletal dysplasia Al Gazali type? The Human Phenotype Ontology provides the following list of signs and symptoms for Lethal short limb skeletal dysplasia Al Gazali...
4. `medquad_ans_0005849-1` (score=0.1942) - What are the signs and symptoms of Spondylometaphyseal dysplasia corner fracture type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia corner fracture t...
5. `medquad_ans_0002946-1` (score=0.0697) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies, elbow dislocation, hypoplasia of the zygomatic bone, scoliosis, omphalocele, hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, tongue-like lumbar vertebral deformities [medquad_ans_0005135-1]. These symptoms occur in approximately 90% of patients for most listed features, except omphalocele, which occurs in 7.5% [medquad_ans_0005135-1].

**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

### Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.967, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001664-3` (score=0.7000) - What causes cri du chat syndrome? Cri du chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected...
2. `medquad_ans_0000251-3` (score=0.7000) - Cri-du-chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected individuals; studies suggest that...
3. `medquad_ans_0000220-1__dup3` (score=0.6122) - Interstitial cystitis\*painful bladder syndrome (IC/PBS) is one of several conditions that causes bladder pain and a need to urinate frequently and urgently. Some doctors have started using the term bladder pain syndro...
4. `medquad_ans_0000251-1` (score=0.6122) - Cri-du-chat (cat's cry) syndrome, also known as 5p- (5p minus) syndrome, is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that ...
5. `medquad_ans_0000251-2` (score=0.5272) - Cri-du-chat syndrome occurs in an estimated 1 in 20,000 to 50,000 newborns. This condition is found in people of all ethnic backgrounds.

**Answer**:

Cri-du-chat syndrome is a chromosomal condition that results when a piece of chromosome 5 is missing, specifically a deletion of the end of the short (p) arm of chromosome 5, written as 5p-. This condition is characterized by a high-pitched cry that sounds like that of a cat, intellectual disability, delayed development, small head size (microcephaly), low birth weight, weak muscle tone (hypotonia) in infancy, and distinctive facial features such as widely set eyes (hypertelorism), low-set ears, a small jaw, and a rounded face. Some children with this condition are born with a heart defect. The signs and symptoms are likely related to the loss of multiple genes on the short arm of chromosome 5, and researchers believe that the loss of the gene CTNND2 is associated with severe intellectual disability in some individuals [medquad_ans_0001664-3]med

**Supported citations**: `medquad_ans_0001664-3`
**Unsupported citations**: _none_

### Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Scores**: context_relevance=1.000, groundedness=0.989, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000719-2__dup1` (score=0.7000) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmen...
2. `medquad_ans_0000718-1__dup1` (score=0.6709) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
3. `medquad_ans_0000719-1__dup1` (score=0.6431) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides ...
4. `medquad_ans_0000807-5` (score=0.6083) - These resources address the diagnosis or management of polymicrogyria: - Gene Review: Gene Review: Polymicrogyria Overview - Genetic Testing Registry: Congenital bilateral perisylvian syndrome - Genetic Testing Regist...
5. `medquad_ans_0000807-3` (score=0.5835) - In most people with polymicrogyria, the cause of the condition is unknown. However, researchers have identified several environmental and genetic factors that can be responsible for the disorder. Environmental causes ...

**Answer**:

The signs and symptoms of bilateral frontoparietal polymicrogyria may include moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze, ataxia, strabismus, increased muscle tone, finger dysmetria, ankle clonus, Babinski sign, broad-based gait, cerebellar hypoplasia, esotropia, exotropia, hyperreflexia, hypertonia, hypoplasia of the brainstem, nystagmus, polymicrogyria with an anterior to posterior gradient, and truncal ataxia [medquad_ans_0000719-2__dup1].

**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

### Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.318, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0005818-1` (score=0.7000) - Spondylocarpotarsal synostosis (SCT) syndrome is an inherited syndrome characterized by disproportionate short stature, abnormalities of the vertebrae in the spine, scoliosis and lordosis, carpal and tarsal fusion (fu...
2. `medquad_ans_0003414-3` (score=0.7000) - What causes Konigsmark Knox Hussels syndrome? Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid ar...
3. `medquad_ans_0003414-4` (score=0.6122) - Is genetic testing available for Konigsmark Knox Hussels syndrome? GeneTests lists the names of laboratories that are performing genetic testing for Konigsmark Knox Hussels syndrome. To view the contact information fo...
4. `medquad_ans_0006375-1` (score=0.6122) - Vohwinkel syndrome is an inherited condition that affects the skin. People with the "classic form" generally have honeycomb-patterned calluses on the palms of the hands and the soles of the feet (palmoplantar keratose...
5. `medquad_ans_0003414-1` (score=0.5272) - Konigsmark Knox Hussels syndrome is an inherited condition that causes both hearing and vision loss. This condition is characterized by late-onset progressive sensorineural deafness and progressive optic atrophy, whic...

**Answer**:

Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid arginine with the amino acid histidine at position 445 in the OPA1 protein, written as Arg445His or R445H [doc_id: medquad_ans_0003414-3].

**Supported citations**: _none_
**Unsupported citations**: `doc_id: medquad_ans_0003414-3`

### Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.666, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004590-1` (score=0.7000) - What are the signs and symptoms of Orofaciodigital syndrome 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 3. If the information is available, the table ...
2. `medquad_ans_0004586-1` (score=0.6976) - What are the signs and symptoms of Orofaciodigital syndrome 11? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 11. If the information is available, the tabl...
3. `medquad_ans_0004593-1` (score=0.6621) - What are the signs and symptoms of Orofaciodigital syndrome 8? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 8. If the information is available, the table ...
4. `medquad_ans_0004592-1` (score=0.6507) - What are the signs and symptoms of Orofaciodigital syndrome 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 5. If the information is available, the table ...
5. `medquad_ans_0004589-2` (score=0.6441) - What are the signs and symptoms of Orofaciodigital syndrome 2? Although the signs and symptoms that occur in people with orofaciodigital syndrome type 2 may vary, the following findings may be present:Facial findings ...

**Answer**:

The provided context does not contain information about the symptoms of Orofaciodigital syndrome 1. [medquad_ans_0004590-1] and [medquad_ans_0004586-1] discuss Orofaciodigital syndrome 3 and syndrome 11, respectively, but there is no mention of syndrome 1. Therefore, the symptoms of Orofaciodigital syndrome 1 cannot be determined from the given context.

**Supported citations**: `medquad_ans_0004590-1`, `medquad_ans_0004586-1`
**Unsupported citations**: _none_

### Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Scores**: context_relevance=1.000, groundedness=0.905, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.7000) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0002838-5` (score=0.6470) - How might hereditary multiple osteochondromas (HMO) be treated? Currently, there is no known medical treatment for HMO. Osteochondromas are not usually removed because they stop growing around age 12. Another consider...
3. `medquad_ans_0000863-1__dup1` (score=0.1218) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For exa...
4. `medquad_ans_0000286-1__dup2` (score=0.1218) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...
5. `medquad_ans_0000718-1__dup2` (score=0.1065) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts...

**Answer**:

Yes, there is information about surgery in the context provided.

Surgery may be considered for hereditary multiple osteochondromas (HMO) if an osteochondroma is causing pain, bone fracture, nerve irritation, or if the tumor continues to grow after the person's bones have stopped growing. The surgical treatment of choice is complete removal of the tumor, though the difficulty and risk depend on the tumor's location near nerves and blood vessels [medquad_ans_0002838-5]. Surgery may also be necessary to correct painful limb abnormalities, including osteotomy to realign deformed bones or procedures to address leg length discrepancies and forearm deformities [medquad_ans_0002838-5]. In rare cases, if an osteochondroma becomes malignant (e.g., low-grade chondrosarcoma), other therapies such as chemotherapy or radiation may be used [medquad_ans_00028

**Supported citations**: `medquad_ans_0002838-5`
**Unsupported citations**: _none_

### Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Scores**: context_relevance=0.875, groundedness=0.979, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0002935-4` (score=0.7000) - How is human T-cell leukemia virus, type 2 diagnosed? Human T-cell leukemia virus, type 2 (HTLV-2) is usually diagnosed based on blood tests that detect antibodies to the virus. However, HTLV-2 is often never suspecte...
2. `medquad_ans_0002935-1` (score=0.6719) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affe...
3. `medquad_ans_0002935-5` (score=0.6701) - How might human T-cell leukemia virus, type 2 be treated? No cure or treatment exists for human T-cell leukemia virus, type 2 (HTLV-2). Management is focused on early detection and preventing the spread of HTLV-2 to o...
4. `medquad_ans_0002934-3` (score=0.6514) - What causes human T-cell leukemia virus, type 1? Human T-cell leukemia virus, type 1 (HTLV-1) occurs when a person is infected by the human T-cell leukemia retrovirus. HTLV-1 is spread by blood transfusions, sexual co...
5. `medquad_ans_0002934-1` (score=0.6508) - Human T-cell leukemia virus, type 1 (HTLV-1) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, some affected people may later de...

**Answer**:

Human T-cell leukemia virus type 2 (HTLV-2) is a retroviral infection that affects the T cells (a type of white blood cell) [medquad_ans_0002935-1]. Although the virus generally causes no signs or symptoms, scientists suspect that some affected people may later develop neurological problems and/or chronic lung infections [medquad_ans_0002935-1]. HTLV-2 is spread by blood transfusions, sexual contact, and sharing needles, and can also be transmitted from mother to child during birth or breastfeeding [medquad_ans_0002935-1]. There is no cure or treatment for HTLV-2, and it is considered a lifelong condition; however, most infected people remain asymptomatic throughout life [medquad_ans_0002935-1]. Diagnosis is usually based on blood tests that detect antibodies to the

**Supported citations**: `medquad_ans_0002935-1`
**Unsupported citations**: _none_

### Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Scores**: context_relevance=0.833, groundedness=0.977, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000522-1__dup1` (score=0.7000) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...
2. `medquad_ans_0000038-1__dup2` (score=0.6613) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chroni...
3. `medquad_ans_0000038-5__dup1` (score=0.6428) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
4. `medquad_ans_0000038-6` (score=0.6163) - Childhood interstitial lung disease (chILD) has many signs and symptoms because the disease has many forms. Signs and symptoms may include: Fast breathing, which also is called tachypnea (tak-ip-NE-ah) Labored breathi...
5. `medquad_ans_0000758-1__dup2` (score=0.6158) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...

**Answer**:

Interstitial lung diseases (ILD) are a large group of diseases that inflame or scar the lungs, making it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Causes include breathing in dust or other particles, such as coal dust (black lung disease), farm dust (farmer's lung), asbestos fibers (asbestosis), iron from mines or welding fumes (siderosis), and silica dust (silicosis). Other causes include autoimmune diseases or occupational exposures to molds, gases, or fumes. Some types have no known cause. Treatment depends on the type of exposure and disease stage and may involve medicines, oxygen therapy, or lung transplant in severe cases [medquad_ans_0000522-1__dup1]. In children, childhood interstitial lung disease (chILD) is a broad term for rare lung diseases affecting babies, children, and teens, with similar symptoms like chronic cough, rapid

**Supported citations**: `medquad_ans_0000522-1__dup1`
**Unsupported citations**: _none_

### Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.662, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000645-1` (score=0.7000) - Melnick-Needles syndrome is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which also ...
2. `medquad_ans_0003743-4` (score=0.7000) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition w...
3. `medquad_ans_0000645-2` (score=0.6122) - Melnick-Needles syndrome is a rare disorder; fewer than 100 cases have been reported worldwide.
4. `medquad_ans_0002735-4` (score=0.6122) - Is Hashimoto's encephalitis inherited? We are aware of only one instance when more than one person in the same family had Hashimoto's encephalitis (HE). To our knowledge, no other cases of familial HE have been report...
5. `medquad_ans_0005695-4` (score=0.5272) - Is spastic diplegia cerebral palsy inherited? Scientists have found that family members of people with cerebral palsy, including spastic diplegia cerebral palsy, have an increased risk of developing the condition. The...

**Answer**:

The provided context does not contain information about whether Melnick-Needles syndrome is inherited. Therefore, based on the given sources, it is not possible to determine if the condition is inherited.

[medquad_ans_0000645-1] [medquad_ans_0000645-2]

**Supported citations**: `medquad_ans_0000645-1`, `medquad_ans_0000645-2`
**Unsupported citations**: _none_

### Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Scores**: context_relevance=1.000, groundedness=0.977, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006138-5` (score=0.7000) - How is tracheobronchomalacia diagnosed? A diagnosis of tracheobronchomalacia (TBM) may be suspected based on the presence of characteristic signs and symptoms or abnormal pulmonary function tests. Additional testing s...
2. `medquad_ans_0006138-6` (score=0.6449) - How might tracheobronchomalacia be treated? Treatment is only medically necessary in people who have signs and symptoms of tracheobronchomalacia (TBM). Management of symptomatic TBM first involves identifying underlyi...
3. `medquad_ans_0006138-1` (score=0.6449) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of ...
4. `medquad_ans_0006138-3` (score=0.6351) - What causes tracheobronchomalacia? The underlying cause of tracheobronchomalacia (TBM) varies by subtype. Most cases of primary TBM (also called congenital TBM) are caused by genetic conditions that weaken the walls o...
5. `medquad_ans_0006138-2` (score=0.6273) - What are the signs and symptoms of tracheobronchomalacia? Tracheobronchomalacia (TBM) is a condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway ...

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse [medquad_ans_0006138-1]. There are two forms of TBM: a congenital form (called primary TBM) that typically develops during infancy or early childhood, and an acquired form (called secondary TBM) that is usually seen in adults [medquad_ans_0006138-1]. Most cases of primary TBM are caused by genetic conditions that weaken the walls of the airway, while the secondary form often results from trauma, chronic inflammation, and/or prolonged compression of the airways [medquad_ans_0006138-1]. The condition is typically progressive and may lead to characteristic symptoms such as shortness of breath

**Supported citations**: `medquad_ans_0006138-1`
**Unsupported citations**: _none_

### Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Scores**: context_relevance=1.000, groundedness=0.993, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006030-1` (score=0.7000) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people m...
2. `medquad_ans_0000371-1__dup1` (score=0.5990) - What are the signs and symptoms of Ankylosis of teeth? The Human Phenotype Ontology provides the following list of signs and symptoms for Ankylosis of teeth. If the information is available, the table below includes h...
3. `medquad_ans_0000880-1__dup2` (score=0.4901) - The temporomandibular joint (TMJ) connects your jaw to the side of your head. When it works well, it enables you to talk, chew, and yawn. For people with TMJ dysfunction, problems with the joint and muscles around it ...
4. `medquad_ans_0000086-1` (score=0.4212) - Auriculo-condylar syndrome is a condition that affects facial development, particularly development of the ears and lower jaw (mandible). Most people with auriculo-condylar syndrome have malformed outer ears ("auricul...
5. `medquad_ans_0004625-1` (score=0.2727) - Osteomesopyknosis is a bone disorder characterized by abnormal hardening of bone (osteosclerosis). It is generally limited to the axial spine, pelvis, and proximal part of the long bones, which is what distinguishes t...

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene, and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

### Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.493, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000554-2` (score=0.7000) - Kallmann syndrome is estimated to affect 1 in 10,000 to 86,000 people and occurs more often in males than in females. Kallmann syndrome 1 is the most common form of the disorder.
2. `medquad_ans_0003845-4` (score=0.7000) - Is McCune Albright syndrome inherited? McCune Albright syndrome (MAS) is not inherited. It is caused by a random change (mutation) in the GNAS gene that occurs very early in development. As a result, some of the body'...
3. `medquad_ans_0000631-4` (score=0.6122) - McCune-Albright syndrome is not inherited. Instead, it is caused by a random mutation in the GNAS gene that occurs very early in development. As a result, some of the body's cells have a normal version of the GNAS gen...
4. `medquad_ans_0003339-3` (score=0.6122) - How is Kallmann syndrome inherited? Kallmann syndrome (KS) may be inherited in an X-linked recessive, autosomal dominant, or autosomal recessive manner depending on the gene(s) responsible. For example: KS due to muta...
5. `medquad_ans_0004143-4` (score=0.5272) - How is Mowat-Wilson inherited? Mowat-Wilson syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of Mowat-Wilson...

**Answer**:

Kallmann syndrome may be inherited in an X-linked recessive, autosomal dominant, or autosomal recessive manner depending on the gene(s) responsible. For example, Kallmann syndrome 1, caused by mutations in the KAL1 (ANOS1) gene, is inherited in an X-linked recessive manner. KS due to mutations in FGFR1, PROKR2, PROK2, CHD7, or FGF8 genes (causing KS types 2, 3, 4, 5, and 6, respectively) is predominantly inherited in an autosomal dominant manner. Mutations in PROKR2 and PROK2 can also be inherited in an autosomal recessive manner. In the majority of cases, the family history appears negative (sporadic occurrence), but affected individuals are at risk of passing the mutation to their children, with the risk varying by genetic cause and potentially up to 5

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Scores**: context_relevance=1.000, groundedness=0.895, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005967-2` (score=0.7000) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how...
2. `medquad_ans_0005970-1` (score=0.6705) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how...
3. `medquad_ans_0005022-1` (score=0.5985) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the tabl...
4. `medquad_ans_0005967-1` (score=0.5940) - Syndactyly type 3 (SD3) is a limb abnormality present at birth that is characterized by complete fusion of the 4th and 5th fingers on both hands. In most cases only the soft tissue is fused, but in some cases the bone...
5. `medquad_ans_0006385-1` (score=0.5139) - What are the signs and symptoms of Waardenburg syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Waardenburg syndrome type 3. If the information is available, the tabl...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and autosomal dominant inheritance [medquad_ans_0005967-2].

**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

### Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Scores**: context_relevance=0.800, groundedness=0.881, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000197-1__dup3` (score=0.7000) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...
2. `medquad_ans_0000021-1__dup3` (score=0.6906) - Espaol Bronchitis (bron-KI-tis) is a condition in which the bronchial tubes become inflamed. These tubes carry air to your lungs. (For more information about the bronchial tubes and airways, go to the Diseases and Con...
3. `medquad_ans_0000021-5__dup1` (score=0.6543) - Acute Bronchitis Acute bronchitis caused by an infection usually develops after you already have a cold or the flu. Symptoms of a cold or the flu include sore throat, fatigue (tiredness), fever, body aches, stuffy or ...
4. `medquad_ans_0000021-4__dup2` (score=0.6036) - Bronchitis is a very common condition. Millions of cases occur every year. Elderly people, infants, and young children are at higher risk for acute bronchitis than people in other age groups. People of all ages can de...
5. `medquad_ans_0000021-3__dup2` (score=0.5872) - Acute Bronchitis Infections or lung irritants cause acute bronchitis. The same viruses that cause colds and the flu are the most common cause of acute bronchitis. Sometimes bacteria can cause the condition. Certain su...

**Answer**:

Chronic bronchitis is an inflammation of the bronchial tubes, the airways that carry air to the lungs, characterized by a persistent cough that produces mucus. It is one type of chronic obstructive pulmonary disease (COPD) and is often caused by cigarette smoking or long-term exposure to air pollution, fumes, or dust. The inflamed bronchial tubes produce excessive mucus, leading to coughing and difficulty breathing. It is a long-term condition that tends to recur or never fully resolves. Diagnosis involves evaluating symptoms, listening to breathing, and possibly other tests. Treatment includes medications to open airways and clear mucus, oxygen therapy if needed, and pulmonary rehabilitation to improve daily management [medquad_ans_0000197-1__dup3].

**Supported citations**: `medquad_ans_0000197-1__dup3`
**Unsupported citations**: _none_

### Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Scores**: context_relevance=0.875, groundedness=0.780, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000935-1` (score=0.7000) - Spinal muscular atrophy is a genetic disorder that affects the control of muscle movement. It is caused by a loss of specialized nerve cells, called motor neurons, in the spinal cord and the part of the brain that is ...
2. `medquad_ans_0000840-1__dup2` (score=0.6497) - Spinal muscular atrophy (SMA) is a genetic disease that attacks nerve cells, called motor neurons, in the spinal cord. These cells communicate with your voluntary muscles - the ones you can control, like in your arms ...
3. `medquad_ans_0000934-5` (score=0.5959) - These resources address the diagnosis or management of spinal and bulbar muscular atrophy: - Gene Review: Gene Review: Spinal and Bulbar Muscular Atrophy - Genetic Testing Registry: Bulbo-spinal atrophy X-linked - Med...
4. `medquad_ans_0000936-5` (score=0.5518) - These resources address the diagnosis or management of spinal muscular atrophy with progressive myoclonic epilepsy: - Genetic Testing Registry: Jankovic Rivera syndrome - Muscular Dystrophy Association: Spinal Muscula...
5. `medquad_ans_0000935-5` (score=0.5212) - These resources address the diagnosis or management of spinal muscular atrophy: - Gene Review: Gene Review: Spinal Muscular Atrophy - Gene Review: Gene Review: Spinal Muscular Atrophy, X-Linked Infantile - Genetic Tes...

**Answer**:

The provided context does not mention any treatments for spinal muscular atrophy. [medquad_ans_0000935-1] describes the types, symptoms, and progression of the disorder but does not include information on available treatments.

**Supported citations**: `medquad_ans_0000935-1`
**Unsupported citations**: _none_

### Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Scores**: context_relevance=0.889, groundedness=0.899, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000001_1-7` (score=0.7000) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - C...
2. `medquad_ans_0000001_1-1` (score=0.6908) - Key Points - Adult acute lymphoblastic leukemia (ALL) is a type of cancer in which the bone marrow makes too many lymphocytes (a type of white blood cell). - Leukemia may affect red blood cells, white blood cells, and...
3. `medquad_ans_0000001_6-7` (score=0.6800) - Key Points - There are different types of treatment for childhood acute lymphoblastic leukemia (ALL). - Children with ALL should have their treatment planned by a team of doctors who are experts in treating childhood ...
4. `medquad_ans_0000001_2-7` (score=0.5766) - Key Points - There are different types of treatment for patients with adult acute myeloid leukemia. - The treatment of adult AML usually has 2 phases. - Four types of standard treatment are used: - Chemotherapy - Radi...
5. `medquad_ans_0000009-1__dup3` (score=0.5525) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. Thes...

**Answer**:

The treatments for Adult Acute Lymphoblastic Leukemia (ALL) include chemotherapy, radiation therapy, chemotherapy with stem cell transplant, targeted therapy, and biologic therapy. The treatment is typically divided into two phases: remission induction therapy, which aims to kill leukemia cells in the blood and bone marrow to achieve remission, and post-remission therapy, which targets any remaining leukemia cells to prevent relapse. Central nervous system (CNS) sanctuary therapy is also commonly given during each phase to address leukemia cells that may hide in the brain and spinal cord. Additionally, patients may consider participating in clinical trials, which are research studies testing new or improved treatments. [medquad_ans_0000001_1-7]

**Supported citations**: `medquad_ans_0000001_1-7`
**Unsupported citations**: _none_

### Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000388-1` (score=0.7000) - Friedreich ataxia is a genetic condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features of...
2. `medquad_ans_0000388-2` (score=0.6719) - Friedreich ataxia is estimated to affect 1 in 40,000 people. This condition is found in people with European, Middle Eastern, or North African ancestry. It is rarely identified in other ethnic groups.
3. `medquad_ans_0002439-3` (score=0.6616) - What causes Friedreich ataxia? Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known...
4. `medquad_ans_0000388-5` (score=0.6607) - These resources address the diagnosis or management of Friedreich ataxia: - Friedreich's Ataxia Research Alliance: Clinical Care Guidelines - Gene Review: Gene Review: Friedreich Ataxia - Genetic Testing Registry: Fri...
5. `medquad_ans_0002439-2` (score=0.6060) - What are the signs and symptoms of Friedreich ataxia? Symptoms usually begin between the ages of 5 and 15 but can, on occasion, appear in adulthood or even as late as age 75. The first symptom to appear is usually dif...

**Answer**:

Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known as a GAA trinucleotide repeat. Normally, this segment is repeated 5 to 33 times within the FXN gene. In people with Friedreich ataxia, the GAA segment is repeated 66 to more than 1,000 times. The length of the GAA trinucleotide repeat appears to be related to the age at which the symptoms of Friedreich ataxia appear. The abnormally long GAA trinucleotide repeat disrupts the production of frataxin, which severely reduces the amount of this protein in cells. Certain nerve and muscle cells cannot function properly with a shortage of frataxin, leading to the characteristic signs and symptoms of Friedreich ataxia [med

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000013-4__dup2` (score=0.7000) - Atrial fibrillation (AF) occurs if the heart's electrical signals don't travel through the heart in a normal way. Instead, they become very rapid and disorganized. Damage to the heart's electrical system causes AF. Th...
2. `medquad_ans_0000013-6` (score=0.6679) - Atrial fibrillation (AF) usually causes the heart's lower chambers, the ventricles, to contract faster than normal. When this happens, the ventricles can't completely fill with blood. Thus, they may not be able to pum...
3. `medquad_ans_0000070-1__dup4` (score=0.6536) - An arrhythmia is a problem with the speed or rhythm of the heartbeat. Atrial fibrillation (AF) is the most common type of arrhythmia. The cause is a disorder in the heart's electrical system. Often, people who have AF...
4. `medquad_ans_0000013-5__dup1` (score=0.5135) - Atrial fibrillation (AF) affects millions of people, and the number is rising. Men are more likely than women to have the condition. In the United States, AF is more common among Whites than African Americans or Hispa...
5. `medquad_ans_0000008-5__dup1` (score=0.5117) - Arrhythmias are very common in older adults. Atrial fibrillation (a common type of arrhythmia that can cause problems) affects millions of people, and the number is rising. Most serious arrhythmias affect people older...

**Answer**:

The provided context does not mention familial atrial fibrillation. Therefore, it is not possible to answer the question based on the given information. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.7000) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the assoc...
2. `medquad_ans_0004627-2` (score=0.6800) - What are the signs and symptoms of Osteopathia striata cranial sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopathia striata cranial sclerosis. If the information is...
3. `medquad_ans_0003331-1` (score=0.3665) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
4. `medquad_ans_0000627-1__dup2` (score=0.3152) - Multiple sclerosis (MS) is a nervous system disease that affects your brain and spinal cord. It damages the myelin sheath, the material that surrounds and protects your nerve cells. This damage slows down or blocks me...
5. `medquad_ans_0000088-3__dup1` (score=0.1474) - The prognosis for craniosynostosis varies depending on whether single or multiple cranial sutures are involved or other abnormalities are present. The prognosis is better for those with single suture involvement and n...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary significantly from person to person, even within a single family. Features of the condition are generally present at birth and may include skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected people may also have developmental delay, hearing loss, heart defects, and/or ophthalmoplegia (paralysis of the muscles surrounding the eyes). Osteopathia striata cranial sclerosis is caused by changes (mutations) in the WTX gene and is inherited in an X-linked dominant manner. Treatment is based on the signs and symptoms present in each person [medquad_ans_0004

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0004097-5` (score=0.7000) - How is mixed connective tissue disease diagnosed? Mixed connective tissue disease (MCTD) is often suspected after a physical examination reveals signs and symptoms associated with the condition. The diagnosis is suppo...
2. `medquad_ans_0004097-2` (score=0.6716) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus eryt...
3. `medquad_ans_0004097-1` (score=0.6447) - Mixed connective tissue disease (MCTD) is a rare autoimmune disorder that is characterized by features commonly seen in three different connective tissue disorders: systemic lupus erythematosus, scleroderma, and polym...
4. `medquad_ans_0004097-6` (score=0.6270) - How might mixed connective tissue disease be treated? There is currently no cure for mixed connective tissue disease (MCTD). However, treatments can help manage symptoms of the condition. For example, medications such...
5. `medquad_ans_0000230-1__dup2` (score=0.6112) - Connective tissue is the material inside your body that supports many of its parts. It is the "cellular glue" that gives your tissues their shape and helps keep them strong. It also helps some of your tissues do their...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. A condition called Raynaud's phenomenon sometimes occurs months or years before other symptoms of MCTD develop. Most people with MCTD have pain in multiple joints, and/or inflammation of joints (arthritis). Muscle weakness, fevers, and fatigue are also common. Other signs and symptoms may include: accumulation of fluid in the tissue of the hands that causes puffiness and swelling (edema), skin findings including lupus-like rashes (including reddish brown patches), reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia), dysfunction of the esophagus

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Scores**: context_relevance=1.000, groundedness=0.878, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000024-1__dup2` (score=0.7000) - Caring for someone who has Alzheimer's disease (AD) can be stressful and overwhelming. It's important to take care of yourself. Ask for and accept help. Talk to the doctor. Find out what treatments might help control ...
2. `medquad_ans_0000004-5__dup2` (score=0.6908) - Medications Can Treat Symptoms There is no known cure for Alzheimer's disease, but there are medicines that can treat symptoms of the disease. Most Alzheimers medicines work best for people in the mild or moderate sta...
3. `medquad_ans_0000003-6__dup1` (score=0.6615) - Most people with Alzheimers disease are cared for at home by family members. Within families, caregiving is provided most often by wives and husbands, followed by daughters. As Alzheimers disease gets worse, the perso...
4. `medquad_ans_0000004-32` (score=0.5749) - People with Alzheimer's disease, those with mild cognitive impairment, those with a family history of Alzheimers, and healthy people with no memory problems who want to help scientists test new treatments may be able ...
5. `medquad_ans_0000004-24` (score=0.5691) - Currently, no medicines or treatments are known to prevent Alzheimer's disease, but scientists are studying many possibilities. These possibilities include lifestyle factors such as exercise and physical activity, a h...

**Answer**:

Treatments for Alzheimer's caregiving include taking care of oneself by asking for and accepting help, talking to a doctor to learn about treatments that may control symptoms or behavior problems, joining a support group for understanding and emotional support, and utilizing adult day care or respite services to provide breaks and peace of mind [medquad_ans_0000024-1__dup2]. Caregivers should also begin planning for the future, such as organizing financial and legal documents, exploring assisted living or nursing home options, and understanding health insurance and Medicare coverage [medquad_ans_0000024-1__dup2]. Additionally, medications such as donepezil, rivastigmine, galantamine, and memantine may help manage symptoms like memory loss and behavioral issues, though they do not stop the disease and may have side effects [medquad_ans_0000004-5\_\_dup

**Supported citations**: `medquad_ans_0000024-1__dup2`
**Unsupported citations**: _none_

### Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Scores**: context_relevance=0.800, groundedness=0.925, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000293-4` (score=0.7000) - Distal myopathy 2 is inherited in an autosomal dominant pattern, which means one copy of the altered MATR3 gene in each cell is sufficient to cause the disorder.
2. `medquad_ans_0000511-4` (score=0.7000) - Most cases of idiopathic inflammatory myopathy are sporadic, which means they occur in people with no history of the disorder in their family. However, several people with idiopathic inflammatory myopathy have had clo...
3. `medquad_ans_0000511-3` (score=0.6122) - Idiopathic inflammatory myopathy is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Researchers have identi...
4. `medquad_ans_0003179-1` (score=0.6122) - Inclusion body myopathy 2, also known as hereditary inclusion body myopathy (HIBM), GNE-related myopathy, distal myopathy with rimmed vacuoles, and Nonaka myopathy, is an inherited condition that primarily affects the...
5. `medquad_ans_0003141-1` (score=0.5272) - Idiopathic inflammatory myopathy refers to a group of conditions that affect the skeletal muscles (muscles used for movement). Although the condition can be diagnosed at any age, idiopathic inflammatory myopathy most ...

**Answer**:

The genetic changes related to idiopathic inflammatory myopathy include variations in several genes, most commonly within the human leukocyte antigen (HLA) complex. These HLA gene variations may influence the risk of developing the disorder. Researchers have identified specific variations of several HLA genes that seem to affect the risk of idiopathic inflammatory myopathy. It is likely that specific genetic variations increase a person's risk, and environmental factors may then trigger the disorder [medquad_ans_0000511-3]. However, idiopathic inflammatory myopathy does not have a clear pattern of inheritance, and inheriting a genetic variation does not guarantee the development of the condition [medquad_ans_0000511-4].

**Supported citations**: `medquad_ans_0000511-3`, `medquad_ans_0000511-4`
**Unsupported citations**: _none_

### Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.550, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000855-5` (score=0.7000) - These resources address the diagnosis or management of pyruvate dehydrogenase deficiency: - Genetic Testing Registry: Pyruvate dehydrogenase E1-beta deficiency - Genetic Testing Registry: Pyruvate dehydrogenase E2 def...
2. `medquad_ans_0000855-2` (score=0.6680) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
3. `medquad_ans_0005223-5` (score=0.6305) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for...
4. `medquad_ans_0005223-6` (score=0.5933) - How might pyruvate dehydrogenase deficiency be treated? Treatment of pyruvate dehydrogenase (PDH) deficiency rarely influences the course of the disease, but goals include stimulating the pyruvate dehydrogenase comple...
5. `medquad_ans_0000855-1` (score=0.5569) - Pyruvate dehydrogenase deficiency is characterized by the buildup of a chemical called lactic acid in the body and a variety of neurological problems. Signs and symptoms of this condition usually first appear shortly ...

**Answer**:

The diagnosis of pyruvate dehydrogenase (PDH) deficiency may be considered in any individual with early-onset neurological disease, especially if it appears to be associated with structural abnormalities in the brain and unexplained lactic acidosis. When lactic acid (also called lactate) and pyruvate in the blood do not seem to be significantly high, an important clue to the diagnosis may be high concentrations of lactate and/or pyruvate in the cerebrospinal fluid (the fluid that surrounds the brain and spinal cord). Additionally, magnetic resonance spectroscopy (MRS) of the brain may show concentrations of lactate in the central nervous system. Analysis of serum and urine amino acids usually shows hyperalaninemia (high levels of the amino acid alanine). When lactic acidosis is present, other disorders involving pyruvate abnormalities are part of the differential diagnosis. However, in all of these conditions,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006562-2` (score=0.7000) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb a...
2. `medquad_ans_0006554-2` (score=0.5166) - What are the signs and symptoms of 15q11.2 microdeletion? The signs and symptoms in people with a 15q11.2 microdeletion can vary widely. Some people with the microdeletion don't have any apparent features, while other...
3. `medquad_ans_0003415-1` (score=0.5102) - Koolen de Vries syndrome, formerly known as 17q21.31 microdeletion syndrome, is a condition caused by a small deletion of genetic material from chromosome 17. The deletion occurs at a location designated as q21.31. Pe...
4. `medquad_ans_0006607-2` (score=0.4665) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
5. `medquad_ans_0006562-3` (score=0.3938) - What causes 17q23.2q23.2 microdeletion syndrome? The syndrome is caused by an interstitial deletion (a deletion that does not involve the ends of a chromosome) encompassing bands 23.1 to 23.2 on the long (q) arm of ch...

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

**Scores**: context_relevance=1.000, groundedness=0.942, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003860-3` (score=0.7000) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
2. `medquad_ans_0003860-1` (score=0.6290) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...
3. `medquad_ans_0000635-5` (score=0.5912) - These resources address the diagnosis or management of MECP2 duplication syndrome: - Cincinnati Children's Hospital: MECP2-Related Disorders - Cleveland Clinic: Spasticity - Gene Review: Gene Review: MECP2 Duplication...
4. `medquad_ans_0000635-3` (score=0.5747) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
5. `medquad_ans_0000635-1` (score=0.5408) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infan...

**Answer**:

MECP2 duplication syndrome is caused by an extra copy (duplication) of the MECP2 gene in each cell, which results from a duplication of genetic material located on the long (q) arm of the X chromosome [medquad_ans_0003860-3]. The size of the duplication can vary, ranging from 100,000 to 900,000 base pairs (100 to 900 kilobases), but this variation does not affect the severity of the condition [medquad_ans_0000635-3]. The MECP2 gene encodes a protein essential for normal brain function, and its duplication leads to excess protein that cannot properly regulate the expression of other genes, resulting in irregular brain activity and the signs and symptoms of the syndrome [medquad_ans_0003860-3].

**Supported citations**: `medquad_ans_0003860-3`, `medquad_ans_0000635-3`
**Unsupported citations**: _none_

### Example 187 (`0004633-2`)

**Question**: What are the symptoms of Osteopetrosis autosomal dominant type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004633-2` (score=0.7000) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 2. If the informatio...
2. `medquad_ans_0004632-2` (score=0.6994) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 1. If the informatio...
3. `medquad_ans_0000759-1` (score=0.6445) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of ...
4. `medquad_ans_0000759-2` (score=0.6303) - Autosomal dominant osteopetrosis is the most common form of the disorder, affecting about 1 in 20,000 people. Autosomal recessive osteopetrosis is rarer, occurring in an estimated 1 in 250,000 people. Other forms of o...
5. `medquad_ans_0004640-2` (score=0.6288) - What are the signs and symptoms of Osteopetrosis autosomal recessive 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 7. If the information is ava...

**Answer**:

The symptoms of Osteopetrosis autosomal dominant type 2 include abnormality of epiphysis morphology, abnormality of the metacarpal bones, abnormality of the metaphyses, aseptic necrosis, bone pain, facial palsy, frontal bossing, joint dislocation, macrocephaly, osteoarthritis, osteomyelitis, recurrent fractures, short distal phalanx of finger, anemia, genu valgum, optic atrophy, short stature, visual impairment, abnormality of leukocytes, carious teeth, hearing impairment, hydrocephalus, hypocalcemia, bone marrow hypocellularity, abnormality of pelvic girdle bone morphology, abnormality of the vertebral endplates, autosomal dominant inheritance, elevated serum acid phosphatase, facial paralysis, fractures of the long bones, generalized osteosclerosis, hip osteoarthritis, juvenile onset, mandibular osteomy

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 188 (`0003459-4`)

**Question**: What are the treatments for Lambert Eaton myasthenic syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.556

**Top docs**:

1. `medquad_ans_0003459-1` (score=0.7000) - Lambert Eaton myasthenic syndrome (LEMS) is a disorder of the neuromuscular junction. The neuromuscular junction is the site where nerve cells meet muscle cells and help activate the muscles. This syndrome occurs when...
2. `medquad_ans_0003459-3` (score=0.6919) - What causes Lambert Eaton myasthenic syndrome? Lambert Eaton myasthenic syndrome is the result of an autoimmune process which causes a disruption of electrical impulses between nerve cells and muscle fibers. In cases ...
3. `medquad_ans_0003459-2` (score=0.6433) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficu...
4. `medquad_ans_0003459-4` (score=0.6413) - How might Lambert-Eaton myasthenic syndrome be treated? Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, ...
5. `medquad_ans_0000084-2__dup1` (score=0.3911) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...

**Answer**:

Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, plasmapheresis (where blood plasma is removed and replaced with fluid, protein, or donated plasma) or IV immunoglobulins, steroids (e.g., prednisone), azathioprine or cyclosporine, and/or 3,4-diaminopyridine. 3,4-diaminopyridine is available in Europe and may be available in the U.S. on a compassionate use basis. While there has been some evidence that either 3,4-diaminopyridine or IV immunoglobulin can improve muscle strength and nerve to muscle cell communication, the degree of benefit (i.e., how much symptoms are improved) still needs to be determined [medquad_ans_00

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000445-1__dup1` (score=0.7000) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...
2. `medquad_ans_0000065-1` (score=0.6797) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down ...
3. `medquad_ans_0000445-2__dup1` (score=0.6712) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes...
4. `medquad_ans_0006569-2` (score=0.3402) - What are the signs and symptoms of 21-hydroxylase deficiency? Symptoms can vary greatly from patient to patient with 21-hydroxylase deficiency, as a result distinct forms of this deficiency have been recognized. Three...
5. `medquad_ans_0003276-1` (score=0.2678) - What are the signs and symptoms of Isolated growth hormone deficiency type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Isolated growth hormone deficiency type 3. If the inform...

**Answer**:

The symptoms of Arginase deficiency include growth deficiency, spasticity (abnormal tensing of the muscles), developmental delay, loss of developmental milestones, intellectual disability, seizures, and microcephaly [medquad_ans_0000445-1__dup1]. Additionally, stiffness, especially in the legs, caused by abnormal tensing of the muscles (spasticity), slower than normal growth, developmental delay, eventual loss of developmental milestones, intellectual disability, seizures, tremor, and difficulty with balance and coordination (ataxia) may occur [medquad_ans_0000065-1]. Occasionally, high protein meals or stress caused by illness or periods without food (fasting) may lead to episodes of irritability, refusal to eat, and vomiting [medquad_ans_0000065-1].

**Supported citations**: `medquad_ans_0000445-1__dup1`, `medquad_ans_0000065-1`
**Unsupported citations**: _none_

### Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Scores**: context_relevance=0.889, groundedness=0.949, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006376-6` (score=0.7000) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but ar...
2. `medquad_ans_0000961-1__dup1` (score=0.6546) - Von Hippel-Lindau disease (VHL) is a rare, genetic disease that causes tumors and cysts to grow in your body. The tumors can be either cancerous or benign. They can grow in your brain and spinal cord, kidneys, pancrea...
3. `medquad_ans_0006376-5` (score=0.6540) - How is von Hippel-Lindau (VHL) disease diagnosed? The diagnosis of von Hippel-Lindau (VHL) disease can be made based on specific clinical criteria (signs and symptoms), or when molecular genetic testing reveals a muta...
4. `medquad_ans_0006376-3` (score=0.6463) - What causes Von Hippel-Lindau disease? Von Hippel-Lindau (VHL) disease is caused by a mutation in the VHL gene. This gene is a tumor suppressor gene, which helps to control cell growth. Mutations in the VHL gene lead ...
5. `medquad_ans_0000022-1__dup1` (score=0.6366) - von Hippel-Lindau disease (VHL) is a rare, genetic multi-system disorder in which non-cancerous tumors grow in certain parts of the body. Slow-growing hemgioblastomas -- benign tumors with many blood vessels -- may de...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. The goal is to treat growths when they cause symptoms but are still small to prevent permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All people with VHL disease should be carefully followed by a physician or medical team familiar with the disorder [medquad_ans_0006376-6]. Additionally, treatment may include surgery or sometimes radiation therapy, with the aim of addressing tumors early while they are small [medquad_ans_0000961-1__dup1].

**Supported citations**: `medquad_ans_0006376-6`, `medquad_ans_0000961-1__dup1`
**Unsupported citations**: _none_

### Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Scores**: context_relevance=0.800, groundedness=0.286, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000738-5` (score=0.7000) - These resources address the diagnosis or management of Noonan syndrome: - Gene Review: Gene Review: Noonan Syndrome - Genetic Testing Registry: Noonan syndrome - Genetic Testing Registry: Noonan syndrome 1 - Genetic T...
2. `medquad_ans_0000738-1` (score=0.6804) - Noonan syndrome is a condition that affects many areas of the body. It is characterized by mildly unusual facial characteristics, short stature, heart defects, bleeding problems, skeletal malformations, and many other...
3. `medquad_ans_0004452-2` (score=0.6227) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004453-3` (score=0.6075) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004450-4` (score=0.5916) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not define or describe "Noonan syndrome 3." Therefore, it is not possible to answer the question based on the given information. [medquad_ans_0000738-5] lists resources related to Noonan syndrome, including multiple entries in the Genetic Testing Registry, but does not specify what "Noonan syndrome 3" entails. [doc_id]medquad_ans_0000738-1[/doc_id] describes general features of Noonan syndrome but does not mention a specific subtype such as Noonan syndrome 3.

**Supported citations**: `medquad_ans_0000738-5`
**Unsupported citations**: `doc_id`, `/doc_id`

### Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Scores**: context_relevance=1.000, groundedness=0.522, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0005865-2` (score=0.7000) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may...
2. `medquad_ans_0005865-3` (score=0.6902) - How might Sprengel deformity be treated? Treatment of Sprengel deformity depends on the degree of functional impairment and/or cosmetic disfigurement. Many people with Sprengel deformity do not need surgery and may ha...
3. `medquad_ans_0005865-1` (score=0.6800) - Sprengel deformity is a congenital condition characterized by abnormal development and elevation of the shoulder blade (scapula). Severity can range considerably from being almost invisible when covered with clothes, ...
4. `medquad_ans_0005501-1` (score=0.2980) - What are the signs and symptoms of Schneckenbecken dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Schneckenbecken dysplasia. If the information is available, the table be...
5. `medquad_ans_0002062-1` (score=0.2521) - What are the signs and symptoms of Ehlers-Danlos syndrome, spondylocheirodysplastic type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, spondylocheirodyspl...

**Answer**:

The signs and symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additional features that may be present include bone, cartilage, or fibrous tissue between the shoulder blade and the spinal bones (vertebrae), scoliosis, Klippel-Feil syndrome, limb length discrepancy, hemivertebrae, missing, fused, or extra ribs (cervical ribs), abnormalities of the collarbone, abnormalities of the chest organs displaced to the opposite side (e.g., liver on the left and heart on the right), spina bifida occulta, and cleft palate. Abnormality of the shoulder and Sprengel anomaly are present in approximately 90% of patients, while cleft palate

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.894, answer_relevance=0.756

**Top docs**:

1. `medquad_ans_0005415-1` (score=0.7000) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
2. `medquad_ans_0005415-4` (score=0.6997) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...
3. `medquad_ans_0005415-3` (score=0.6833) - How is Robinow syndrome inherited? Robinow syndrome may be inherited in an autosomal recessive or autosomal dominant manner. Autosomal recessive (AR) inheritance means both copies of the responsible gene in each cell ...
4. `medquad_ans_0000882-5` (score=0.6818) - These resources address the diagnosis or management of Robinow syndrome: - Gene Review: Gene Review: Autosomal Dominant Robinow Syndrome - Gene Review: Gene Review: ROR2-Related Robinow Syndrome - Genetic Testing Regi...
5. `medquad_ans_0000882-1` (score=0.6625) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...

**Answer**:

Genetic changes related to Robinow syndrome include mutations in the ROR2 gene associated with autosomal recessive Robinow syndrome and mutations in the WNT5A or DVL1 gene associated with autosomal dominant Robinow syndrome. In some cases, the underlying cause remains unknown despite genetic testing [medquad_ans_0005415-1][medquad_ans_0005415-4].

**Supported citations**: `medquad_ans_0005415-1`, `medquad_ans_0005415-4`
**Unsupported citations**: _none_

### Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Scores**: context_relevance=1.000, groundedness=0.830, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000179-12` (score=0.7000) - Eating healthy foods can help you keep up your energy and strength. All dialysis and transplant centers have a dietitian. The dietitian helps people with kidney failure learn about healthy food choices. You should tal...
2. `medquad_ans_0000158-13` (score=0.6970) - For people who are on dialysis or approaching total kidney failure, adequate nutrition is important for maintaining energy, strength, healthy sleep patterns, bone health, heart health, and good mental health. A person...
3. `medquad_ans_0000180-1__dup3` (score=0.6836) - Kidney failure means your kidneys no longer work well enough to do their job. You need treatment to replace the work your damaged kidneys have stopped doing. The treatments for kidney failure are - hemodialysis - peri...
4. `medquad_ans_0000180-8` (score=0.6575) - Eating the right foods can help you feel better when you are on dialysis or have a kidney transplant. Staying healthy with kidney failure requires watching how much of these elements are included in your diet: - Prote...
5. `medquad_ans_0000138-11` (score=0.6106) - You and your doctor will work together to choose a treatment that's best for you. The publications of the NIDDK Kidney Failure Series can help you learn about the specific issues you will face. Booklets - Treatment Me...

**Answer**:

The provided context does not define or explain "Kidney Failure: Eat Right to Feel Right on Hemodialysis." It mentions that more information about nutrition for people on hemodialysis is provided in the NIDDK health topic, "Eat Right to Feel Right on Hemodialysis," but does not contain the content of that topic. Therefore, based on the given context, the query cannot be answered. [medquad_ans_0000179-12]

**Supported citations**: `medquad_ans_0000179-12`
**Unsupported citations**: _none_

### Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Scores**: context_relevance=0.889, groundedness=0.945, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0000677-2` (score=0.7000) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
2. `medquad_ans_0000677-1` (score=0.6714) - Mucolipidosis type IV is an inherited disorder characterized by delayed development and vision impairment that worsens over time. The severe form of the disorder is called typical mucolipidosis type IV, and the mild f...
3. `medquad_ans_0004153-1` (score=0.6061) - Mucolipidosis type 4 is a metabolic condition that affects the body's ability to process certain carbohydrates and fats. As a result, these materials accumulate in cells leading to the various signs and symptoms of th...
4. `medquad_ans_0000677-5` (score=0.5920) - These resources address the diagnosis or management of mucolipidosis type IV: - Gene Review: Gene Review: Mucolipidosis IV - Genetic Testing Registry: Ganglioside sialidase deficiency - MedlinePlus Encyclopedia: Gastr...
5. `medquad_ans_0000674-1` (score=0.5680) - Mucolipidosis II alpha/beta (also known as I-cell disease) is a progressively debilitating disorder that affects many parts of the body. Most affected individuals do not survive past early childhood. At birth, childre...

**Answer**:

Mucolipidosis type IV is estimated to occur in 1 in 40,000 people [medquad_ans_0000677-2].

**Supported citations**: `medquad_ans_0000677-2`
**Unsupported citations**: _none_

### Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Scores**: context_relevance=1.000, groundedness=0.992, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.7000) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chi...
2. `medquad_ans_0000063-19` (score=0.4060) - Skin cancer is caused by DNA damage, which can result from excess exposure to ultraviolet (UV) light. Having a previous occurrence of skin cancer or a close family member with the disease also puts you at risk for ski...
3. `medquad_ans_0006196-1` (score=0.3150) - Trigeminal trophic syndrome is a rare disease that affects the skin on the side of the nose, supplied by the trigeminal nerve. People with trigeminal trophic syndrome have a loss of sensation in the nose or abnormal s...
4. `medquad_ans_0002125-1` (score=0.2720) - What are the signs and symptoms of Epidermolysis bullosa simplex, Dowling-Meara type? The Human Phenotype Ontology provides the following list of signs and symptoms for Epidermolysis bullosa simplex, Dowling-Meara typ...
5. `medquad_ans_0002062-1` (score=0.0697) - What are the signs and symptoms of Ehlers-Danlos syndrome, spondylocheirodysplastic type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, spondylocheirodyspl...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - how big and deep your wound is, where it is, how long it takes to heal, your age, and your inherited tendency to scar. Scars usually fade over time but never go away completely [medquad_ans_0000795-1__dup1].

**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

### Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.786, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0001045-1` (score=0.7000) - X-linked adrenoleukodystrophy is a genetic disorder that occurs primarily in males. It mainly affects the nervous system and the adrenal glands, which are small glands located on top of each kidney. In this disorder, ...
2. `medquad_ans_0006488-4` (score=0.0765) - How is X-linked adrenal hypoplasia congenita inherited? X-linked adrenal hypoplasia congenita is inherited in an X-linked recessive pattern. A condition is considered X-linked if the mutated gene that causes the disor...
3. `medquad_ans_0001045-5` (score=0.0765) - These resources address the diagnosis or management of X-linked adrenoleukodystrophy: - Gene Review: Gene Review: X-Linked Adrenoleukodystrophy - Genetic Testing Registry: Adrenoleukodystrophy - Genomics Education Pro...
4. `medquad_ans_0006488-1` (score=0.0669) - X-linked adrenal hypoplasia congenita is an inherited disorder that mainly affects males. It involves many hormone-producing (endocrine) tissues in the body, particularly a pair of small glands on top of each kidney c...
5. `medquad_ans_0001045-4` (score=0.0669) - X-linked adrenoleukodystrophy is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each...

**Answer**:

The provided context does not specify the exact genetic changes related to X-linked adrenoleukodystrophy. It describes the disorder's effects on the nervous system and adrenal glands, as well as its types and symptoms, but does not detail the underlying genetic mutations or molecular mechanisms. [medquad_ans_0001045-1]

**Supported citations**: `medquad_ans_0001045-1`
**Unsupported citations**: _none_

### Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Scores**: context_relevance=1.000, groundedness=0.984, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000988-2` (score=0.7000) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of Europe...
2. `medquad_ans_0000988-1` (score=0.7000) - Transthyretin amyloidosis is a slowly progressive condition characterized by the buildup of abnormal deposits of a protein called amyloid (amyloidosis) in the body's organs and tissues. These protein deposits most fre...
3. `medquad_ans_0000136-1__dup4` (score=0.6157) - Amyloidosis is a rare disease that occurs when amyloid proteins are deposited in tissues and organs. Amyloid proteins are abnormal proteins that the body cannot break down and recycle, as it does with normal proteins....
4. `medquad_ans_0000303-1__dup1` (score=0.5489) - What are the signs and symptoms of Amyloidosis familial visceral? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyloidosis familial visceral. If the information is available, the ...
5. `medquad_ans_0000302-1__dup1` (score=0.4914) - What are the signs and symptoms of Amyloidosis corneal? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyloidosis corneal. If the information is available, the table below includes...

**Answer**:

The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence is estimated to be one in 538 people. Among Americans of European descent, it affects one in 100,000 people. The cardiac form affects between 3 percent and 3.9 percent of African Americans and approximately 5 percent of people in some areas of West Africa [medquad_ans_0000988-2].

**Supported citations**: `medquad_ans_0000988-2`
**Unsupported citations**: _none_

### Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Scores**: context_relevance=1.000, groundedness=0.826, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.7000) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the informatio...
2. `medquad_ans_0006016-1` (score=0.0765) - What are the signs and symptoms of Taurodontism, microdontia, and dens invaginatus? The Human Phenotype Ontology provides the following list of signs and symptoms for Taurodontism, microdontia, and dens invaginatus. I...
3. `medquad_ans_0001787-1` (score=0.0765) - What are the signs and symptoms of Dentin dysplasia, type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Dentin dysplasia, type 1. If the information is available, the table belo...
4. `medquad_ans_0005769-1` (score=0.0669) - What are the signs and symptoms of Spinocerebellar ataxia 20? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia 20. If the information is available, the table be...
5. `medquad_ans_0005856-1` (score=0.0669) - What are the signs and symptoms of Spondylometaphyseal dysplasia with dentinogenesis imperfecta? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia with de...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include those listed by the Human Phenotype Ontology (HPO), which compiles information on how often a sign or symptom occurs in individuals with this condition. The frequency of these symptoms is typically estimated as a percentage or presented as a fraction (e.g., 25/25), indicating the number of patients with the symptom out of the total number examined in a specific study. However, the specific symptoms themselves are not detailed in the provided context [medquad_ans_0001780-1].

**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

### Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Scores**: context_relevance=0.875, groundedness=0.746, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000782-1` (score=0.7000) - Periventricular heterotopia is a condition in which nerve cells (neurons) do not migrate properly during the early development of the fetal brain, from about the 6th week to the 24th week of pregnancy. Heterotopia mea...
2. `medquad_ans_0004818-2` (score=0.6997) - What are the signs and symptoms of periventricular nodular heterotopia? The condition is first noticed when seizures appear, often during the teenage years. The nodules around the ventricles are then typically discove...
3. `medquad_ans_0004818-1` (score=0.6710) - Periventricular heterotopia is a condition in which the nerve cells (neurons) do not migrate properly during the early development of the fetal brain from about the 6th week to the 24th week of pregnancy. Affected peo...
4. `medquad_ans_0000782-2` (score=0.6643) - Periventricular heterotopia is a rare condition. Its incidence is unknown.
5. `medquad_ans_0004818-4` (score=0.6092) - How might periventricular nodular heterotopia be treated? Treatment of epilepsy generally follows principles for a seizure disorder caused by a known structural brain abnormality; carbamezipine is most often used, bec...

**Answer**:

The provided context does not specify the number of people affected by periventricular heterotopia. [medquad_ans_0000782-1] and [medquad_ans_0004818-2] describe the condition, its symptoms, and associated features but do not include epidemiological data or prevalence statistics. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000782-1`, `medquad_ans_0004818-2`
**Unsupported citations**: _none_

### Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.837, answer_relevance=0.782

**Top docs**:

1. `medquad_ans_0003462-3` (score=0.7000) - How is Landau-Kleffner syndrome (LKS) diagnosed? LKS is diagnosed based on clinical features and the results of an electroencephalogram (EEG), a recording of the electric activity of the brain. All LKS children have a...
2. `medquad_ans_0000007_2-6` (score=0.1042) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000009_2-8` (score=0.0912) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0003462-1` (score=0.0912) - Landau-Kleffner syndrome (LKS) is a rare, childhood neurological disorder characterized by the sudden or gradual development of aphasia (the inability to understand or express language) and an abnormal electro-encepha...
5. `medquad_ans_0003462-2` (score=0.0785) - What are the signs and symptoms of Landau-Kleffner syndrome? Landau-Kleffner syndrome is characterized by the sudden or gradual development of aphasia (the inability to understand or express language) in previously no...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for Landau-Kleffner Syndrome (LKS). Therefore, based on the available information, no specific research or clinical trials for LKS are described [medquad_ans_0003462-3].

**Supported citations**: `medquad_ans_0003462-3`
**Unsupported citations**: _none_

### Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Scores**: context_relevance=0.800, groundedness=0.923, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001935-1` (score=0.7000) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystroph...
2. `medquad_ans_0004514-4` (score=0.6299) - How is oculopharyngeal muscular dystrophy inherited?
3. `medquad_ans_0002213-1` (score=0.5841) - Facioscapulohumeral muscular dystrophy is a disorder characterized by muscle weakness and wasting (atrophy). This condition gets its name from the areas of the body that are affected most often: muscles in the face (f...
4. `medquad_ans_0000306-1` (score=0.4950) - Muscular dystrophies are a group of genetic conditions characterized by progressive muscle weakness and wasting (atrophy). The Duchenne and Becker types of muscular dystrophy are two related conditions that primarily ...
5. `medquad_ans_0003604-1` (score=0.4797) - Limb-girdle muscular dystrophy type 2A (LGMD2A) is an autosomal recessive limb-girdle muscular dystrophy characterized by progressive, symmetrical weakness of the proximal limb and girdle muscles (mainly those around ...

**Answer**:

Muscular dystrophies are a group of genetic conditions characterized by progressive muscle weakness and wasting (atrophy) [medquad_ans_0002213-1]. The Duchenne and Becker types of muscular dystrophy are two related conditions that primarily affect skeletal muscles used for movement and heart (cardiac) muscle [medquad_ans_0000306-1]. These forms of muscular dystrophy occur almost exclusively in males and are caused by different mutations in the same gene, with Duchenne muscular dystrophy resulting in a more severe and rapidly progressive form, while Becker muscular dystrophy is generally milder and progresses more slowly [medquad_ans_0000306-1]. Duchenne muscular dystrophy is inherited in an X-linked recessive fashion and is caused by a mutation in the DMD gene, leading to the absence of the dystrophin protein in muscles [medquad_ans_00

**Supported citations**: `medquad_ans_0002213-1`, `medquad_ans_0000306-1`
**Unsupported citations**: _none_

### Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.342, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0005019-1` (score=0.7000) - What are the signs and symptoms of Preaxial polydactyly type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 1. If the information is available, the tabl...
2. `medquad_ans_0004948-1` (score=0.6376) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that t...
3. `medquad_ans_0005022-1` (score=0.5851) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the tabl...
4. `medquad_ans_0005020-1` (score=0.5232) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the tabl...
5. `medquad_ans_0005971-1` (score=0.4782) - What are the signs and symptoms of Syndactyly-polydactyly-earlobe syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly-polydactyly-earlobe syndrome. If the informatio...

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding with no other physical anomalies or intellectual impairment, or it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits, with surgical management depending greatly on the complexity of the deformity [1] [2].

**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

### Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Scores**: context_relevance=1.000, groundedness=0.933, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002519-2` (score=0.7000) - What are the signs and symptoms of Geniospasm? The Human Phenotype Ontology provides the following list of signs and symptoms for Geniospasm. If the information is available, the table below includes how often the sym...
2. `medquad_ans_0002519-1` (score=0.7000) - Hereditary geniospasm is a movement disorder that causes episodes of involuntary tremors of the chin and lower lip. The episodes may last anywhere from a few seconds to hours and may occur spontaneously or be brought ...
3. `medquad_ans_0004534-2` (score=0.6122) - What are the signs and symptoms of Omenn syndrome? Infants with Omenn syndrome typically present shortly after birth, usually by 3 months of age. This is similar to other types of severe combined immunodeficiency (SCI...
4. `medquad_ans_0002519-3` (score=0.6122) - How is hereditary geniospasm inherited? Hereditary geniospasm is inherited in an autosomal dominant manner. This means that having only one mutated copy of the causative gene in each body cell is sufficient to cause s...
5. `medquad_ans_0000108-2__dup1` (score=0.5272) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The symptoms of Geniospasm include episodes of involuntary tremors of the chin and lower lip. These episodes may last from a few seconds to hours and can occur spontaneously or be triggered by stress. The episodes typically first appear in infancy or childhood and tend to decrease in frequency with age [medquad_ans_0002519-1]. Chin myoclonus is also listed as a sign and symptom associated with Geniospasm [medquad_ans_0002519-2].

**Supported citations**: `medquad_ans_0002519-1`, `medquad_ans_0002519-2`
**Unsupported citations**: _none_

### Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=0.900, groundedness=0.828, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000078-1__dup2` (score=0.7000) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...
2. `medquad_ans_0000758-1__dup2` (score=0.6356) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
3. `medquad_ans_0000078-4__dup2` (score=0.5815) - Sometimes doctors can find out what is causing pulmonary fibrosis (lung scarring). For example, exposure to environmental pollutants and certain medicines can cause the disease. Environmental pollutants include inorga...
4. `medquad_ans_0000078-5__dup1` (score=0.4039) - The signs and symptoms of idiopathic pulmonary fibrosis (IPF) develop over time. They may not even begin to appear until the disease has done serious damage to your lungs. Once they occur, they're likely to get worse ...
5. `medquad_ans_0000109-4__dup2` (score=0.3168) - Pulmonary hypertension (PH) begins with inflammation and changes in the cells that line your pulmonary arteries. Other factors also can affect the pulmonary arteries and cause PH. For example, the condition may develo...

**Answer**:

The provided context does not specify any genetic changes related to idiopathic pulmonary fibrosis (IPF). It mentions that genetics may play a role in causing IPF and that if more than one member of a family has IPF, it is called familial IPF, but it does not detail specific genetic changes [medquad_ans_0000078-1__dup2]. Therefore, based on the given information, the genetic changes related to IPF cannot be determined.

**Supported citations**: `medquad_ans_0000078-1__dup2`
**Unsupported citations**: _none_

### Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.794, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0003664-1` (score=0.7000) - Romano-Ward syndrome is the most common form of inherited long QT syndrome. Symptoms include arrhythmia, fainting, cardiac arrest, and sudden death. There are six different types of this syndrome, long QT 1 through 6....
2. `medquad_ans_0003593-4` (score=0.1533) - How is Liddle syndrome inherited? Liddle syndrome is inherited in an autosomal dominant manner. This means that only one mutated copy of the disease-causing gene in each cell is sufficient to cause the condition. The ...
3. `medquad_ans_0005008-1` (score=0.1533) - Potocki-Shaffer syndrome is a contiguous gene deletion syndrome associated with deletions in a specific region of chromosome 11 (11p11.2). The characteristic features of Potocki-Shaffer syndrome include openings in th...
4. `medquad_ans_0004750-4` (score=0.1341) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is pass...
5. `medquad_ans_0000813-3` (score=0.1341) - Potocki-Shaffer syndrome (also known as proximal 11p deletion syndrome) is caused by a deletion of genetic material from the short (p) arm of chromosome 11 at a position designated 11p11.2. The size of the deletion va...

**Answer**:

Yes, Potocki-Shaffer syndrome is inherited. It is a contiguous gene deletion syndrome caused by deletions in a specific region of chromosome 11 (11p11.2), and the features result from the loss of several genes on the short (p) arm of chromosome 11 [medquad_ans_0005008-1]. While the context does not explicitly state the inheritance pattern, it describes the condition as resulting from deletions, which are typically inherited or de novo. However, based on the information provided, the syndrome is associated with genetic deletions that occur in a manner consistent with chromosomal inheritance, and no specific pattern (e.g., autosomal dominant) is directly stated. Therefore, the inheritance is implied through the genetic deletion mechanism, but the exact mode is not explicitly detailed in the provided context.

[medquad_ans_0005008-1]

**Supported citations**: `medquad_ans_0005008-1`
**Unsupported citations**: _none_

### Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Scores**: context_relevance=1.000, groundedness=0.400, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000925-1__dup1` (score=0.7000) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
2. `medquad_ans_0000227-5` (score=0.4533) - These resources address the diagnosis or management of CHILD syndrome: - Gene Review: Gene Review: NSDHL-Related Disorders - Genetic Testing Registry: Child syndrome These resources from MedlinePlus offer information ...
3. `medquad_ans_0000816-5` (score=0.4504) - These resources address the diagnosis or management of preeclampsia: - Eunice Kennedy Shriver National Institute of Child Health and Human Development: How Do Health Care Providers Diagnose Preeclampsia, Eclampsia, an...
4. `medquad_ans_0000223-10` (score=0.0907) - You can take the following steps to help prevent your child from getting a UTI: - Teach your child not to hold in urine and to go to the bathroom whenever your child feels the urge. - Teach your child how to properly ...
5. `medquad_ans_0000177-1__dup2` (score=0.0907) - Summary : Children's healthy development depends on safe and positive experiences when they are very young. If you work or go to school, you want to know that your child is in good hands while you are away. You may ch...

**Answer**:

No, there is no information about child care in the provided context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Scores**: context_relevance=0.800, groundedness=0.954, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000046-2` (score=0.7000) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
2. `medquad_ans_0000046-1` (score=0.6908) - Alternating hemiplegia of childhood is a neurological condition characterized by recurrent episodes of temporary paralysis, often affecting one side of the body (hemiplegia). During some episodes, the paralysis altern...
3. `medquad_ans_0000046-5` (score=0.6718) - These resources address the diagnosis or management of alternating hemiplegia of childhood: - The Great Ormond Street Hospital - University of Utah School of Medicine These resources from MedlinePlus offer information...
4. `medquad_ans_0000016-3__dup1` (score=0.6450) - Children with the benign form of alternating hemiplegia have a good prognosis. Those who experience the more severe form have a poor prognosis because intellectual and mental capacities do not respond to drug therapy,...
5. `medquad_ans_0000016-1__dup1` (score=0.6439) - Alternating hemiplegia is a rare neurological disorder that develops in childhood, most often before the child is 18 months old. The disorder is characterized by recurrent episodes of paralysis that involve one or bot...

**Answer**:

Approximately 1 in 1 million people are affected by alternating hemiplegia of childhood [medquad_ans_0000046-2].

**Supported citations**: `medquad_ans_0000046-2`
**Unsupported citations**: _none_

### Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Scores**: context_relevance=1.000, groundedness=0.974, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000159-3__dup2` (score=0.7000) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time...
2. `medquad_ans_0000569-1__dup2` (score=0.6899) - If you have lupus, your immune system attacks healthy cells and tissues by mistake. This can damage your joints, skin, blood vessels and organs. There are many kinds of lupus. The most common type, systemic lupus eryt...
3. `medquad_ans_0000186-1__dup1` (score=0.6621) - Lupus (also called systemic lupus erythematosus) is a disorder of the immune system. Normally, the immune system protects the body against invading infections and cancers. In lupus, the immune system is over-active an...
4. `medquad_ans_0003688-1` (score=0.5728) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, h...
5. `medquad_ans_0003687-2` (score=0.4916) - What are the signs and symptoms of Lupus? You can read about the signs and symptoms of lupus from MedlinePlus and the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS). The Human Phenotype ...

**Answer**:

The symptoms of lupus include joint pain or swelling, muscle pain, fever with no known cause, fatigue, red rashes often on the face (also called the "butterfly rash"), and in some cases, neurological disorders such as mild cognitive dysfunction, organic brain syndrome, peripheral neuropathies, sensory neuropathy, psychological problems (including personality changes, paranoia, mania, and schizophrenia), seizures, transverse myelitis, paralysis, and stroke. Additionally, lupus can cause high blood pressure, foamy urine, and edema (swelling), usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems, such as those seen in lupus nephritis, may develop at the same time or shortly after lupus symptoms appear [medquad_ans_0000159-3__dup2]medquad_ans_0000569-1\_\_dup

**Supported citations**: `medquad_ans_0000159-3__dup2`
**Unsupported citations**: _none_

### Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.973, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0001395-1` (score=0.7000) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
2. `medquad_ans_0000194-1` (score=0.6901) - Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features of this disorder include a failure to gain weight and grow at the expected rate (failure to thrive), a...
3. `medquad_ans_0001398-1` (score=0.6898) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
4. `medquad_ans_0001397-1` (score=0.6811) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
5. `medquad_ans_0000194-2` (score=0.6656) - Cockayne syndrome occurs in about 2 per million newborns in the United States and Europe.

**Answer**:

Cockayne syndrome is a rare condition characterized by short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. It also includes failure to thrive in the newborn, microcephaly, and impaired nervous system development. Other symptoms may include hearing loss, tooth decay, and eye and bone abnormalities. The syndrome is caused by mutations in either the ERCC8 (CSA) or ERCC6 (CSB) genes and is inherited in an autosomal recessive pattern. It can be divided into subtypes based on severity and age of onset: type 1 (classical or "moderate") with onset in early childhood, type 2 (severe or "early-onset") with severe symptoms apparent at birth, and type 3 (milder) with later onset in childhood [medquad_ans_0001395-1]medquad_ans_000019

**Supported citations**: `medquad_ans_0001395-1`
**Unsupported citations**: _none_

### Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Scores**: context_relevance=0.833, groundedness=0.987, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000029-1__dup5` (score=0.7000) - An Infection of the Gums and Surrounding Tissues Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. The two forms of gum disease are gingivitis, a mild form that is...
2. `medquad_ans_0000029-6__dup1` (score=0.6898) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead...
3. `medquad_ans_0000029-8__dup1` (score=0.6817) - Gingivitis is inflammation of the gums. In gingivitis, the gums become red, swollen and can bleed easily. Gingivitis is a mild form of gum disease. It can usually be reversed with daily brushing and flossing, and regu...
4. `medquad_ans_0000029-13` (score=0.6628) - There are a number of risk factors that can increase your chances of developing periodontal disease. - Smoking is one of the most significant risk factors associated with the development of gum disease and can even lo...
5. `medquad_ans_0000029-12` (score=0.6518) - In some studies, researchers have observed that people with periodontal disease (when compared to people without periodontal disease) were more likely to develop heart disease or have difficulty controlling their bloo...

**Answer**:

Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. The two forms of gum disease are gingivitis, a mild form that is reversible with good oral hygiene, and periodontitis, a more severe form that can damage the soft tissues and bone that support teeth. If left untreated, periodontitis can lead to tooth loss [medquad_ans_0000029-1__dup5]. In its early stages, gum disease is usually painless, and many people are not aware that they have it; in more advanced cases, it can cause sore gums, pain when chewing, and even tooth loss [medquad_ans_0000029-6__dup1]. Gum disease is typically caused by poor brushing and flossing habits that allow dental plaque — a sticky film of bacteria — to build up on the teeth, and if not removed,

**Supported citations**: `medquad_ans_0000029-1__dup5`, `medquad_ans_0000029-6__dup1`
**Unsupported citations**: _none_

### Example 212 (`0000494-2`)

**Question**: How many people are affected by hyperkalemic periodic paralysis ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000494-5` (score=0.7000) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis ...
2. `medquad_ans_0000494-2` (score=0.6813) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
3. `medquad_ans_0000494-1` (score=0.6625) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to...
4. `medquad_ans_0000126-1__dup1` (score=0.6443) - Familial periodic paralyses are a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. They are characterized by episodes in which the affect...
5. `medquad_ans_0003000-2` (score=0.6437) - What are the signs and symptoms of Hyperkalemic periodic paralysis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hyperkalemic periodic paralysis. If the information is available, ...

**Answer**:

An estimated 1 in 200,000 people are affected by hyperkalemic periodic paralysis [medquad_ans_0000494-2].

**Supported citations**: `medquad_ans_0000494-2`
**Unsupported citations**: _none_

### Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0005212-1` (score=0.7000) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end o...
2. `medquad_ans_0005212-3` (score=0.0877) - Is Pyle disease inherited? Pyle disease in inherited in an autosomal recessive manner, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive conditio...
3. `medquad_ans_0004813-1` (score=0.0776) - Periodic fever, aphthous stomatitis, pharyngitis, cervical adenitis (PFAPA) is a periodic disease, which is a heterogeneous group of disorders characterized by short episodes of illness that regularly recur for severa...
4. `medquad_ans_0005016-1` (score=0.0679) - Prader-Willi syndrome (PWS) is a genetic condition that affects many parts of the body. Infants with PWS have severe hypotonia (low muscle tone), feeding difficulties, and slow growth. In later infancy or early childh...
5. `medquad_ans_0005212-2` (score=0.0679) - What are the signs and symptoms of Pyle disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyle disease. If the information is available, the table below includes how often the...

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (relative constriction of the diaphysis and flaring of the metaphysis), widening of the ribs and clavicles, platyspondyly (flattening of the spine bones), and cortical thinning. Cranial involvement is minimal, with some individuals showing mild hyperostosis of the skull base and thickening of the frontal and occipital bones. Pyle disease is inherited in an autosomal recessive manner, meaning both copies of the gene in each cell have mutations, and the parents of an affected individual each carry one copy of the mutated gene [medquad_ans_0005212-1][medquad_ans_0005212-3].

**Supported citations**: `medquad_ans_0005212-1`, `medquad_ans_0005212-3`
**Unsupported citations**: _none_

### Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Scores**: context_relevance=0.600, groundedness=0.902, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000051-1__dup2` (score=0.7000) - Espaol Deep vein thrombosis (throm-BO-sis), or DVT, is a blood clot that forms in a vein deep in the body. Blood clots occur when blood thickens and clumps together. Most deep vein blood clots occur in the lower leg o...
2. `medquad_ans_0000256-1__dup3` (score=0.6995) - Deep vein thrombosis, or DVT, is a blood clot that forms in a vein deep in the body. Most deep vein clots occur in the lower leg or thigh. If the vein swells, the condition is called thrombophlebitis. A deep vein thro...
3. `medquad_ans_0000051-5__dup1` (score=0.6680) - The signs and symptoms of deep vein thrombosis (DVT) might be related to DVT itself or pulmonary embolism (PE). See your doctor right away if you have signs or symptoms of either condition. Both DVT and PE can cause s...
4. `medquad_ans_0000104-1__dup4` (score=0.6340) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent b...
5. `medquad_ans_0000051-4__dup2` (score=0.6335) - The risk factors for deep vein thrombosis (DVT) include: A history of DVT. Conditions or factors that make your blood thicker or more likely to clot than normal. Some inherited blood disorders (such as factor V Leiden...

**Answer**:

Deep vein thrombosis (DVT) is caused by blood thickening and clumping together, which leads to the formation of a blood clot in a deep vein in the body. Most DVTs occur in the lower leg or thigh, and risk factors include sitting still for a long time, certain medicines, and disorders that increase the likelihood of blood clots [medquad_ans_0000051-1\_\_dup2][medquad_ans_0000256-1__dup3].

**Supported citations**: `medquad_ans_0000051-1__dup2`, `medquad_ans_0000256-1__dup3`
**Unsupported citations**: _none_

### Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.130, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004079-1` (score=0.7000) - Complex II deficiency is a mitochondrial disease. Mitochondria are specialized compartments in cells that create more than 90% of the energy needed by the body. In mitochondrial diseases, the mitochondria don't work c...
2. `medquad_ans_0000661-1` (score=0.6793) - Mitochondrial complex III deficiency is a genetic condition that can affect several parts of the body, including the brain, kidneys, liver, heart, and the muscles used for movement (skeletal muscles). Signs and sympto...
3. `medquad_ans_0004079-4` (score=0.6227) - How might mitochondrial complex II deficiency be treated? Treatment options for complex II deficiency may be similar to those for other mitochondrial disorders in general.[8677] The United Mitochondrial Disease Founda...
4. `medquad_ans_0000661-3` (score=0.6117) - Mitochondrial complex III deficiency can be caused by mutations in one of several genes. The proteins produced from these genes either are a part of or help assemble a group of proteins called complex III. The two mos...
5. `medquad_ans_0004081-2` (score=0.5881) - What are the signs and symptoms of Mitochondrial complex IV deficiency? There are currently 4 known forms of COX deficiency. The range and severity of signs and symptoms can vary widely from case to case. In one form,...

**Answer**:

The provided context does not contain any information about mitochondrial complex IV deficiency. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Scores**: context_relevance=1.000, groundedness=0.477, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000078-1` (score=0.7000) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...
2. `medquad_ans_0000078-2` (score=0.6898) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.
3. `medquad_ans_0000078-5` (score=0.6793) - These resources address the diagnosis or management of ataxia with oculomotor apraxia: - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type 1 - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type ...
4. `medquad_ans_0000496-1__dup1` (score=0.6494) - What are the signs and symptoms of Ataxia with Oculomotor Apraxia Type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Ataxia with Oculomotor Apraxia Type 2. If the information is...
5. `medquad_ans_0000495-1__dup1` (score=0.5678) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...

**Answer**:

Ataxia with oculomotor apraxia is a rare condition characterized by progressive problems with movement, specifically difficulty coordinating movements (ataxia), which is often the first symptom. The hallmark feature is oculomotor apraxia, which makes it difficult to move the eyes side-to-side, requiring individuals to turn their head to see peripheral vision. There are multiple types, with type 1 and type 2 being the most common. Type 1 typically presents with ataxia beginning around age 4, while type 2 begins around age 15. Both types share features such as involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, chorea and myoclonus tend to disappear gradually, whereas in type 2, these movement problems persist throughout life. Type 1 individuals often develop hand and foot wasting (atro

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Scores**: context_relevance=1.000, groundedness=0.908, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000221-1` (score=0.7000) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that fo...
2. `medquad_ans_0000221-5` (score=0.6903) - These resources address the diagnosis or management of LAMM syndrome: - Gene Review: Gene Review: Congenital Deafness with Labyrinthine Aplasia, Microtia, and Microdontia - Genetic Testing Registry: Deafness with laby...
3. `medquad_ans_0001744-1` (score=0.6802) - What are the signs and symptoms of Deafness with labyrinthine aplasia microtia and microdontia (LAMM)? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness with labyrinthine apla...
4. `medquad_ans_0001496-1` (score=0.6426) - Congenital deafness with vitiligo and achalasia is a syndrome characterized by deafness present from birth (congenital), associated with short stature, vitiligo, muscle wasting and achalasia (swallowing difficulties)....
5. `medquad_ans_0001496-2` (score=0.4444) - What are the signs and symptoms of Congenital deafness with vitiligo and achalasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital deafness with vitiligo and achalasia. I...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (LAMM syndrome) is a condition that affects the development of the ears and teeth. It is characterized by the complete absence of inner ear structures (labyrinthine aplasia), leading to profound sensorineural hearing loss present from birth. Affected individuals often have abnormally small outer ears (microtia) with narrow ear canals and unusually small, widely spaced teeth (microdontia). The condition also causes delayed motor skill development, such as sitting and crawling, due to balance issues from inner ear abnormalities. Additional signs and symptoms include abnormalities of cranial nerves (seen in 90% of patients), microdontia (90%), abnormal nasal morphology, and other features like pointed chin, wide nasal bridge, and skin tags [medquad_ans_0000221-1]medquad_ans_0001

**Supported citations**: `medquad_ans_0000221-1`
**Unsupported citations**: _none_

### Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000064-4` (score=0.7000) - Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases of Apert syndrome result from new mutations in ...
2. `medquad_ans_0000420-3__dup1` (score=0.6448) - How is Apert syndrome diagnosed? Is genetic testing needed to confirm the diagnosis? Apert syndrome and the other conditions associated with FGFR-related craniosynostosis were clinically defined long before the molecu...
3. `medquad_ans_0000064-1` (score=0.6439) - Apert syndrome is a genetic disorder characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the head and fa...
4. `medquad_ans_0000420-1__dup1` (score=0.6439) - Apert syndrome is a disorder mainly characterized by craniosynostosis (premature fusion of skull bones, causing abnormalities in the shape of the head and face) and syndactyly (fusion or webbing or fingers and/or toes...
5. `medquad_ans_0000064-2` (score=0.6365) - Apert syndrome affects an estimated 1 in 65,000 to 88,000 newborns.

**Answer**:

Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. However, almost all cases result from new mutations and occur in people with no family history of the disorder. Individuals with Apert syndrome can pass the condition to their offspring [medquad_ans_0000064-4].

**Supported citations**: `medquad_ans_0000064-4`
**Unsupported citations**: _none_

### Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Scores**: context_relevance=0.750, groundedness=0.544, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.7000) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organi...
2. `medquad_ans_0000341-1__dup3` (score=0.6997) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella bu...
3. `medquad_ans_0000341-3__dup2` (score=0.6903) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact ...
4. `medquad_ans_0000341-4__dup2` (score=0.6701) - In the United States, Q fever outbreaks have resulted mainly from occupational exposure involving veterinarians, meat processing plant workers, sheep and dairy workers, livestock farmers, and researchers at facilities...
5. `medquad_ans_0000359-1__dup1` (score=0.5459) - A fever is a body temperature that is higher than normal. It is not an illness. It is part of your body's defense against infection. Most bacteria and viruses that cause infections do well at the body's normal tempera...

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air contaminated by dried placental material, birth fluids, and excreta of infected animals. Other rare modes of transmission include tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, individuals with acute Q fever develop a chronic form of the disease. Treatment of the acute form involves antibiotics, while chronic Q fever treatment depends on the symptoms [

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Scores**: context_relevance=1.000, groundedness=0.947, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000038-5__dup1` (score=0.7000) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
2. `medquad_ans_0000038-1__dup2` (score=0.6902) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chroni...
3. `medquad_ans_0000038-9` (score=0.6800) - At this time, most types of childhood interstitial lung disease (chILD) can't be prevented. People who have a family history of inherited (genetic) interstitial lung disease may want to consider genetic counseling. A ...
4. `medquad_ans_0001152-2` (score=0.6243) - How might chILD be treated? There is no single treatment for interstitial lung diseases in children. Different forms of chILD require different treatments and support depending on the condition. The goals of treatment...
5. `medquad_ans_0000522-1__dup1` (score=0.6067) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...

**Answer**:

Children who have a family history of interstitial lung disease or childhood interstitial lung disease (chILD) are at risk. Other risk factors include having an inherited surfactant disorder or a family history of such disorders, problems with aspiration (inhaling food, liquid, or vomit into the lungs), immune system disorders, exposure to environmental lung irritants such as molds and chemicals, systemic or autoimmune diseases (e.g., collagen vascular disease or inflammatory bowel disease), undergoing cancer treatments like radiation and chemotherapy, or having a bone marrow transplant or lung transplant [medquad_ans_0000038-5__dup1].

**Supported citations**: `medquad_ans_0000038-5__dup1`
**Unsupported citations**: _none_

### Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Scores**: context_relevance=0.833, groundedness=0.949, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.7000) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in lat...
2. `medquad_ans_0000087-1__dup1` (score=0.6700) - Vasculitis is an inflammation of blood vessels, which includes the veins, arteries, and capillaries. Inflammation occurs with infection or is thought to be due to a faulty immune system response. It also can be caused...
3. `medquad_ans_0000404-1__dup1` (score=0.6239) - Giant cell arteritis is a disorder that causes inflammation of your arteries, usually in the scalp, neck, and arms. It narrows the arteries, which keeps blood from flowing well. Giant cell arteritis often occurs with ...
4. `medquad_ans_0006001-1` (score=0.6139) - Takayasu arteritis is a condition that causes inflammation of the main blood vessel that carries blood from the heart to the rest of the body (aorta) and its associated branched blood vessels. As a result of the infla...
5. `medquad_ans_0002107-1` (score=0.5157) - Enthesitis-related juvenile idiopathic arthritis is a subtype of juvenile idiopathic arthritis that is characterized by both arthritis and inflammation of an enthesitis site (the point at which a ligament, tendon, or ...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels, specifically affecting the temporal arteries located at the lower sides of the skull, directly underneath the temple. It is generally diagnosed in late childhood or early adulthood and often presents with no signs or symptoms aside from a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically without a family history. It is usually treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

### Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.518, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001082-2` (score=0.7000) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyry...
2. `medquad_ans_0001082-1` (score=0.6994) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as...
3. `medquad_ans_0006577-1` (score=0.6659) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particul...
4. `medquad_ans_0001082-5` (score=0.6295) - These resources address the diagnosis or management of 2-methylbutyryl-CoA dehydrogenase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of 2-methylbutyryl-CoA dehydrogenase These resources from...
5. `medquad_ans_0006577-2` (score=0.5470) - What are the signs and symptoms of 2-methylbutyryl-CoA dehydrogenase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for 2-methylbutyryl-CoA dehydrogenase deficiency. If the ...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a rare metabolic disorder in which individuals lack adequate levels of the enzyme 2-methylbutyryl-CoA dehydrogenase, which is necessary for the proper processing of the amino acid isoleucine. This deficiency leads to the abnormal buildup of organic acids in the blood, urine, and tissues, which can be toxic and cause a range of health problems. Symptoms vary widely, from severe and life-threatening to mild or absent, and may include poor feeding, lack of energy, vomiting, difficulty breathing, seizures, coma, poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills. Symptoms can be triggered by fasting, infections, or increased protein intake. The disorder is more common among individuals of Hmong ancestry, with an incidence of 1 in 250 to 1 in 500 people of Hmong ancestry [

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005841-2` (score=0.7000) - What are the signs and symptoms of Spondyloepiphyseal dysplasia Maroteaux type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia Maroteaux type. If the in...
2. `medquad_ans_0005840-2` (score=0.6489) - What are the signs and symptoms of Spondyloepiphyseal dysplasia congenita? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia congenita. If the information ...
3. `medquad_ans_0000947-1` (score=0.6489) - Spondyloepiphyseal dysplasia congenita is an inherited bone growth disorder that results in short stature (dwarfism), skeletal abnormalities, and problems with vision and hearing. This condition affects the bones of t...
4. `medquad_ans_0005840-1` (score=0.6256) - Spondyloepiphyseal dysplasia congenita is an inherited disorder of bone growth that affects the bones of the spine and ends of the long bones in the arms and legs. Features of this condition include short stature (dwa...
5. `medquad_ans_0005839-1` (score=0.6021) - What are the signs and symptoms of Spondyloepiphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia. If the information is available, the ta...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include abnormal form of the vertebral bodies (90%), abnormality of epiphysis morphology (90%), micromelia (90%), narrow chest (90%), short neck (90%), short stature (90%), short thorax (90%), skeletal dysplasia (90%), broad forehead (50%), cleft palate (50%), hyperlordosis (50%), hypertelorism (50%), malar flattening (50%), osteoarthritis (50%), talipes (50%), cataract (7.5%), glaucoma (7.5%), hearing impairment (7.5%), kyphosis (7.5%), myopia (7.5%), nystagmus (7.5%), retinal detachment (7.5%), and scoliosis (7.5%)

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Scores**: context_relevance=1.000, groundedness=0.908, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000853-2` (score=0.7000) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
2. `medquad_ans_0005219-1` (score=0.6916) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepti...
3. `medquad_ans_0005219-3` (score=0.6904) - What causes pyridoxine-dependent epilepsy? Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each ...
4. `medquad_ans_0000853-1` (score=0.6801) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepti...
5. `medquad_ans_0000853-5` (score=0.6738) - These resources address the diagnosis or management of pyridoxine-dependent epilepsy: - Gene Review: Gene Review: Pyridoxine-Dependent Epilepsy - Genetic Testing Registry: Pyridoxine-dependent epilepsy - MedlinePlus E...

**Answer**:

Pyridoxine-dependent epilepsy is caused by mutations in the ALDH7A1 gene, which lead to a deficiency of the enzyme -aminoadipic semialdehyde (-AASA) dehydrogenase (also known as antiquitin). This enzyme is involved in the breakdown of the amino acid lysine in the brain. When antiquitin is deficient, a molecule that interferes with vitamin B6 function builds up in various tissues. This disruption in vitamin B6 function is believed to contribute to the seizures characteristic of the condition. The condition is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations. However, in some individuals, no mutations in the ALDH7A1 gene are identified, and the cause of the condition remains unknown [medquad_ans_0005219-3].

**Supported citations**: `medquad_ans_0005219-3`
**Unsupported citations**: _none_

### Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Scores**: context_relevance=1.000, groundedness=0.941, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000004_6-2` (score=0.7000) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
2. `medquad_ans_0000004_1-2` (score=0.6698) - Age, gender, and Epstein-Barr infection can affect the risk of adult Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will g...
3. `medquad_ans_0000004_6-1` (score=0.6616) - Key Points - Childhood Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two types of childhood Hodgkin lymphoma. - Epstein-Barr virus infection increases the risk o...
4. `medquad_ans_0000004_1-1` (score=0.5649) - Key Points - Adult Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two main types of Hodgkin lymphoma: classical and nodular lymphocyte-predominant. - Age, gender,...
5. `medquad_ans_0000004_6-5` (score=0.5531) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whet...

**Answer**:

Individuals at risk for childhood Hodgkin lymphoma include those who are infected with the Epstein-Barr virus, infected with the human immunodeficiency virus (HIV), have certain diseases of the immune system, have a personal history of mononucleosis ("mono"), or have a parent or sibling with a personal history of Hodgkin lymphoma. Being exposed to common infections in early childhood may decrease the risk of Hodgkin lymphoma in children due to its effect on the immune system [medquad_ans_0000004_6-2].

**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

### Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Scores**: context_relevance=1.000, groundedness=0.992, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001795-2` (score=0.7000) - What are the signs and symptoms of Dermatitis herpetiformis ? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatitis herpetiformis . If the information is available, the table be...
2. `medquad_ans_0001795-1` (score=0.6813) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow...
3. `medquad_ans_0001795-3` (score=0.6619) - How might dermatitis herpetiformis be treated? The antibiotic dapsone is extremely effective in treating this condition. Symptomatic improvement may occur in as little as several hours after the first dose. However, d...
4. `medquad_ans_0000095-2__dup3` (score=0.6443) - Dermatitis herpetiformis is caused by the deposit of immunoglobulin A (IgA) in the skin, which triggers further immunologic reactions resulting in lesion formation. DH is an external manifestation of an abnormal immun...
5. `medquad_ans_0000095-1__dup4` (score=0.6348) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation ...

**Answer**:

Dermatitis herpetiformis is a rare, chronic skin disorder characterized by groups of severely itchy blisters and raised skin lesions, which are more common on the knees, elbows, buttocks, and shoulder blades. The slow onset of symptoms usually begins during adulthood, but children can also be affected. Other symptoms may include fluid-filled sores, red lesions that resemble hives, and itchiness, redness, and burning. The exact cause is not known, but it is frequently associated with the inability to digest gluten. People with this condition are typically treated with the drug dapsone [medquad_ans_0001795-1].

**Supported citations**: `medquad_ans_0001795-1`
**Unsupported citations**: _none_

### Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Scores**: context_relevance=0.857, groundedness=0.969, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000709-5` (score=0.7000) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Regi...
2. `medquad_ans_0000040-4__dup1` (score=0.6801) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to myotonia congenita and also supports additional research through grants to major research institutions across the countr...
3. `medquad_ans_0000040-2__dup1` (score=0.6432) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors ...
4. `medquad_ans_0004287-1` (score=0.5974) - Myotonia congenita is a genetic condition characterized by the inability of the skeletal muscles to quickly relax after a voluntary movement. The symptoms associated with the condition typically appear in childhood an...
5. `medquad_ans_0000040-1__dup1` (score=0.5812) - Myotonia congenita is an inherited neuromuscular disorder characterized by the inability of muscles to quickly relax after a voluntary contraction. The condition is present from early childhood, but symptoms can be mi...

**Answer**:

Most people with myotonia congenita do not require special treatments; stiff muscles usually resolve with exercise or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors have had some success with medications such as quinine or anticonvulsant drugs such as phenytoin. Physical therapy and other rehabilitative therapies are also sometimes used to improve muscle function [medquad_ans_0000040-2__dup1].

**Supported citations**: `medquad_ans_0000040-2__dup1`
**Unsupported citations**: _none_

### Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Scores**: context_relevance=0.833, groundedness=0.914, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0006449-5` (score=0.7000) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lif...
2. `medquad_ans_0006449-1` (score=0.6523) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver,...
3. `medquad_ans_0000133-13` (score=0.6488) - - Wilson disease is a genetic disease that prevents the body from removing extra copper. - Normally, the liver filters extra copper and releases it into bile. In Wilson disease, the liver does not filter copper correc...
4. `medquad_ans_0000133-12` (score=0.6274) - A person cannot prevent Wilson disease; however, people with a family history of Wilson disease, especially those with an affected sibling or parent, should talk with a health care provider about testing. A health car...
5. `medquad_ans_0000973-1__dup1` (score=0.6174) - Wilson disease is a rare inherited disorder that prevents your body from getting rid of extra copper. You need a small amount of copper from food to stay healthy. Too much copper is poisonous. Normally, your liver rel...

**Answer**:

For Wilson disease, treatment involves a lifelong effort to reduce and control the amount of copper that accumulates in the body. This may include medications, dietary modifications such as reducing intake of copper-rich foods (e.g., shellfish, liver, mushrooms, nuts, chocolate), and in some cases, a liver transplant if treatment is not effective or if liver failure develops [medquad_ans_0006449-5]. The condition is caused by a mutation in the ATP7B gene and is inherited in an autosomal recessive manner [medquad_ans_0006449-1]. People with a family history of Wilson disease should consult a healthcare provider about testing [medquad_ans_0000133-13].

**Supported citations**: `medquad_ans_0006449-5`, `medquad_ans_0006449-1`, `medquad_ans_0000133-13`
**Unsupported citations**: _none_

### Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.969, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000592-3` (score=0.7000) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are re...
2. `medquad_ans_0000592-4` (score=0.6614) - Most cases of Lennox-Gastaut syndrome are sporadic, which means they occur in people with no history of the disorder in their family. However, 3 to 30 percent of people with this condition have a family history of som...
3. `medquad_ans_0000592-1` (score=0.6277) - Lennox-Gastaut syndrome is a form of severe epilepsy that begins in childhood. It is characterized by multiple types of seizures and intellectual disability. People with Lennox-Gastaut syndrome begin having frequent s...
4. `medquad_ans_0000592-5` (score=0.5392) - These resources address the diagnosis or management of Lennox-Gastaut syndrome: - Cleveland Clinic - Genetic Testing Registry: Epileptic encephalopathy Lennox-Gastaut type - National Institute of Neurological Disorder...
5. `medquad_ans_0000592-2` (score=0.0830) - Lennox-Gastaut syndrome affects an estimated 1 in 50,000 to 1 in 100,000 children. This condition accounts for about 4 percent of all cases of childhood epilepsy. For unknown reasons, it appears to be more common in m...

**Answer**:

Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component [medquad_ans_0000592-3].

**Supported citations**: `medquad_ans_0000592-3`
**Unsupported citations**: _none_

### Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.890, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000066-2` (score=0.7000) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
2. `medquad_ans_0000066-5` (score=0.6994) - These resources address the diagnosis or management of arginine:glycine amidinotransferase deficiency: - Gene Review: Gene Review: Creatine Deficiency Syndromes - Genetic Testing Registry: Arginine:glycine amidinotran...
3. `medquad_ans_0003472-1` (score=0.6706) - L-arginine:glycine amidinotransferase (AGAT) deficiency is a rare condition that primarily affects the brain. People with AGAT deficiency generally have mild to moderate intellectual disability. Other signs and sympto...
4. `medquad_ans_0003472-2` (score=0.6520) - What are the signs and symptoms of L-arginine:glycine amidinotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for L-arginine:glycine amidinotransferase deficiency....
5. `medquad_ans_0000416-3` (score=0.2171) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...

**Answer**:

The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families [medquad_ans_0000066-2]. Therefore, the exact number of people affected is not known.

**Supported citations**: `medquad_ans_0000066-2`
**Unsupported citations**: _none_

### Example 231 (`0002399-2`)

**Question**: What are the symptoms of Focal dermal hypoplasia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000379-1` (score=0.7000) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. About 90 percent of affected individuals are female. Males usually have milder signs and symptoms than females. ...
2. `medquad_ans_0002399-1` (score=0.6994) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. The skin abnormalities are present from birth and can include streaks of very thin skin (dermal hypoplasia), cut...
3. `medquad_ans_0002399-2` (score=0.6727) - What are the signs and symptoms of Focal dermal hypoplasia? Focal dermal hypoplasia is usually evident from birth and primarily affects the skin, skeleton, eyes, and face. The signs and symptoms of vary widely, althou...
4. `medquad_ans_0000379-2` (score=0.6355) - Focal dermal hypoplasia appears to be a rare condition, although its exact prevalence is unknown.
5. `medquad_ans_0002401-1` (score=0.4889) - What are the signs and symptoms of Focal facial dermal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Focal facial dermal dysplasia. If the information is available, the ...

**Answer**:

Symptoms of focal dermal hypoplasia include skin abnormalities present from birth such as streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), small clusters of veins on the surface of the skin (telangiectases), and streaks of slightly darker or lighter skin. These skin changes may cause pain, itching, irritation, or lead to skin infections. Wart-like growths called papillomas develop with age and may form around the nostrils, lips, anus, and female genitalia, and can also be present in the throat (esophagus or larynx), causing problems with swallowing, breathing, or sleeping. Affected individuals may have small, ridged fingernails and toenails, sparse or brittle scalp hair or absent hair. Hand and foot abnormalities include missing fingers or toes

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=0.7000) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
2. `medquad_ans_0000418-5__dup1` (score=0.5836) - A diagnosis of trichinellosis is made in patients whose signs and symptoms are compatible with trichinellosis, have a positive laboratory test for Trichinella, and who can recall eating raw or undercooked pork or wild...
3. `medquad_ans_0000038-2__dup3` (score=0.5241) - People can get infected with Babesia parasites in several ways: - The main way is through the bite of an infected tick—during outdoor activities in areas where babesiosis is found (see below). - A less common way is b...
4. `medquad_ans_0000269-2__dup2` (score=0.3621) - There are three different filarial species that can cause lymphatic filariasis in humans. Most of the infections worldwide are caused by Wuchereria bancrofti. In Asia, the disease can also be caused by Brugia malayi a...
5. `medquad_ans_0000418-1__dup2` (score=0.0697) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...

**Answer**:

People at risk for Trichinellosis (also known as Trichinosis) include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating even small amounts of undercooked meat during preparation or cooking puts individuals at risk. Additionally, individuals who consume undercooked wild game, especially bear meat, are at risk. The risk from commercially raised and properly prepared pork is very low, but it remains a concern when meat is not cooked adequately [medquad_ans_0000418-2__dup1]. Outbreaks can occur when multiple people consume the same infected meat [medquad_ans_0000418-2__dup1].

**Supported citations**: `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

### Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.732, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001924-2` (score=0.7000) - What are the signs and symptoms of Dravet syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Dravet syndrome. If the information is available, the table below includes how oft...
2. `medquad_ans_0000108-3__dup1` (score=0.0913) - As children with Dravet syndrome get older, their decline in cognitive function stabilizes, and in many, it improves slightly. However, most teenagers with Dravet syndrome are dependent on caregivers. The degree of co...
3. `medquad_ans_0000108-1__dup2` (score=0.0809) - Dravet syndrome, also called severe myoclonic epilepsy of infancy (SMEI), is a severe form of epilepsy. It appears during the first year of life with frequent febrile seizures fever-related seizures that, by definitio...
4. `medquad_ans_0005016-2` (score=0.0809) - What are the signs and symptoms of Prader-Willi syndrome? In infancy, Prader-Willi syndrome (PWS) is characterized by weak muscle tone (hypotonia), feeding difficulties, poor growth, and delayed development. In later ...
5. `medquad_ans_0002063-2` (score=0.0708) - What are the signs and symptoms of Ehlers-Danlos syndrome, vascular type? The signs and symptoms of Ehlers-Danlos syndrome (EDS), vascular type vary but may include: Fragile tissues (including arteries, muscles and in...

**Answer**:

The symptoms of Dravet syndrome include absence seizures, ataxia, cerebral atrophy, cortical visual impairment, epileptic encephalopathy, focal seizures with impairment of consciousness or awareness, generalized myoclonic seizures, hemiclonic seizures, infantile onset, mental deterioration, motor delay, postnatal microcephaly, and status epilepticus [medquad_ans_0001924-2]. Additionally, children with Dravet syndrome experience poor development of language and motor skills, hyperactivity, and difficulty relating to others [medquad_ans_0000108-1__dup2]. The degree of cognitive impairment correlates with the frequency of seizures [medquad_ans_0000108-3__dup2].

**Supported citations**: `medquad_ans_0001924-2`, `medquad_ans_0000108-1__dup2`
**Unsupported citations**: `medquad_ans_0000108-3__dup2`

### Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Scores**: context_relevance=0.889, groundedness=0.898, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000011-13` (score=0.7000) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...
2. `medquad_ans_0002533-3` (score=0.6840) - What causes Gerstmann-Straussler-Scheinker disease? Gerstmann-Straussler-Scheinker disease (GSS) is usually caused by certain changes (mutations) in the PRNP gene. PRNP encodes a protein called prion protein. Although...
3. `medquad_ans_0000089-4__dup1` (score=0.0928) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system, and to use that knowledge to reduce the burden of neurological di...
4. `medquad_ans_0001663-3` (score=0.0822) - What causes Creutzfeldt-Jakob disease? Some researchers believe an unusual 'slow virus' or another organism causes Creutzfeldt-Jakob disease (CJD). However, they have never been able to isolate a virus or other organi...
5. `medquad_ans_0004784-4` (score=0.0822) - How is Pelizaeus-Merzbacher disease inherited?

**Answer**:

Genetic changes related to prion disease include mutations in the PRNP gene, which encode the prion protein. These mutations result in the production of an abnormally shaped prion protein, which can build up in the brain and damage or destroy neurons [medquad_ans_0002533-3]. In inherited forms of CJD, an abnormal gene leads to changes in the prion protein, making it likely to assume an abnormal shape [medquad_ans_0000011-13]. Thus, mutations in the PRNP gene are a key genetic change associated with prion diseases such as Gerstmann-Straussler-Scheinker disease and inherited Creutzfeldt-Jakob disease.

**Supported citations**: `medquad_ans_0002533-3`, `medquad_ans_0000011-13`
**Unsupported citations**: _none_

### Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0006430-1` (score=0.7000) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information ...
2. `medquad_ans_0000014-1__dup1` (score=0.2147) - What are the signs and symptoms of Abruzzo Erickson syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Abruzzo Erickson syndrome. If the information is available, the table be...
3. `medquad_ans_0001034-1` (score=0.0697) - Weyers acrofacial dysostosis is a disorder that affects the development of the teeth, nails, and bones. Dental abnormalities can include small, peg-shaped teeth; fewer teeth than normal (hypodontia); and one front too...
4. `medquad_ans_0003459-2` (score=0.0697) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficu...
5. `medquad_ans_0003474-2` (score=0.0601) - What are the signs and symptoms of Larsen syndrome? The signs and symptoms of Larsen syndrome vary from person to person, but may include the following: Joint dislocation (especially of the hips, knees, and elbows) Hy...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, autosomal dominant inheritance, cleft palate, cleft upper lip, high palate, hydronephrosis, hypoplasia of the radius, hypotelorism, long face, mesomelia, narrow face, oligodactyly (hands), proximal placement of thumb, and proximal radial head dislocation [medquad_ans_0006430-1].

**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

### Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Scores**: context_relevance=1.000, groundedness=0.901, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.7000) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the ta...
2. `medquad_ans_0004524-1` (score=0.6609) - What are the signs and symptoms of Okamoto syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Okamoto syndrome. If the information is available, the table below includes how o...
3. `medquad_ans_0005030-1` (score=0.5414) - What are the signs and symptoms of Premature ovarian failure, familial? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature ovarian failure, familial. If the information is ava...
4. `medquad_ans_0002061-2` (score=0.4900) - What are the signs and symptoms of Ehlers-Danlos syndrome, progeroid type? Ehlers-Danlos syndrome refers to a group of connective tissue disorders characterized by stretchy or kneadable skin, double jointedness, and d...
5. `medquad_ans_0004961-1` (score=0.4199) - What are the signs and symptoms of Polyneuropathy mental retardation acromicria premature menopause? The Human Phenotype Ontology provides the following list of signs and symptoms for Polyneuropathy mental retardation...

**Answer**:

The symptoms of Premature aging Okamoto type include abnormality of the blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face [medquad_ans_0005029-1]. The frequency of these symptoms is based on data from the Human Phenotype Ontology, which compiles information from sources such as Orphanet, with frequencies often expressed as percentages or fractions representing the proportion of patients exhibiting each symptom [medquad_ans_0005029-1].

**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

### Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.747, answer_relevance=0.489

**Top docs**:

1. `medquad_ans_0000979-1` (score=0.7000) - Thrombocytopenia-absent radius (TAR) syndrome is characterized by the absence of a bone called the radius in each forearm. Affected individuals also have a shortage (deficiency) of blood cells involved in clotting (pl...
2. `medquad_ans_0004919-1` (score=0.0902) - Platelet storage pool deficiency refers to a group of conditions that are caused by problems with the platelet granules. Platelet granules are tiny storage sacs found within the platelets which release various substan...
3. `medquad_ans_0000329-2` (score=0.0902) - Essential thrombocythemia affects an estimated 1 to 24 per 1 million people worldwide.
4. `medquad_ans_0001065-2` (score=0.0799) - The estimated incidence of X-linked thrombocytopenia is between 1 and 10 per million males worldwide; this condition is rarer among females.
5. `medquad_ans_0001079-3` (score=0.0699) - Most people with a 1q21.1 microdeletion are missing a sequence of about 1.35 million DNA building blocks (base pairs), also written as 1.35 megabases (Mb), in the q21.1 region of chromosome 1. However, the exact size ...

**Answer**:

The provided context does not include the number of people affected by thrombocytopenia-absent radius (TAR) syndrome. Therefore, it is not possible to determine the prevalence or incidence from the given information. [medquad_ans_0000979-1] describes the clinical features of TAR syndrome but does not provide population statistics.

Answer: Insufficient information.

**Supported citations**: `medquad_ans_0000979-1`
**Unsupported citations**: _none_

### Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Scores**: context_relevance=0.909, groundedness=0.862, answer_relevance=0.836

**Top docs**:

1. `medquad_ans_0000007_2-6` (score=0.7000) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000972-5` (score=0.7000) - These resources address the diagnosis or management of Tay-Sachs disease: - Gene Review: Gene Review: Hexosaminidase A Deficiency - Genetic Testing Registry: Tay-Sachs disease - MedlinePlus Encyclopedia: Tay-Sachs Dis...
3. `medquad_ans_0006017-5` (score=0.6122) - How might children with Tay-Sachs disease be treated? Although several attempts have been made at purified enzyme replacement therapy for children with Tay-Sachs disease, none has been successful. Cellular infusions a...
4. `medquad_ans_0000009_2-8` (score=0.6122) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000032_1-8` (score=0.5272) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any specific research or clinical trials being conducted for Tay-Sachs disease. While it lists resources related to the diagnosis and management of Tay-Sachs disease, such as gene review, genetic testing, and drug therapy, there is no information about ongoing clinical trials or research efforts for this condition [medquad_ans_0000972-5]. Therefore, based on the given context, the research or clinical trials for Tay-Sachs disease cannot be identified.

**Supported citations**: `medquad_ans_0000972-5`
**Unsupported citations**: _none_

### Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000008-1` (score=0.7000) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achro...
2. `medquad_ans_0000008-2` (score=0.6997) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on o...
3. `medquad_ans_0000008-5` (score=0.6816) - These resources address the diagnosis or management of achromatopsia: - Gene Review: Gene Review: Achromatopsia - Genetic Testing Registry: Achromatopsia - MedlinePlus Encyclopedia: Color Vision Test These resources f...
4. `medquad_ans_0000008-3` (score=0.6620) - Achromatopsia results from changes in one of several genes: CNGA3, CNGB3, GNAT2, PDE6C, or PDE6H. A particular CNGB3 gene mutation underlies the condition in Pingelapese islanders. Achromatopsia is a disorder of the r...
5. `medquad_ans_0000050-1__dup1` (score=0.6353) - What are the signs and symptoms of Achromatopsia 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Achromatopsia 3. If the information is available, the table below includes how oft...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. The condition also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals may have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia results from mutations in genes such as CNGA3, CNGB3, GNAT2, PDE6C, or PDE6H, which affect the function of cone cells in the retina,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Scores**: context_relevance=1.000, groundedness=0.859, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000544-1__dup1` (score=0.7000) - Knee replacement is surgery for people with severe knee damage. Knee replacement can relieve pain and allow you to be more active. Your doctor may recommend it if you have knee pain and medicine and other treatments a...
2. `medquad_ans_0000039-11` (score=0.7000) - Treatments your doctor will likely recommend before knee replacement include - exercises to strengthen the muscles around the knee and improve flexibility - weight loss, if needed, to reduce the load the knee must bea...
3. `medquad_ans_0000039-2__dup2` (score=0.6538) - There are many different types and designs of artificial knees. Most consist of three components: - the femoral component, which is the part that attaches to the thigh bone - the tibial component, the part that attach...
4. `medquad_ans_0000039-26` (score=0.6218) - You can learn more about knee replacement from the following resources. National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) Information Clearinghouse National Institutes of Health 1 AMS Circl...
5. `medquad_ans_0000039-13` (score=0.6103) - Knee replacement may be either total or partial/unicompartmental. In total knee replacement, as the name suggests, the entire knee joint is replaced. You will likely need a total knee replacement if you have damage to...

**Answer**:

The context provided does not list treatments for knee replacement itself, but rather describes treatments recommended _before_ knee replacement. These include exercises to strengthen the muscles around the knee and improve flexibility, weight loss if needed, walking aids such as canes to reduce stress on the joint, shoe inserts to improve knee alignment, and medicines to relieve pain [medquad_ans_0000039-11].

Knee replacement treatments involve surgical procedures such as total or partial knee replacement, where damaged cartilage and bone are removed and replaced with artificial components (metal and plastic) [medquad_ans_0000544-1__dup1]. The artificial knee typically consists of three components: the femoral component (attached to the thigh bone), the tibial component (attached to the shin bone), and the patellar component (attached to the knee cap) [medquad_ans_0000039-2

**Supported citations**: `medquad_ans_0000039-11`, `medquad_ans_0000544-1__dup1`
**Unsupported citations**: _none_

### Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.868, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000608-1__dup1` (score=0.7000) - Metabolic syndrome is a group of conditions that put you at risk for heart disease and diabetes. These conditions are - High blood pressure - High blood glucose, or blood sugar, levels - High levels of triglycerides, ...
2. `medquad_ans_0000090-1__dup2` (score=0.6822) - Metabolicsyndrome is the name for a group of risk factors that raises your risk for heart disease and other health problems, such as diabetes and stroke. The term "metabolic" refers to the biochemical processes involv...
3. `medquad_ans_0000044-11` (score=0.5924) - Metabolic syndrome, also called insulin resistance syndrome, is a group of traits and medical conditions linked to overweight and obesity that puts people at risk for both CVD and type 2 diabetes. Metabolic syndrome i...
4. `medquad_ans_0000090-4__dup2` (score=0.5806) - People at greatest risk for metabolic syndrome have these underlying causes: Abdominal obesity (a large waistline) An inactive lifestyle Insulin resistance Some people are at risk for metabolic syndrome because they t...
5. `medquad_ans_0004775-2` (score=0.2001) - What are the signs and symptoms of Pearson syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pearson syndrome. If the information is available, the table below includes how o...

**Answer**:

The provided context does not mention any symptoms of metabolic syndrome. It describes the risk factors and conditions associated with metabolic syndrome, such as high blood pressure, high blood glucose, high triglycerides, low HDL cholesterol, and excess fat around the waist, but does not list symptoms [medquad_ans_0000608-1\_\_dup1][medquad_ans_0000090-1__dup2]. Therefore, based on the given information, no symptoms can be identified.

**Supported citations**: `medquad_ans_0000608-1__dup1`, `medquad_ans_0000090-1__dup2`
**Unsupported citations**: _none_

### Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Scores**: context_relevance=1.000, groundedness=0.263, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000108-2__dup1` (score=0.7000) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
2. `medquad_ans_0005211-1` (score=0.7000) - What are the signs and symptoms of Pyknoachondrogenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyknoachondrogenesis. If the information is available, the table below includ...
3. `medquad_ans_0002152-2` (score=0.6122) - What are the signs and symptoms of Erdheim-Chester disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Erdheim-Chester disease. If the information is available, the table below ...
4. `medquad_ans_0000549-1__dup1` (score=0.6122) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...
5. `medquad_ans_0004625-2` (score=0.5272) - What are the signs and symptoms of Osteomesopyknosis? Osteomesopyknosis may cause chronic, low-grade back pain in the thoracic (middle) and lumbar (lower) regions. It is considered a mild form of osteosclerosis and is...

**Answer**:

The symptoms of Pyknoachondrogenesis include abnormal vertebral ossification, abnormality of pelvic girdle bone morphology, abnormality of the mouth, abnormality of the sacrum, depressed nasal ridge, enlarged thorax, increased bone mineral density, low-set, posteriorly rotated ears, macrocephaly, micromelia, palpebral edema, premature birth, short stature, short thorax, and thickened nuchal skin fold. These symptoms are reported in approximately 90% of patients with this condition [medquad_ans_005211-1]. Additionally, the condition is inherited in an autosomal recessive manner, and stillbirth has been observed in some cases [medquad_ans_005211-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_005211-1`

### Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003966-1` (score=0.7000) - What are the signs and symptoms of Metachromatic leukodystrophy due to saposin B deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy due to sapos...
2. `medquad_ans_0000647-1` (score=0.6919) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the sub...
3. `medquad_ans_0003566-1` (score=0.6808) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to def...
4. `medquad_ans_0003965-2` (score=0.6429) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the ta...
5. `medquad_ans_0000647-2` (score=0.6366) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigra...

**Answer**:

The symptoms of Metachromatic leukodystrophy include abnormality of the periventricular white matter, Babinski sign, CNS demyelination, decreased nerve conduction velocity, developmental regression, dysarthria, dysphagia, gait ataxia, hyperreflexia, hyporeflexia, loss of speech, mental deterioration, muscular hypotonia, peripheral demyelination, polyneuropathy, seizures, spastic tetraparesis, urinary incontinence, and variable expressivity [medquad_ans_0003966-1]. These symptoms result from the accumulation of sulfatides in cells, particularly in the nervous system, leading to progressive destruction of white matter in the central and peripheral nervous systems [medquad_ans_0000647-1].

**Supported citations**: `medquad_ans_0003966-1`, `medquad_ans_0000647-1`
**Unsupported citations**: _none_

### Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.806, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000673-4` (score=0.7000) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
2. `medquad_ans_0000186-1` (score=0.7000) - Chronic granulomatous disease is a disorder that causes the immune system to malfunction, resulting in a form of immunodeficiency. Immunodeficiencies are conditions in which the immune system is not able to protect th...
3. `medquad_ans_0000362-4` (score=0.6122) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000186-4` (score=0.6122) - When chronic granulomatous disease is caused by mutations in the CYBB gene, the condition is inherited in an X-linked recessive pattern. The CYBB gene is located on the X chromosome, which is one of the two sex chromo...
5. `medquad_ans_0000095-4` (score=0.5272) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...

**Answer**:

Chronic granulomatous disease is not described as being inherited in the provided context. The information states that the condition causes immune system malfunction leading to immunodeficiency and recurrent infections, with symptoms often appearing in childhood, but there is no mention of inheritance patterns such as autosomal dominant or unknown inheritance [medquad_ans_0000186-1]. Therefore, based on the given context, it cannot be determined whether chronic granulomatous disease is inherited.

**Supported citations**: `medquad_ans_0000186-1`
**Unsupported citations**: _none_

### Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.978, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000061-2__dup1` (score=0.7000) - There is no cure for central cord syndrome although some people recover near-normal function. There is no standard course of treatment, although drug therapy, surgery, and rest are often part of the program. Magnetic ...
2. `medquad_ans_0000061-1__dup1` (score=0.6803) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to a...
3. `medquad_ans_0000061-3__dup1` (score=0.6710) - The prognosis for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict he likeliho...
4. `medquad_ans_0000061-4__dup1` (score=0.5120) - Our understanding of central cord syndrome has increased greatly in recent decades as a result of research funded conducted by the National Institute of Neurological Disorders and Stroke (NINDS). Much of this research...
5. `medquad_ans_0000062-1__dup2` (score=0.4406) - Central pain syndrome is a neurological condition caused by damage to or dysfunction of the central nervous system (CNS), which includes the brain, brainstem, and spinal cord. This syndrome can be caused by stroke, mu...

**Answer**:

The outlook for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict the likelihood that neurological recovery may occur naturally. Those who receive medical intervention soon after their injury often have good outcomes. Many people with the disorder recover substantial function after their initial injury, and the ability to walk is recovered in most cases, although some impairment may remain. Improvement occurs first in the legs, then the bladder, and may be seen in the arms. Hand function recovers last, if at all [medquad_ans_0000061-1__dup1]. Recent reports indicate that earlier surgery may improve chances for recovery, and surgery can be beneficial in individuals with persistent compression of the spinal cord and ongoing neurological deterioration [medquad_ans_0000061-2__dup1]. There is no cure, and recovery is generally not complete

**Supported citations**: `medquad_ans_0000061-1__dup1`, `medquad_ans_0000061-2__dup1`
**Unsupported citations**: _none_

### Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Scores**: context_relevance=0.600, groundedness=0.939, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000094-13` (score=0.7000) - - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no ...
2. `medquad_ans_0000094-1__dup6` (score=0.6533) - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no ap...
3. `medquad_ans_0000094-4__dup3` (score=0.6527) - Cyclic vomiting syndrome is more common in children than adults, although reports of the syndrome in adults have increased in recent years.1 Usually, children are about 5 years old when diagnosed with cyclic vomiting ...
4. `medquad_ans_0000094-11` (score=0.6089) - A person should stay away from known triggers, especially during the well phase, as well as - get adequate sleep to prevent exhaustion - treat sinus problems or allergies - seek help on reducing stress and anxiety - a...
5. `medquad_ans_0000094-3__dup3` (score=0.5753) - The cause of cyclic vomiting syndrome is unknown. However, some experts believe that some possible problems with bodily functions may contribute to the cause, such as the following: - gastrointestinal motilitythe way ...

**Answer**:

The cause of cyclic vomiting syndrome is unknown [medquad_ans_0000094-13].

**Supported citations**: `medquad_ans_0000094-13`
**Unsupported citations**: _none_

### Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Scores**: context_relevance=0.875, groundedness=0.922, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000005-19` (score=0.7000) - Most insurance plans, including health maintenance organizations (HMOs), will cover treatment for anxiety disorders. Check with your insurance company and find out. If you dont have insurance, the Health and Human Ser...
2. `medquad_ans_0000176-1__dup2` (score=0.6701) - All kids misbehave some times. And some may have temporary behavior problems due to stress. For example, the birth of a sibling, a divorce, or a death in the family may cause a child to act out. Behavior disorders are...
3. `medquad_ans_0000180-1__dup2` (score=0.6690) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not alwa...
4. `medquad_ans_0000776-5` (score=0.6289) - These resources address the diagnosis or management of Partington syndrome: - American Academy of Child and Adolescent Psychiatry: Services in School for Children with Special Needs - American Academy of Pediatrics: W...
5. `medquad_ans_0000006-4__dup1` (score=0.5938) - Several components of the NIH support research on developmental disorders such as ADHD. Research programs of the NINDS, the National Institute of Mental Health (NIMH), and the National Institute of Child Health and Hu...

**Answer**:

Yes, there is information about child mental health. Warning signs of more serious behavior problems in children include harming or threatening themselves or others, damaging property, lying or stealing, poor school performance, skipping school, early use of smoking, drinking, or drugs, early sexual activity, frequent tantrums and arguments, and consistent hostility towards authority figures. Additionally, problems in more than one setting (at school, at home, with peers), changes in appetite or sleep, social withdrawal or fear of previously harmless things, returning to younger behaviors like bedwetting, signs of being upset such as sadness or tearfulness, signs of self-destructive behavior like head-banging or sudden injuries, and repeated thoughts of death may indicate a serious mental health issue [medquad_ans_0000176-1__dup2]. It is important to recognize and treat mental illnesses in children early, as they can become a regular part of the child's behavior and are

**Supported citations**: `medquad_ans_0000176-1__dup2`
**Unsupported citations**: _none_

### Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.817, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000246-5` (score=0.7000) - These resources address the diagnosis or management of Cowden syndrome: - Gene Review: Gene Review: PTEN Hamartoma Tumor Syndrome (PHTS) - Genetic Testing Registry: Cowden syndrome - Genetic Testing Registry: Cowden s...
2. `medquad_ans_0000673-2__dup1` (score=0.7000) - What are the signs and symptoms of Behcet's disease? Symptoms of Behcet's disease include recurrent ulcers in the mouth (resembling canker sores) and on the genitals, and eye inflammation (uveitis). The disorder may a...
3. `medquad_ans_0000003-2__dup3` (score=0.6122) - Signs and symptoms of Cushing's syndrome vary, but most people with the disorder have upper body obesity, a rounded face, increased fat around the neck, and relatively slender arms and legs. Children tend to be obese ...
4. `medquad_ans_0000246-1` (score=0.6122) - Cowden syndrome is a disorder characterized by multiple noncancerous, tumor-like growths called hamartomas and an increased risk of developing certain cancers. Almost everyone with Cowden syndrome develops hamartomas....
5. `medquad_ans_0001618-3` (score=0.5272) - What causes Cowden syndrome? Most cases of Cowden syndrome are caused by changes (mutations) in the PTEN gene. PTEN is a tumor suppressor gene which means that it encodes a protein that helps keep cells from growing a...

**Answer**:

The provided context does not include information about the symptoms of Cowden syndrome. While it lists resources related to the diagnosis and management of Cowden syndrome, such as genetic testing and diagnostic approaches, it does not specify the clinical signs or symptoms associated with the condition [medquad_ans_0000246-5]. Therefore, based on the given context, the symptoms of Cowden syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000246-5`
**Unsupported citations**: _none_

### Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.742, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000395-5` (score=0.7000) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
2. `medquad_ans_0000395-1` (score=0.6898) - Fukuyama congenital muscular dystrophy is an inherited condition that predominantly affects the muscles, brain, and eyes. Congenital muscular dystrophies are a group of genetic conditions that cause muscle weakness an...
3. `medquad_ans_0000395-2` (score=0.6793) - Fukuyama congenital muscular dystrophy is seen almost exclusively in Japan, where it is the second most common form of childhood muscular dystrophy (after Duchenne muscular dystrophy). Fukuyama congenital muscular dys...
4. `medquad_ans_0000576-5` (score=0.5855) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...
5. `medquad_ans_0006267-3` (score=0.5595) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...

**Answer**:

The provided context does not specify any treatments for Fukuyama congenital muscular dystrophy. While it details the signs, symptoms, and complications of the condition, including muscle weakness, brain abnormalities, seizures, vision impairment, and aspiration pneumonia, no information on therapeutic interventions or management strategies is included [medquad_ans_0000395-1]. Therefore, based on the given context, treatments for Fukuyama congenital muscular dystrophy cannot be determined.

**Supported citations**: `medquad_ans_0000395-1`
**Unsupported citations**: _none_

### Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000004_6-1` (score=0.7000) - Key Points - Childhood Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two types of childhood Hodgkin lymphoma. - Epstein-Barr virus infection increases the risk o...
2. `medquad_ans_0000483-1__dup1` (score=0.5828) - Hodgkin disease is a type of lymphoma. Lymphoma is a cancer of a part of the immune system called the lymph system. The first sign of Hodgkin disease is often an enlarged lymph node. The disease can spread to nearby l...
3. `medquad_ans_0000004_7-1` (score=0.5808) - Key Points - Childhood non-Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - The main types of lymphoma are Hodgkin lymphoma and non-Hodgkin lymphoma. - There are three major ...
4. `medquad_ans_0000004_1-1` (score=0.5534) - Key Points - Adult Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two main types of Hodgkin lymphoma: classical and nodular lymphocyte-predominant. - Age, gender,...
5. `medquad_ans_0000004_6-3` (score=0.5365) - Signs of childhood Hodgkin lymphoma include swollen lymph nodes, fever, night sweats, and weight loss. These and other signs and symptoms may be caused by childhood Hodgkin lymphoma or by other conditions. Check with ...

**Answer**:

Tests that examine the lymph system are used to detect (find) and diagnose childhood Hodgkin lymphoma [medquad_ans_0000004_6-1].

**Supported citations**: `medquad_ans_0000004_6-1`
**Unsupported citations**: _none_

### Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.929, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0003361-1` (score=0.7000) - KBG syndrome is a rare condition characterized mainly by skeletal abnormalities, distinctive facial features, and intellectual disability. Specific signs and symptoms may include delayed bone age; abnormalities of the...
2. `medquad_ans_0000117-4` (score=0.7000) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
3. `medquad_ans_0006446-4` (score=0.6122) - Is Williams syndrome inherited?
4. `medquad_ans_0000556-1` (score=0.6122) - KBG syndrome is a rare disorder that affects several body systems. "KBG" represents the surname initials of the first families diagnosed with the disorder. Common signs and symptoms in individuals with this condition ...
5. `medquad_ans_0000673-4` (score=0.5272) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.

**Answer**:

Yes, KBG syndrome can be inherited. It is inherited in an autosomal dominant manner, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one affected parent. However, in some cases, the condition occurs sporadically due to a new mutation, with no family history of the disorder [medquad_ans_0003361-1]. Additionally, KBG syndrome is sometimes caused by a mutation in the ANKRD11 gene, and it is inherited in an autosomal dominant pattern [medquad_ans_0003361-1].

**Supported citations**: `medquad_ans_0003361-1`
**Unsupported citations**: _none_

### Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Scores**: context_relevance=0.833, groundedness=0.993, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000195-1__dup1` (score=0.7000) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
2. `medquad_ans_0000194-1__dup1` (score=0.6988) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
3. `medquad_ans_0000193-1__dup1` (score=0.6704) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000029-1` (score=0.0993) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. Howeve...
5. `medquad_ans_0000029-2` (score=0.0888) - Aicardi-Goutieres syndrome is a rare disorder. Its exact prevalence is unknown.

**Answer**:

Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and physical disability. Additional symptoms may include epilepsy, painful, itchy skin lesions (chilblains), vision problems, and joint stiffness. Symptoms usually progress over several months before the disease course stabilizes. There are six different types of Aicardi-Goutieres syndrome, which are distinguished by the gene that causes the condition: TREX1, RNASEH2A, RNASEH2B, RNASEH2C, SAMHD1, and ADAR genes. Most cases are inherited in an autosomal recessive pattern, although rare autosomal dominant cases have been reported. Treatment is symptomatic and supportive [medquad_ans_0000195-1__dup1].

**Supported citations**: `medquad_ans_0000195-1__dup1`
**Unsupported citations**: _none_

### Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Scores**: context_relevance=0.917, groundedness=0.882, answer_relevance=0.783

**Top docs**:

1. `medquad_ans_0000026_2-7` (score=0.7000) - Key Points - There are different types of treatment for patients with transitional cell cancer of the renal pelvis and ureter. - One type of standard treatment is used: - Surgery - New types of treatment are being tes...
2. `medquad_ans_0000026_1-7` (score=0.7000) - Key Points - There are different types of treatment for patients with renal cell cancer. - Five types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Biologic therapy - Targeted therapy ...
3. `medquad_ans_0000006_2-7` (score=0.4062) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000026_3-7` (score=0.4009) - Key Points - There are different types of treatment for patients with Wilms tumor and other childhood kidney tumors. - Children with Wilms tumor or other childhood kidney tumors should have their treatment planned by ...
5. `medquad_ans_0000003_5-7` (score=0.3912) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context discusses clinical trials for transitional cell cancer of the renal pelvis and ureter, not for renal cell cancer. Therefore, there is no information available in the context about research or clinical trials specifically for renal cell cancer. [medquad_ans_0000026_2-7]

**Supported citations**: `medquad_ans_0000026_2-7`
**Unsupported citations**: _none_

### Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Scores**: context_relevance=0.857, groundedness=0.502, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000190-9` (score=0.7000) - - Primary biliary cirrhosis is a chronic disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. - When chronic inflammation damages the bile ducts, bile and toxi...
2. `medquad_ans_0000190-6` (score=0.6565) - Treatment for primary biliary cirrhosis depends on how early a health care provider diagnoses the disease and whether complications are present. In the early stages of primary biliary cirrhosis, treatment can slow the...
3. `medquad_ans_0000190-3__dup2` (score=0.6273) - The first and most common symptoms of primary biliary cirrhosis are - fatigue, or feeling tired - itching skin, and darkened skin in itching areas due to scratching - dry eyes and mouth Some people may have jaundice, ...
4. `medquad_ans_0000190-2__dup2` (score=0.6115) - The causes of primary biliary cirrhosis are unknown. Most research suggests it is an autoimmune disease. The immune system protects people from infection by identifying and destroying bacteria, viruses, and other pote...
5. `medquad_ans_0000190-5__dup1` (score=0.5991) - A health care provider may use the following tests to diagnose primary biliary cirrhosis: - a medical and family history - a physical exam - blood tests - imaging tests - a liver biopsy A health care provider usually ...

**Answer**:

For primary biliary cirrhosis, treatment depends on the stage of the disease and the presence of complications. In the early stages, treatment can slow the progression of liver damage to cirrhosis. In the early stages of cirrhosis, the goals are to slow tissue scarring in the liver and prevent complications. As cirrhosis progresses, additional treatments and hospitalization may be needed to manage complications. Health care providers prescribe ursodiol (Actigall, Urso) to treat primary biliary cirrhosis. Ursodiol is a nontoxic bile acid taken orally that replaces more toxic bile acids normally produced by the liver, reducing levels of bilirubin and liver enzymes in the blood. Early treatment with ursodiol reduces the likelihood of needing a liver transplant and improves survival; however, it can still slow liver damage even when started later in the disease. A liver transplant may be considered when cirrhosis leads to

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Scores**: context_relevance=1.000, groundedness=0.914, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=0.7000) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People ...
2. `medquad_ans_0000916-1__dup2` (score=0.6428) - Summary : A tsunami is a series of huge ocean waves created by an underwater disturbance. Causes include earthquakes, landslides, volcanic eruptions, or meteorites--chunks of rock from space that strike the surface of...
3. `medquad_ans_0000967-1__dup1` (score=0.0765) - Summary : Playing in the water - whether swimming, boating or diving - can be fun. It can also be dangerous, especially for children. Being safe can help prevent injuries and drowning. To stay safe in the water - Avoi...
4. `medquad_ans_0001853-3` (score=0.0765) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0000378-2` (score=0.0669) - Floating-Harbor syndrome is a rare disorder; only about 50 cases have been reported in the medical literature.

**Answer**:

Drowning is a condition that occurs when a person takes in too much water, particularly into the lungs, which can lead to respiratory failure and death. It can happen quickly and silently, even with as little as an inch or two of water. Babies can drown in a sink or bathtub, and preschoolers are most at risk in swimming pools. People with seizure disorders are also at increased risk in the water. Drowning is the most common cause of death related to a tsunami [medquad_ans_0000916-1__dup2]. Precautions include fencing around pools, supervising children near any body of water, avoiding alcohol or sedatives when swimming or boating, wearing life jackets when boating, and learning CPR [medquad_ans_0000299-1__dup2].

**Supported citations**: `medquad_ans_0000916-1__dup2`, `medquad_ans_0000299-1__dup2`
**Unsupported citations**: _none_

### Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Scores**: context_relevance=1.000, groundedness=0.253, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002058-3` (score=0.7000) - What causes Ehlers-Danlos syndrome, kyphoscoliosis type? Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is caused by changes (mutations) in the PLOD1 gene. This gene encodes an enzyme that helps process molecules w...
2. `medquad_ans_0002058-5` (score=0.6993) - How is Ehlers-Danlos syndrome, kyphoscoliosis type diagnosed? A diagnosis of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is typically based on the presence of characteristic signs and symptoms. The following tes...
3. `medquad_ans_0000313-1` (score=0.6391) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and sympt...
4. `medquad_ans_0002058-1` (score=0.6249) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragi...
5. `medquad_ans_0000313-2` (score=0.5444) - Although it is difficult to estimate the overall frequency of Ehlers-Danlos syndrome, the combined prevalence of all types of this condition may be about 1 in 5,000 individuals worldwide. The hypermobility and classic...

**Answer**:

Ehlers-Danlos syndrome, kyphoscoliosis type is a disorder of the connective tissues that affects the skin, bones, blood vessels, and other organs and tissues. It is caused by mutations in the PLOD1 gene, which encodes an enzyme essential for collagen formation and stability. These mutations result in reduced levels of functional enzyme, disrupting collagen networks and leading to weakened connective tissues. This condition is characterized by kyphoscoliosis (abnormal curvature of the spine) and other signs and symptoms related to connective tissue fragility. Diagnosis is typically based on characteristic signs and symptoms, and may be confirmed through urine tests, skin biopsy, or genetic testing for mutations in the PLOD1 gene [medquad_ans_0002058-3, medquad_ans_00058-5, medquad_ans_0000313-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0002058-3, medquad_ans_00058-5, medquad_ans_0000313-1`

### Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.923, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001860-2` (score=0.7000) - What are the signs and symptoms of Dihydropteridine reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropteridine reductase deficiency. If the information is...
2. `medquad_ans_0001860-1` (score=0.6421) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...
3. `medquad_ans_0006567-2` (score=0.6084) - What are the signs and symptoms of 2,4-Dienoyl-CoA reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for 2,4-Dienoyl-CoA reductase deficiency. If the information is a...
4. `medquad_ans_0004302-1` (score=0.5511) - What are the signs and symptoms of NADH cytochrome B5 reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for NADH cytochrome B5 reductase deficiency. If the informatio...
5. `medquad_ans_0001612-1` (score=0.5329) - What are the signs and symptoms of Cortisone reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Cortisone reductase deficiency. If the information is available, th...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include cognitive impairment, microcephaly, cerebral calcification, choreoathetosis, dysphagia, dystonia, episodic fever, excessive salivation, hyperphenylalaninemia, hypertonia, infantile onset, intellectual disability, irritability, muscular hypotonia, myoclonus, progressive neurologic deterioration, seizures, tremor, and variable expressivity [medquad_ans_0001860-2]. These symptoms arise due to impaired renewal of tetrahydrobiopterin (BH4), leading to elevated phenylalanine levels and reduced neurotransmitter levels such as dopamine and serotonin [medquad_ans_0001860-1].

**Supported citations**: `medquad_ans_0001860-2`, `medquad_ans_0001860-1`
**Unsupported citations**: _none_

### Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Scores**: context_relevance=0.857, groundedness=0.963, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000669-2` (score=0.7000) - The prevalence of monilethrix is unknown.
2. `medquad_ans_0000669-1` (score=0.0889) - Monilethrix is a condition that affects hair growth. Its most characteristic feature is that individual strands of hair have a beaded appearance like the beads of a necklace. The name monilethrix comes from the Latin ...
3. `medquad_ans_0000484-2` (score=0.0787) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
4. `medquad_ans_0000669-3` (score=0.0787) - Monilethrix is caused by mutations in one of several genes. Mutations in the KRT81 gene, the KRT83 gene, the KRT86 gene, or the DSG4 gene account for most cases of monilethrix. These genes provide instructions for mak...
5. `medquad_ans_0000770-2` (score=0.0689) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.

**Answer**:

The exact incidence of monilethrix is unknown, but it is estimated to affect 1 in 87,000 people [medquad_ans_0000484-2].

**Supported citations**: `medquad_ans_0000484-2`
**Unsupported citations**: _none_

### Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Scores**: context_relevance=1.000, groundedness=0.505, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.7000) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...
2. `medquad_ans_0002733-2` (score=0.6802) - What are the signs and symptoms of Hartnup disease? The signs and symptoms of Hartnup disease may vary and include the following: Skin findings: sensitivity to sunlight Neurologic symptoms: ataxia, spasticity, headach...
3. `medquad_ans_0002152-1` (score=0.0669) - Erdheim-Chester disease is a rare condition that can affect many different organs of the body. This condition, which usually affects adults, is characterized by excessive production and accumulation of histiocytes (sp...
4. `medquad_ans_0001853-3` (score=0.0669) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0000108-2__dup1` (score=0.0577) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system, making it a type of aminoaciduria. It is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. The condition may be diagnosed through newborn screening tests. Most individuals are asymptomatic, but those who do show symptoms typically experience onset between ages 3 and 9, although it can occasionally present in adulthood. Signs and symptoms include skin photosensitivity, neurologic findings (such as ataxia, spasticity, headaches, and hypotonia), psychiatric symptoms (like anxiety, emotional instability, and mood changes), and ocular findings (such as double vision, nystagmus, strabismus, and photophobia). Symptoms may be triggered by sunlight exposure, fever, drugs, or emotional or physical stress and often last 1–4 weeks before spontaneous

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Scores**: context_relevance=0.833, groundedness=0.746, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000428-5` (score=0.7000) - These resources address the diagnosis or management of glycogen storage disease type VII: - Genetic Testing Registry: Glycogen storage disease, type VII - The Swedish Information Centre for Rare Diseases These resourc...
2. `medquad_ans_0002615-4` (score=0.6295) - How is glycogen storage disease type 13 diagnosed? Glycogen storage disease type 13 is diagnosed by taking a sample of muscle tissue (muscle biopsy) to determine if there is enough of the chemical enolase working in t...
3. `medquad_ans_0002615-3` (score=0.6273) - What causes glycogen storage disease type 13? Glycogen storage disease type 13 (GSD13) is caused by changes (mutations) in the ENO3 gene. Glycogen is a substance that is stored in muscle tissue and is used as an impor...
4. `medquad_ans_0000427-5` (score=0.5527) - These resources address the diagnosis or management of glycogen storage disease type VI: - Gene Review: Gene Review: Glycogen Storage Disease Type VI - Genetic Testing Registry: Glycogen storage disease, type VI These...
5. `medquad_ans_0000423-5` (score=0.5467) - These resources address the diagnosis or management of glycogen storage disease type III: - Gene Review: Gene Review: Glycogen Storage Disease Type III - Genetic Testing Registry: Glycogen storage disease type III The...

**Answer**:

The provided context does not contain information about the genetic changes related to glycogen storage disease type VII. [medquad_ans_0000428-5] mentions resources for diagnosis or management of glycogen storage disease type VII, including genetic testing, but does not specify the genetic changes associated with it. Therefore, based on the given context, the genetic changes related to glycogen storage disease type VII cannot be determined.

**Supported citations**: `medquad_ans_0000428-5`
**Unsupported citations**: _none_

### Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.964, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000977-1` (score=0.7000) - Thiamine-responsive megaloblastic anemia syndrome is a rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Megaloblastic anemia occurs when a person has a low numb...
2. `medquad_ans_0006061-1` (score=0.6994) - Thiamine-responsive megaloblastic anemia syndrome is a very rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Affected individuals begin to show symptoms of this...
3. `medquad_ans_0000977-5` (score=0.6994) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megal...
4. `medquad_ans_0000977-2` (score=0.6817) - Thiamine-responsive megaloblastic anemia syndrome has been reported in approximately 30 families worldwide. Its prevalence is unknown.
5. `medquad_ans_0006061-2` (score=0.6518) - What are the signs and symptoms of Thiamine responsive megaloblastic anemia syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thiamine responsive megaloblastic anemia syndrom...

**Answer**:

The treatments for thiamine-responsive megaloblastic anemia syndrome include high doses of vitamin B1 (thiamine), which can treat the anemia. In some cases, thiamine treatment can reduce the amount of insulin a person needs for diabetes management. People with this syndrome usually require insulin to treat their diabetes [medquad_ans_0000977-1]. However, it is unclear whether thiamine treatment can improve or prevent hearing loss [medquad_ans_0000977-1].

**Supported citations**: `medquad_ans_0000977-1`
**Unsupported citations**: _none_

### Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.915, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004939-1` (score=0.7000) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the ...
2. `medquad_ans_0000587-4__dup1` (score=0.6925) - Is there a cure or treatment for autosomal recessive polycystic kidney disease? Although a cure or treatment for the underlying genetic cause of autosomal recessive polycystic kidney disease does not exist, advancemen...
3. `medquad_ans_0004939-2` (score=0.6890) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their abilit...
4. `medquad_ans_0000804-1` (score=0.6667) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the...
5. `medquad_ans_0000804-5` (score=0.6327) - These resources address the diagnosis or management of polycystic kidney disease: - Gene Review: Gene Review: Polycystic Kidney Disease, Autosomal Dominant - Gene Review: Gene Review: Polycystic Kidney Disease, Autoso...

**Answer**:

Treatment for polycystic kidney disease is tailored to the individual based on their signs and symptoms. For autosomal dominant polycystic kidney disease (ADPKD), which typically presents between ages 30 and 40 (though sometimes earlier), management includes controlling high blood pressure with medication, managing urinary tract infections with antibiotics, and addressing complications such as kidney failure, which may require dialysis or kidney transplantation [medquad_ans_0004939-1]. In cases of severe kidney enlargement, nephrectomy (removal of one or both kidneys) may be performed [medquad_ans_0000587-4__dup1]. For autosomal recessive polycystic kidney disease (ARPKD), which usually manifests in infancy or early childhood and can be life-threatening, treatment is primarily symptomatic and supportive. This includes mechanical ventilation for respiratory issues due to enlarged kidneys, dialysis during

**Supported citations**: `medquad_ans_0004939-1`, `medquad_ans_0000587-4__dup1`
**Unsupported citations**: _none_

### Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Scores**: context_relevance=0.818, groundedness=0.932, answer_relevance=0.745

**Top docs**:

1. `medquad_ans_0000061-5__dup2` (score=0.7000) - Scientists are making rapid progress in understanding the complexities of rheumatoid arthritis. They are learning more about how and why it develops and why some people have more severe symptoms than others. Research ...
2. `medquad_ans_0000782-1__dup1` (score=0.6047) - Rheumatoid arthritis (RA) is a form of arthritis that causes pain, swelling, stiffness and loss of function in your joints. It can affect any joint but is common in the wrist and fingers. More women than men get rheum...
3. `medquad_ans_0000061-9` (score=0.5168) - Scientists believe that rheumatoid arthritis may result from the interaction of many factors such as genetics, hormones, and the environment. Although rheumatoid arthritis sometimes runs in families, the actual cause ...
4. `medquad_ans_0000061-3__dup3` (score=0.5091) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by i...
5. `medquad_ans_0000061-2__dup2` (score=0.5018) - Actual Cause Is Unknown Scientists believe that rheumatoid arthritis may result from the interaction of many factors such as genetics, hormones, and the environment. Although rheumatoid arthritis sometimes runs in fam...

**Answer**:

Research on rheumatoid arthritis is focused on understanding how and why the disease develops, including identifying possible triggers such as infectious agents like viruses and bacteria that may contribute to the disease in individuals with a genetic predisposition [medquad_ans_0000061-5__dup2]. Scientists are also investigating the complex relationships between hormonal, nervous, and immune systems to explain why more women than men develop rheumatoid arthritis, including the role of steroid hormones such as estrogen and testosterone [medquad_ans_0000061-5__dup2]. Additionally, researchers are exploring why the disease often improves during pregnancy [medquad_ans_0000061-5__dup2].

Efforts are underway to develop drugs that can reduce inflammation and slow or stop the disease with few side effects [medquad_ans_0000061-5__dup2]. For those interested in clinical trials

**Supported citations**: `medquad_ans_0000061-5__dup2`
**Unsupported citations**: _none_

### Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002561-4` (score=0.7000) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (an...
2. `medquad_ans_0002561-2` (score=0.7000) - What are the signs and symptoms of Glass-Chapman-Hockley syndrome? Glass-Chapman-Hockley syndrome has only been described in one family with five affected family members in three generations. The signs and symptoms se...
3. `medquad_ans_0002561-1` (score=0.6520) - The Glass-Chapman-Hockley syndrome is a very rare disease. To date, the syndrome has only been reported in one family with five members affected in three generations. The first patients were two brothers that had an a...
4. `medquad_ans_0003845-5` (score=0.3786) - How might McCune Albright syndrome be treated? Although there is no cure for McCune Albright syndrome (MAS), drug treatments may help some of the endocrine symptoms, and surgery can help repair some of the bone proble...
5. `medquad_ans_0003103-4` (score=0.0679) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...

**Answer**:

The treatment for Glass-Chapman-Hockley syndrome is typically surgery, which is based on the person's specific signs and symptoms. The goal of the surgery is to increase the space in the front (anterior) part of the skull. The operation is usually performed when the person is between 9 to 12 months of age. If other sutures, other than the coronal suture, are involved, other surgeries may be performed [medquad_ans_0002561-4].

**Supported citations**: `medquad_ans_0002561-4`
**Unsupported citations**: _none_

### Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.762, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000868-1` (score=0.7000) - Renpenning syndrome is a disorder that almost exclusively affects males, causing developmental delay, moderate to severe intellectual disability, and distinctive physical features. Individuals with Renpenning syndrome...
2. `medquad_ans_0003845-5` (score=0.3497) - How might McCune Albright syndrome be treated? Although there is no cure for McCune Albright syndrome (MAS), drug treatments may help some of the endocrine symptoms, and surgery can help repair some of the bone proble...
3. `medquad_ans_0000868-2` (score=0.0900) - Renpenning syndrome is a rare disorder; its prevalence is unknown. More than 60 affected individuals in at least 15 families have been identified.
4. `medquad_ans_0000868-5` (score=0.0787) - These resources address the diagnosis or management of Renpenning syndrome: - Genetic Testing Registry: Renpenning syndrome 1 - Greenwood Genetics Center: X-Linked Intellectual Disability - Kennedy Krieger Institute: ...
5. `medquad_ans_0000247-2__dup2` (score=0.0787) - There is no cure for Rett syndrome. Treatment for the disorder is symptomatic, focusing on the management of symptoms, and supportive. Medication may be needed for breathing irregularities and motor difficulties, and ...

**Answer**:

There are no treatments mentioned for Renpenning syndrome in the provided context. The available information describes the symptoms, features, and genetic basis of the disorder but does not include any details about treatment options [medquad_ans_0000868-1]. Therefore, based on the given context, treatments for Renpenning syndrome are not specified.

**Supported citations**: `medquad_ans_0000868-1`
**Unsupported citations**: _none_

### Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Scores**: context_relevance=1.000, groundedness=0.444, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001100-1__dup1` (score=0.7000) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is a...
2. `medquad_ans_0001111-1` (score=0.5197) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2N? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2N. If the information is ava...
3. `medquad_ans_0001112-1` (score=0.4468) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2O? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2O. If the information is ava...
4. `medquad_ans_0001093-1__dup1` (score=0.4359) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is ava...
5. `medquad_ans_0000173-1` (score=0.3592) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...

**Answer**:

The provided context does not contain information about the symptoms of Charcot-Marie-Tooth disease type 2G. [medquad_ans_0001100-1] and [medquad_ans_0001111-1] discuss symptoms for types 2B2 and 2N, respectively, but there is no mention of type 2G. Therefore, the symptoms of Charcot-Marie-Tooth disease type 2G cannot be determined from the given context.

**Supported citations**: `medquad_ans_0001111-1`
**Unsupported citations**: `medquad_ans_0001100-1`

### Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Scores**: context_relevance=1.000, groundedness=0.960, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001393-1` (score=0.7000) - Coccygodynia is a rare condition in that causes pain in and around the coccyx (tailbone). Although various causes have been described for the condition, the more common causes are direct falls and injury.
2. `medquad_ans_0001393-2` (score=0.0901) - What signs and symptoms are associated with coccygodynia? The classic symptom is pain when pressure is applied to the tailbone, such as when sitting on a hard chair. Symptoms usually improve with relief of pressure wh...
3. `medquad_ans_0000676-2__dup1` (score=0.0798) - What are the symptoms of Bell's palsy?
4. `medquad_ans_0001393-4` (score=0.0798) - What treatment is available for coccygodynia? Treatment for coccygodynia generally falls into conservative management or surgical intervention categories. The conservative approach typically includes hot sitz baths, N...
5. `medquad_ans_0000108-2__dup1` (score=0.0698) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The symptoms of coccygodynia include pain when pressure is applied to the tailbone, such as when sitting on a hard chair, which usually improves with relief of pressure when standing or walking. Other symptoms include immediate and severe pain when moving from sitting to standing, pain during bowel movements, pain during sex, and a deep ache in the region of the tailbone [medquad_ans_0001393-2].

**Supported citations**: `medquad_ans_0001393-2`
**Unsupported citations**: _none_

### Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Scores**: context_relevance=1.000, groundedness=0.943, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0004025-1` (score=0.7000) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgen...
2. `medquad_ans_0003270-2` (score=0.6688) - What are the signs and symptoms of Isolated corpus callosum agenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Isolated corpus callosum agenesis. If the information is availab...
3. `medquad_ans_0002446-2` (score=0.5480) - What are the signs and symptoms of Frontonasal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontonasal dysplasia. If the information is available, the table below incl...
4. `medquad_ans_0005878-1` (score=0.5277) - What are the signs and symptoms of Stargardt macular degeneration absent or hypoplastic corpus callosum mental retardation and dysmorphic features? The Human Phenotype Ontology provides the following list of signs and...
5. `medquad_ans_0002760-2` (score=0.5187) - What are the signs and symptoms of Hemifacial microsomia? People with hemifacial microsomia may have various signs and symptoms, including: Facial asymmetry Abnormalities of the outer ear such as absence, reduced size...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, and microcephaly. These symptoms are listed in the Human Phenotype Ontology as part of the signs and symptoms for this condition [medquad_ans_0004025-1].

**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

### Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.994, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000562-5` (score=0.7000) - These resources address the diagnosis or management of Klippel-Feil syndrome: - Genetic Testing Registry: Klippel Feil syndrome - Genetic Testing Registry: Klippel-Feil syndrome 1, autosomal dominant - Genetic Testing...
2. `medquad_ans_0003402-5` (score=0.6908) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to d...
3. `medquad_ans_0003402-2` (score=0.6525) - What are the signs and symptoms of Klippel Feil syndrome? Klippel Feil syndrome is characterized by the fusion of 2 or more spinal bones in the neck (cervical vertebrae). The condition is present from birth (congenita...
4. `medquad_ans_0003402-3` (score=0.6387) - What causes Klippel Feil syndrome (KFS)? The specific underlying causes and mechanisms of Klippel Feil syndrome (KFS)are not well understood. In general medical researchers believe KFS happens when the tissue of the e...
5. `medquad_ans_0003402-6` (score=0.6347) - How might Klippel-Feil syndrome be treated? There is no cure for Klippel Feil syndrome (KFS); treatment is generally symptomatic and supportive. Management depends on the features and severity in each person, and can ...

**Answer**:

Klippel Feil syndrome is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal abnormalities, and additional imaging studies may be needed to assess the extent of the abnormality. Other initial exams are needed to detect additional physical abnormalities or underlying conditions, including examination of the chest to rule out involvement of the heart and lungs, examination of the chest wall to detect possible rib anomalies, MRI for spinal stenosis or neurological deficits, ultrasound of the kidneys for renal abnormalities, hearing evaluation due to the high incidence of hearing loss, and various lab tests to assess organ function. Additional tests or consultations with specialists may be recommended depending on the features present in each person with KFS [medquad_ans_0003402-5].

**Supported citations**: `medquad_ans_0003402-5`
**Unsupported citations**: _none_

### Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.833, groundedness=0.510, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000222-1` (score=0.7000) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
2. `medquad_ans_0000222-4` (score=0.6625) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or c...
3. `medquad_ans_0000222-5` (score=0.6619) - These resources address the diagnosis or management of congenital diaphragmatic hernia: - Boston Children's Hospital - Children's Hospital of Philadelphia - Columbia University Medical Center: DHREAMS - Columbia Unive...
4. `medquad_ans_0000222-2` (score=0.6552) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
5. `medquad_ans_0000222-3` (score=0.6267) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...

**Answer**:

Congenital diaphragmatic hernia is a defect in the diaphragm resulting from abnormal development before birth. The diaphragm, composed of muscle and fibrous tissue, separates the abdominal and chest cavities. A defect allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs and potentially leading to pulmonary hypoplasia and life-threatening breathing difficulties at birth. The condition can range from a thinned area in the diaphragm to its complete absence. It is often classified by location: Bochdalek hernias (80–90% of cases, involving the side or back of the diaphragm), Morgagni hernias (about 2% of cases, involving the front part and less likely to cause severe symptoms), and rarer types affecting the central region or involving only a thin membrane. In 5 to 10 percent of cases, symptoms appear later in life,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Scores**: context_relevance=1.000, groundedness=0.533, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001088-1__dup1` (score=0.7000) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
2. `medquad_ans_0000173-1` (score=0.6785) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
3. `medquad_ans_0001094-1__dup1` (score=0.5676) - Charcot-Marie-Tooth disease type 1E (CMT1E) is a form of Charcot-Marie-Tooth disease, which is a group of rare conditions that affect the peripheral nerves. Signs and symptoms of CMT1E generally become apparent betwee...
4. `medquad_ans_0001093-1__dup1` (score=0.4851) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is ava...
5. `medquad_ans_0001100-1__dup1` (score=0.4738) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is a...

**Answer**:

Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves, which connect the brain and spinal cord to muscles and sensory cells that detect sensations such as touch, pain, heat, and sound. Defects in many different genes cause different forms of this disease, and there are over 40 types. Common symptoms include foot drop, foot deformity, loss of lower leg muscle, numbness in the foot or leg, slapping gait, weakness of the hips, legs, or feet, balance difficulties, clumsiness, muscle weakness in the feet, foot abnormalities such as high arches (pes cavus), flat feet (pes planus), or curled toes (hammer toes), difficulty flexing the foot or walking on the heel, and decreased sensitivity to touch, heat, and other sensations. The disease usually becomes apparent in adolescence or early adulthood but may onset at any time from early childhood through late adulthood. Symptoms

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Scores**: context_relevance=1.000, groundedness=0.914, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000685-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000669-2` (score=0.7000) - The prevalence of monilethrix is unknown.
3. `medquad_ans_0000669-1` (score=0.6122) - Monilethrix is a condition that affects hair growth. Its most characteristic feature is that individual strands of hair have a beaded appearance like the beads of a necklace. The name monilethrix comes from the Latin ...
4. `medquad_ans_0000205-4` (score=0.6122) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000784-4` (score=0.5272) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

Monilethrix is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, but they typically do not show signs and symptoms of the condition [medquad_ans_0000685-4]. This pattern of inheritance is consistent with the information provided for monilethrix [medquad_ans_0000669-1]. Therefore, monilethrix is inherited.

**Supported citations**: `medquad_ans_0000685-4`, `medquad_ans_0000669-1`
**Unsupported citations**: _none_

### Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Scores**: context_relevance=0.875, groundedness=0.452, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000215-15` (score=0.7000) - - Urinary tract infections (UTIs) usually occur when the body fails to remove bacteria rapidly from the urinary tract. - UTIs affect about 3 percent of children in the United States every year. - Most UTIs are not ser...
2. `medquad_ans_0000223-12` (score=0.6817) - - A urinary tract infection (UTI) is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscope. - Most UTIs are caused by bacteria that live in the bowel, ...
3. `medquad_ans_0000071-7__dup1` (score=0.6805) - Urinary tract infections (UTI) are a common bladder problem, especially as people age. UTIs are the second most common type of infection in the body. Each year, UTIs cause more than 8 million visits to health care pro...
4. `medquad_ans_0000223-2__dup2` (score=0.6414) - Most UTIs are caused by bacteria that live in the bowel, the part of the digestive tract where stool is changed from liquid to solid. The bacterium Escherichia coli (E. coli) causes most UTIs. The urinary tract has se...
5. `medquad_ans_0000215-4__dup2` (score=0.6332) - Urinary tract infections affect about 3 percent of children in the United States every year. UTIs account for more than 1 million visits to pediatricians offices every year.1

**Answer**:

Urinary tract infections (UTIs) in children are infections in the urinary tract caused by microorganisms, most commonly bacteria from the bowel, such as Escherichia coli (E. coli). These infections can occur anywhere in the urinary system, including the bladder, kidneys, and urethra, though they are most common in the bladder (cystitis). Symptoms may range from mild (e.g., slight burning with urination or unusual-smelling urine) to severe (e.g., severe pain and high fever), and some children may have no symptoms at all. UTIs in young children may indicate an underlying abnormality in the urinary tract that could lead to repeated infections. Most UTIs are not serious, but chronic kidney infections can cause permanent damage. Diagnosis is typically made by testing a urine sample, and treatment involves antibiotics. After recovery, parents should consult a healthcare provider to check for conditions like vesicoureteral reflux (VUR

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.989, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000804-1` (score=0.7000) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the...
2. `medquad_ans_0004939-2` (score=0.6592) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their abilit...
3. `medquad_ans_0004939-1` (score=0.6179) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the ...
4. `medquad_ans_0000587-4__dup1` (score=0.5423) - Is there a cure or treatment for autosomal recessive polycystic kidney disease? Although a cure or treatment for the underlying genetic cause of autosomal recessive polycystic kidney disease does not exist, advancemen...
5. `medquad_ans_0004945-1` (score=0.5352) - Polycystic liver disease is an inherited condition characterized by many cysts of various sizes scattered throughout the liver. Abdominal discomfort from swelling of the liver may occur; however, most affected individ...

**Answer**:

The complications of Polycystic Kidney Disease include dangerously high blood pressure (hypertension), pain in the back or sides, blood in the urine (hematuria), recurrent urinary tract infections, kidney stones, heart valve abnormalities, and an increased risk of an abnormal bulging (aneurysm) in the aorta or in blood vessels at the base of the brain. Aneurysms can be life-threatening if they tear or rupture [medquad_ans_0000804-1].

**Supported citations**: `medquad_ans_0000804-1`
**Unsupported citations**: _none_

### Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Scores**: context_relevance=1.000, groundedness=0.507, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000649-1__dup1` (score=0.7000) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contami...
2. `medquad_ans_0000123-4__dup2` (score=0.6903) - Four types of viruses cause most cases of viral gastroenteritis. Rotavirus Rotavirus is the leading cause of gastroenteritis among infants and young children. Rotavirus infections are most common in infants 3 to 15 mo...
3. `medquad_ans_0000123-6` (score=0.5858) - Viral gastroenteritis is usually diagnosed based on symptoms alone. People who have symptoms that are severe or last for more than a few days may want to see a health care provider for additional tests. A health care ...
4. `medquad_ans_0000949-1__dup1` (score=0.0776) - Viruses are capsules with genetic material inside. They are very tiny, much smaller than bacteria. Viruses cause familiar infectious diseases such as the common cold, flu and warts. They also cause severe illnesses su...
5. `medquad_ans_0000391-1__dup1` (score=0.0679) - Have you ever had the "stomach flu?" What you probably had was gastroenteritis - not a type of flu at all. Gastroenteritis is an inflammation of the lining of the intestines caused by a virus, bacteria or parasites. V...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by noroviruses, a group of related viruses. Infection leads to an inflammation of the stomach and intestines, with common symptoms including diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms are fever, headache, and body aches. The illness can spread from person to person, through contaminated food or water, or by touching a contaminated surface. Norovirus is the most common calicivirus and the most common cause of viral gastroenteritis in adults, often responsible for epidemics, especially from October to April. Symptoms typically appear 1 to 2 days after exposure and last for 1 to 3 days. There is no specific medicine to treat norovirus infections; treatment includes bed rest and plenty of fluids to prevent dehydration. Prevention can be achieved through proper hand washing and safe food preparation [medquad_ans_0000649-

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.900, groundedness=0.937, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000222-4` (score=0.7000) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or c...
2. `medquad_ans_0000222-3` (score=0.6909) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...
3. `medquad_ans_0000222-5` (score=0.6803) - These resources address the diagnosis or management of congenital diaphragmatic hernia: - Boston Children's Hospital - Children's Hospital of Philadelphia - Columbia University Medical Center: DHREAMS - Columbia Unive...
4. `medquad_ans_0000222-1` (score=0.6721) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
5. `medquad_ans_0001497-3` (score=0.6455) - What causes congenital diaphragmatic hernia? Congenital diaphragmatic hernia (CDH) can occur as an isolated finding, as part of a genetic syndrome or chromosome abnormality, or as part of a complex but nonsyndromic se...

**Answer**:

Genetic changes related to congenital diaphragmatic hernia include changes in single genes and chromosomal abnormalities that affect several genes. Some syndromes associated with congenital diaphragmatic hernia, such as Donnai-Barrow syndrome, Fryns syndrome, and Pallister-Killian mosaic syndrome, are caused by changes in single genes or chromosomal abnormalities. Researchers are also studying changes in several genes involved in diaphragm development, including transcription factors that regulate gene expression and genes that provide instructions for making proteins involved in cell structure or cell migration during embryonic development [medquad_ans_0000222-3]. However, in more than 80 percent of cases, no known genetic syndrome or chromosomal abnormality is identified, and the cause remains unknown [medquad_ans_0000222-3].

**Supported citations**: `medquad_ans_0000222-3`
**Unsupported citations**: _none_

### Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0003306-3` (score=0.7000) - What causes Jones syndrome? The exact, underlying genetic cause of Jones syndrome is not yet known.
2. `medquad_ans_0002717-3` (score=0.4142) - What causes Hallermann-Streiff syndrome? The genetic cause of Hallerman-Streiff syndrome has not been identified. It reportedly typically occurs randomly for unknown reasons (sporadically), most likely due to a new sp...
3. `medquad_ans_0006414-3` (score=0.3853) - What causes Weill-Marchesani syndrome? Weill-Marchesani syndrome is usually caused by mutations in the ADAMTS10 gene. Two families have been found with mutations in different genes, one with a mutation in FBN1 and one...
4. `medquad_ans_0004534-3` (score=0.1361) - What causes Omenn syndrome? Omenn syndrome is a genetically heterogeneous condition (meaning that it may be caused by a number of different genes). While most cases are attributed to mutations in the RAG genes (RAG-1 ...
5. `medquad_ans_0002533-3` (score=0.1361) - What causes Gerstmann-Straussler-Scheinker disease? Gerstmann-Straussler-Scheinker disease (GSS) is usually caused by certain changes (mutations) in the PRNP gene. PRNP encodes a protein called prion protein. Although...

**Answer**:

Omenn syndrome is a genetically heterogeneous condition that may be caused by a number of different genes. Most cases are attributed to mutations in the RAG genes (RAG-1 and RAG2 genes mapped to chromosome band 11p13), but recent reports describe Omenn syndrome in the absence of RAG mutations. Mutations in ARTEMIS, ADA, ILRA2, ILRA7, CHD7, and DNA ligase 4 have also been described in the medical literature. Some cases are associated with 22q11 microdeletion syndrome [medquad_ans_0004534-3].

**Supported citations**: `medquad_ans_0004534-3`
**Unsupported citations**: _none_

### Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000097-1` (score=0.7000) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...
2. `medquad_ans_0000097-5` (score=0.6797) - These resources address the diagnosis or management of ARCA1: - Gene Review: Gene Review: SYNE1-Related Autosomal Recessive Cerebellar Ataxia - Genetic Testing Registry: Spinocerebellar ataxia, autosomal recessive 8 -...
3. `medquad_ans_0005793-1` (score=0.6108) - What are the signs and symptoms of Spinocerebellar ataxia autosomal recessive 8? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia autosomal recessive 8. If the ...
4. `medquad_ans_0000495-1__dup1` (score=0.6031) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...
5. `medquad_ans_0000078-2` (score=0.5835) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive movement problems due to a loss (atrophy) of nerve cells in the cerebellum, the part of the brain that coordinates movement. Signs and symptoms first appear in early to mid-adulthood and include impaired speech (dysarthria), problems with coordination and balance (ataxia), or both. Other features may include difficulty with movements that involve judging distance or scale (dysmetria), abnormal eye movements (nystagmus), and problems following the movements of objects with the eyes. The movement problems are slowly progressive and may eventually result in the need for a cane, walker, or wheelchair [medquad_ans_0000097-1].

**Supported citations**: `medquad_ans_0000097-1`
**Unsupported citations**: _none_

### Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Scores**: context_relevance=0.750, groundedness=0.897, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000049-2__dup1` (score=0.7000) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
2. `medquad_ans_0000050-2__dup1` (score=0.6991) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
3. `medquad_ans_0000123-1__dup2` (score=0.6801) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Symptoms may include - A limp or paralyzed...
4. `medquad_ans_0000049-1__dup1` (score=0.6740) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Erb-Duchenne (Erb's) palsy refers to paral...
5. `medquad_ans_0000050-4__dup1` (score=0.6540) - The NINDS conducts and supports research on injuries to the nervous system such as brachial plexus injuries. Much of this research is aimed at finding ways to prevent and treat these disorders.

**Answer**:

Treatment for brachial plexus injuries includes physical therapy and, in some cases, surgery [medquad_ans_0000049-2\_\_dup1][medquad_ans_0000050-2__dup1][medquad_ans_0000123-1__dup2].

**Supported citations**: `medquad_ans_0000049-2__dup1`, `medquad_ans_0000050-2__dup1`, `medquad_ans_0000123-1__dup2`
**Unsupported citations**: _none_

### Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Scores**: context_relevance=0.833, groundedness=0.920, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000216-4__dup1` (score=0.7000) - When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis. Hydronephrosis is the most common problem found during prenatal ...
2. `medquad_ans_0000216-7` (score=0.6921) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
3. `medquad_ans_0000216-8` (score=0.6909) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PU...
4. `medquad_ans_0000154-7` (score=0.6711) - The complications of kidney dysplasia can include - hydronephrosis of the working kidney. A baby with kidney dysplasia in only one kidney might have other urinary tract defects. When other defects in the urinary tract...
5. `medquad_ans_0000071-11` (score=0.6582) - Although everyone has some risk for UTIs, some people are more likely to get UTIs than others. These include people who have - spinal cord injuries or other nerve damage around the bladder - a blockage in the urinary ...

**Answer**:

Urine blockage in newborns can be caused by various defects in the urinary tract, including vesicoureteral reflux (VUR), ureteropelvic junction (UPJ) obstruction, bladder outlet obstruction (BOO) such as posterior urethral valves (PUV), and ureterocele. Additionally, genetic conditions affecting multiple body systems, such as prune belly syndrome (PBS), esophageal atresia (EA), and congenital heart defects, can contribute to urine blockage. Urine blockage may also result from spina bifida and other birth defects affecting the spinal cord [medquad_ans_0000216-4\_\_dup1][medquad_ans_0000216-8].

**Supported citations**: `medquad_ans_0000216-4__dup1`, `medquad_ans_0000216-8`
**Unsupported citations**: _none_

### Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Scores**: context_relevance=0.778, groundedness=0.749, answer_relevance=0.689

**Top docs**:

1. `medquad_ans_0001035-1__dup1` (score=0.7000) - Centronuclear myopathy refers to a group of rare, inherited conditions that affect the muscles. There are three main forms of the condition that are differentiated by their pattern of inheritance: X-linked Myotubular ...
2. `medquad_ans_0006500-1` (score=0.5610) - X-linked scapuloperoneal myopathy is an inherited muscular dystrophy characterized by weakness and wasting of the muscles in the lower legs and the area of the shoulder blades. In some individuals, facial muscles may ...
3. `medquad_ans_0000526-1` (score=0.1059) - Intranuclear rod myopathy is a disorder that primarily affects skeletal muscles, which are muscles that the body uses for movement. People with intranuclear rod myopathy have severe muscle weakness (myopathy) and poor...
4. `medquad_ans_0003179-1` (score=0.1059) - Inclusion body myopathy 2, also known as hereditary inclusion body myopathy (HIBM), GNE-related myopathy, distal myopathy with rimmed vacuoles, and Nonaka myopathy, is an inherited condition that primarily affects the...
5. `medquad_ans_0000526-3` (score=0.0926) - Intranuclear rod myopathy is caused by a mutation in the ACTA1 gene. This gene provides instructions for making a protein called skeletal alpha ()-actin, which is part of the actin protein family. Actin proteins are i...

**Answer**:

The provided context does not specify the number of people affected by intranuclear rod myopathy. [medquad_ans_0000526-1] describes the condition's signs, symptoms, and prognosis but does not include population prevalence or affected人数. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000526-1`
**Unsupported citations**: _none_

### Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000036-11` (score=0.7000) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. There are two main types of high blood pressure: primary and secondary high blood pressure...
2. `medquad_ans_0000036-1__dup3` (score=0.6596) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. What Is Blood Pressure? Blood pressure is the force of blood pushing against the walls of ...
3. `medquad_ans_0000071-1__dup2` (score=0.6578) - Espaol High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. Measuring Blood Pressure Blood pressure is the force of blood pushing against the w...
4. `medquad_ans_0000058-3__dup4` (score=0.6497) - Blood pressure is the force of blood flow inside your blood vessels. Blood pressure is written with two numbers separated by a slash. For example, a blood pressure result of 130/80 is said as 130 over 80. The first nu...
5. `medquad_ans_0000471-1__dup1` (score=0.6027) - Blood pressure is the force of your blood pushing against the walls of your arteries. Each time your heart beats, it pumps blood into the arteries. Your blood pressure is highest when your heart beats, pumping the blo...

**Answer**:

High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. It is defined as the force of blood pushing against the walls of the blood vessels as the heart pumps blood. If blood pressure rises and stays high over time, it is called high blood pressure. High blood pressure is dangerous because it makes the heart work too hard and the high force of blood flow can harm arteries and organs such as the heart, kidneys, brain, and eyes [medquad_ans_0000036-1__dup3]. There are two main types: primary (or essential) high blood pressure, which is the most common type and develops over years as a person ages, and secondary high blood pressure, which is caused by another medical condition or use of certain medicines and usually resolves after the cause is treated or removed [medquad_ans_0000036-11].

**Supported citations**: `medquad_ans_0000036-1__dup3`, `medquad_ans_0000036-11`
**Unsupported citations**: _none_

### Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Scores**: context_relevance=0.750, groundedness=0.052, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000570-4` (score=0.7000) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
2. `medquad_ans_0006488-4` (score=0.6779) - How is X-linked adrenal hypoplasia congenita inherited? X-linked adrenal hypoplasia congenita is inherited in an X-linked recessive pattern. A condition is considered X-linked if the mutated gene that causes the disor...
3. `medquad_ans_0001065-4` (score=0.6694) - This condition is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In females (who have two X chromosomes), a mutation ...
4. `medquad_ans_0006502-4` (score=0.6621) - How is X-linked hypophosphatemia inherited? X-linked hypophosphatemia (XLH) is caused by mutations in the PHEX gene, and is inherited in an X-linked dominant manner. This means that the gene responsible for the condit...
5. `medquad_ans_0006488-1` (score=0.3874) - X-linked adrenal hypoplasia congenita is an inherited disorder that mainly affects males. It involves many hormone-producing (endocrine) tissues in the body, particularly a pair of small glands on top of each kidney c...

**Answer**:

The provided context does not contain information about the inheritance pattern of X-linked dystonia-parkinsonism. Therefore, it is not possible to determine based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.533, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000120-1` (score=0.7000) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process keto...
2. `medquad_ans_0000120-2` (score=0.6903) - Beta-ketothiolase deficiency appears to be very rare. It is estimated to affect fewer than 1 in 1 million newborns.
3. `medquad_ans_0000702-1__dup1` (score=0.6802) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ket...
4. `medquad_ans_0000120-5` (score=0.6426) - These resources address the diagnosis or management of beta-ketothiolase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of acetyl-CoA acetyltransferase These resources from MedlinePlus offer in...
5. `medquad_ans_0000120-3` (score=0.4979) - Mutations in the ACAT1 gene cause beta-ketothiolase deficiency. This gene provides instructions for making an enzyme that is found in the energy-producing centers within cells (mitochondria). This enzyme plays an esse...

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. Signs and symptoms typically appear between the ages of 6 months and 24 months and include episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, known as ketoacidotic attacks, may lead to coma and are frequently triggered by infections, periods without food (fasting), or increased intake of protein-rich foods. The condition is inherited in an autosomal recessive fashion and is caused by mutations in the ACAT1 gene, which provides instructions for making an enzyme found in the energy-producing centers within cells (mitochondria). This enzyme plays an essential role in breaking down proteins and fats from the diet

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 285 (`0000006_5-7`)

**Question**: What are the stages of Childhood Brain Stem Glioma ?

**Scores**: context_relevance=0.778, groundedness=0.771, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000006_5-1` (score=0.7000) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - T...
2. `medquad_ans_0000006_5-7` (score=0.6692) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the...
3. `medquad_ans_0000006_5-4` (score=0.6678) - The signs and symptoms of brain stem glioma are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain. - The size of the tumor and whether it has spread all thro...
4. `medquad_ans_0000006_3-1` (score=0.6004) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
5. `medquad_ans_0002574-1` (score=0.5424) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...

**Answer**:

The provided context does not mention any stages of childhood brain stem glioma. Therefore, it is not possible to determine the stages based on the given information. [medquad_ans_0000006_5-1]

**Supported citations**: `medquad_ans_0000006_5-1`
**Unsupported citations**: _none_

### Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.528, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000266-1__dup2` (score=0.7000) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal c...
2. `medquad_ans_0000266-3__dup1` (score=0.6691) - With treatment, individuals with tethered spinal cord syndrome have a normal life expectancy. However, some neurological and motor impairments may not be fully correctable. Surgery soon after symptoms emerge appears t...
3. `medquad_ans_0004257-2` (score=0.6248) - What are the signs and symptoms of myelomeningocele? A baby born with a myelomeningocele may have a sac sticking out of the mid to lower back that the doctor cannot see through when shining a light behind it. Symptoms...
4. `medquad_ans_0000061-1__dup1` (score=0.5971) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to a...
5. `medquad_ans_0000257-1__dup2` (score=0.5064) - Spinal cord infarction is a stroke either within the spinal cord or the arteries that supply it. It is caused by arteriosclerosis or a thickening or closing of the major arteries to the spinal cord. Frequently spinal ...

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course. Symptoms in children may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is often linked to improper growth of the neural tube during fetal development and is closely associated with spina bifida. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. Delayed symptom presentation is related to chronic strain on the spinal cord, which may be exacerbated

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.024, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000058-1__dup1` (score=0.7000) - Carpal tunnel syndrome (CTS) occurs when the median nerve, which runs from the forearm into the palm of the hand, becomes pressed or squeezed at the wrist. The carpal tunnel is a narrow, rigid passageway of ligament a...
2. `medquad_ans_0000155-1__dup3` (score=0.6903) - You're working at your desk, trying to ignore the tingling or numbness you've had for some time in your hand and wrist. Suddenly, a sharp, piercing pain shoots through the wrist and up your arm. Just a passing cramp? ...
3. `medquad_ans_0000419-1__dup1` (score=0.6345) - No matter how old you are or what you do for a living, you are always using your hands. When there is something wrong with them, you may not be able to do your regular activities. Hand problems include - Carpal tunnel...
4. `medquad_ans_0000469-1` (score=0.6168) - Hereditary neuropathy with liability to pressure palsies is a disorder that affects peripheral nerves. These nerves connect the brain and spinal cord to muscles as well as sensory cells that detect touch, pain, and te...
5. `medquad_ans_0000136-5__dup2` (score=0.6157) - The symptoms of dialysis-related amyloidosis may include - pain, stiffness, and fluid in the joints. - abnormal, fluid-containing sacs, called cysts, in some bones. - carpal tunnel syndrome, caused by unusual buildup ...

**Answer**:

The provided context does not contain information about the outlook or prognosis for carpal tunnel syndrome. Therefore, it is not possible to determine the outlook based on the given sources. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.111, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004155-5` (score=0.7000) - What treatment is available for mucopolysaccharidosis I (MPS I)? The two main treatments for MPS I are enzyme replacement therapy (ERT) and bone marrow transplant. Both of these treatments work by replacing the missin...
2. `medquad_ans_0003845-5` (score=0.5290) - How might McCune Albright syndrome be treated? Although there is no cure for McCune Albright syndrome (MAS), drug treatments may help some of the endocrine symptoms, and surgery can help repair some of the bone proble...
3. `medquad_ans_0005077-3` (score=0.1620) - How is progressive bulbar palsy treated? Treatments aim to help people cope with the symptoms of progressive bulbar palsy, such as feeding tubes, devices to help with talking, and medicines to treat muscle spasms, wea...
4. `medquad_ans_0000296-3` (score=0.1620) - Mutations in the LRP2 gene cause Donnai-Barrow syndrome. The LRP2 gene provides instructions for making a protein called megalin, which functions as a receptor. Receptor proteins have specific sites into which certain...
5. `medquad_ans_0000296-1` (score=0.1395) - Donnai-Barrow syndrome is an inherited disorder that affects many parts of the body. This disorder is characterized by unusual facial features, including prominent, wide-set eyes with outer corners that point downward...

**Answer**:

The provided context does not contain information about the treatments for Donnai-Barrow syndrome. Therefore, a specific answer cannot be given based on the available content. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 289 (`0000811-1`)

**Question**: What is (are) porphyria ?

**Scores**: context_relevance=0.750, groundedness=0.990, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0001134-1` (score=0.7000) - Chester porphyria is a unique type of porphyria with the signs and symptoms of acute intermittent porphyria (AIP) and the biochemical defects of both AIP and variegate porphyria (VP). Chester porphyria does not confor...
2. `medquad_ans_0000189-4__dup2` (score=0.6793) - The exact rates of porphyria are unknown and vary around the world. For example, porphyria cutanea tarda is most common in the United States, and variegate porphyria is most common in South America.1
3. `medquad_ans_0000811-2` (score=0.6407) - The exact prevalence of porphyria is unknown, but it probably ranges from 1 in 500 to 1 in 50,000 people worldwide. Overall, porphyria cutanea tarda is the most common type of porphyria. For some forms of porphyria, t...
4. `medquad_ans_0000189-10` (score=0.5968) - - Porphyrias are rare disorders that affect mainly the skin or nervous system and may cause abdominal pain. - Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosyntheti...
5. `medquad_ans_0000811-1` (score=0.5904) - Porphyria is a group of disorders caused by abnormalities in the chemical steps that lead to heme production. Heme is a vital molecule for all of the body's organs, although it is most abundant in the blood, bone marr...

**Answer**:

Porphyrias are rare disorders that affect mainly the skin or nervous system and may cause abdominal pain. Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosynthetic pathway. Most porphyrias result from inheriting an abnormal gene, also called a gene mutation, from one parent. Porphyria cutanea tarda is usually an acquired disorder, meaning factors other than genes cause the enzyme deficiency [medquad_ans_0000189-10].

**Supported citations**: `medquad_ans_0000189-10`
**Unsupported citations**: _none_

### Example 290 (`0000585-1__dup1`)

**Question**: What is (are) Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy ?

**Scores**: context_relevance=1.000, groundedness=0.335, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000585-1__dup1` (score=0.7000) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
2. `medquad_ans_0004557-1` (score=0.5046) - Optic atrophy 1 is a condition that mainly affects vision, but may include other features. Vision loss typically begins within the first decade of life; severity varies widely among affected people (from nearly normal...
3. `medquad_ans_0000261-1__dup1` (score=0.3474) - Alport syndrome is a genetic condition characterized by kidney disease, hearing loss, and eye abnormalities. Most affected individuals experience progressive loss of kidney function, usually resulting in end-stage kid...
4. `medquad_ans_0000585-2__dup1` (score=0.0697) - What are the signs and symptoms of Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal recessive opt...
5. `medquad_ans_0005818-1` (score=0.0601) - Spondylocarpotarsal synostosis (SCT) syndrome is an inherited syndrome characterized by disproportionate short stature, abnormalities of the vertebrae in the spine, scoliosis and lordosis, carpal and tarsal fusion (fu...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. Autosomal recessive inheritance has been suggested [medquad_ans_0000585-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000585-1`

### Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.098, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0005069-3` (score=0.7000) - What genes are related to Hutchinson-Gilford progeria syndrome? Mutations in the LMNA gene cause Hutchinson-Gilford progeria syndrome. The LMNA gene provides instructions for making a protein called lamin A. This prot...
2. `medquad_ans_0000167-2__dup3` (score=0.2118) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up...
3. `medquad_ans_0000794-1` (score=0.2118) - Phosphoglycerate kinase deficiency is a genetic disorder that affects the body's ability to break down the simple sugar glucose, which is the primary energy source for most cells. Researchers have described two major ...
4. `medquad_ans_0004167-3` (score=0.1853) - What causes Muir-Torre syndrome? Muir-Torre syndrome is a subtype of Lynch syndrome and may be caused by changes (mutations) in either the MLH1, MSH2, or MSH6 gene. These genes give the body instructions to make prote...
5. `medquad_ans_0000794-5` (score=0.1853) - These resources address the diagnosis or management of phosphoglycerate kinase deficiency: - Children Living with Inherited Metabolic Diseases (CLIMB) (UK): Phosphoglycerate Kinase Deficiency - Genetic Testing Registr...

**Answer**:

The provided context does not contain information about the genetic changes related to phosphoglycerate kinase deficiency. Therefore, the answer cannot be determined from the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Scores**: context_relevance=0.800, groundedness=0.785, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000109-4__dup2` (score=0.7000) - Pulmonary hypertension (PH) begins with inflammation and changes in the cells that line your pulmonary arteries. Other factors also can affect the pulmonary arteries and cause PH. For example, the condition may develo...
2. `medquad_ans_0000673-4` (score=0.1184) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
3. `medquad_ans_0000850-5` (score=0.1184) - These resources address the diagnosis or management of pulmonary veno-occlusive disease: - Genetic Testing Registry: Pulmonary veno-occlusive disease These resources from MedlinePlus offer information about the diagno...
4. `medquad_ans_0000453-1` (score=0.1036) - Hepatic veno-occlusive disease with immunodeficiency (also called VODI) is a hereditary disorder of the liver and immune system. Its signs and symptoms appear after the first few months of life. Hepatic veno-occlusive...
5. `medquad_ans_0004784-4` (score=0.1036) - How is Pelizaeus-Merzbacher disease inherited?

**Answer**:

The provided context does not contain information about the specific genetic changes related to pulmonary veno-occlusive disease. While it mentions that hepatic veno-occlusive disease with immunodeficiency (VODI) is a hereditary disorder and that children with VODI are prone to recurrent infections, no details about the genetic mutations or changes are given [medquad_ans_0000850-5]. Therefore, based on the available context, the genetic changes related to pulmonary veno-occlusive disease cannot be determined.

**Supported citations**: `medquad_ans_0000850-5`
**Unsupported citations**: _none_

### Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Scores**: context_relevance=1.000, groundedness=0.957, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0001006-3` (score=0.7000) - Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. In the liver, enzymes break down tyrosine in a five step process, resulting in molecules that are either excreted by t...
2. `medquad_ans_0003520-3` (score=0.6727) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. Th...
3. `medquad_ans_0004086-3` (score=0.3249) - What causes mitochondrial genetic disorders? Mitochondrial genetic disorders can be caused by changes (mutations) in either the mitochondrial DNA or nuclear DNA that lead to dysfunction of the mitochondria. Most DNA (...
4. `medquad_ans_0001006-1` (score=0.1317) - Tyrosinemia is a genetic disorder characterized by disruptions in the multistep process that breaks down the amino acid tyrosine, a building block of most proteins. If untreated, tyrosine and its byproducts build up i...
5. `medquad_ans_0001006-2` (score=0.1152) - Worldwide, tyrosinemia type I affects about 1 in 100,000 individuals. This type is more common in Norway where 1 in 60,000 to 74,000 individuals are affected. Tyrosinemia type I is even more common in Quebec, Canada w...

**Answer**:

Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively [medquad_ans_0001006-3]. These mutations result in a decrease in the activity of specific enzymes involved in the breakdown of tyrosine, leading to the accumulation of toxic levels of tyrosine and its byproducts [medquad_ans_0001006-3]. The FAH gene provides instructions for the fumarylacetoacetate hydrolase enzyme, which is responsible for the final step in tyrosine breakdown; the TAT gene encodes tyrosine aminotransferase, involved in the first step; and the HPD gene encodes 4-hydroxyphenylpyruvate dioxygenase, responsible for the second step [medquad_ans_0001006-3

**Supported citations**: `medquad_ans_0001006-3`
**Unsupported citations**: _none_

### Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Scores**: context_relevance=1.000, groundedness=0.900, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000436-5` (score=0.7000) - These resources address the diagnosis or management of Graves disease: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Graves disease 2 - Genetic Testing Registry: Graves disease 3 -...
2. `medquad_ans_0000738-3__dup1` (score=0.4506) - What treatments are available for birdshot chorioretinopathy? Unfortunately, there is currently no cure for birdshot chorioretinopathy. Because this condition is rare, there are no established guidelines for treatment...
3. `medquad_ans_0002675-2` (score=0.1053) - What are the signs and symptoms of Graves' disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Graves' disease. If the information is available, the table below includes how oft...
4. `medquad_ans_0000004-6__dup2` (score=0.0921) - People with Graves disease have three treatment options: radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves disease in the United States. Graves disease ...
5. `medquad_ans_0000108-2__dup1` (score=0.0921) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The treatments for Graves' disease include therapies aimed at managing symptoms and controlling thyroid function. Specific treatment options include antithyroid medications, beta-blockers to manage symptoms such as tremors and rapid heart rate, radioactive iodine therapy to reduce thyroid function, and in some cases, surgery to remove the thyroid gland. These treatment options are supported by resources such as the Graves' Disease & Thyroid Foundation, which provides information on treatment options, and the Thyroid Disease Manager, which covers the diagnosis and treatment of Graves' disease [medquad_ans_0000436-5]. Additionally, the National Institute of Diabetes and Digestive and Kidney Diseases provides information on thyroid function tests, which may be used in the management of the disease [medquad_ans_0000436-5].

**Supported citations**: `medquad_ans_0000436-5`
**Unsupported citations**: _none_

### Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Scores**: context_relevance=0.800, groundedness=0.488, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000039-1__dup2` (score=0.7000) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, ...
2. `medquad_ans_0000039-3__dup1` (score=0.6718) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten dise...
3. `medquad_ans_0000545-1` (score=0.6450) - Juvenile Batten disease is an inherited disorder that primarily affects the nervous system. After a few years of normal development, children with this condition develop progressive vision loss, intellectual and motor...
4. `medquad_ans_0000545-2` (score=0.6360) - Juvenile Batten disease is the most common type of NCL, but its exact prevalence is unknown. Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, wher...
5. `medquad_ans_0000039-2__dup1` (score=0.6221) - As yet, no specific treatment is known that can halt or reverse the symptoms of Batten disease. However, seizures can sometimes be reduced or controlled with anticonvulsant drugs, and other medical problems can be tre...

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. It is characterized by a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. The disease often presents with early signs such as personality and behavior changes, slow learning, clumsiness, or stumbling. Vision loss is frequently an early symptom and may lead to suspicion during an eye exam. Over time, affected children experience cognitive impairment, worsening seizures, and progressive loss of sight and motor skills, eventually becoming blind, bedridden, and demented. Juvenile Batten disease, a form of the disorder, typically manifests between ages 4 and 8 with progressive vision loss, intellectual and motor disability, speech difficulties, and seizures. It is part of a group of disorders known as neuronal ceroid lipofuscinoses (NCLs), which affect the nervous system and cause progressive issues with vision, movement

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002061-1` (score=0.7000) - Ehlers-Danlos syndrome progeroid type is a genetic disorder of the connective tissue, which is the material between the cells of the body that gives tissues form and strength. The disorder primarily affects the skin, ...
2. `medquad_ans_0000318-1__dup1` (score=0.7000) - Ehlers-Danlos syndrome (EDS) is a group of inherited disorders that weaken connective tissues. Connective tissues are proteins that support skin, bones, blood vessels, and other organs. EDS usually affects your skin, ...
3. `medquad_ans_0002061-3` (score=0.6122) - What causes Ehlers-Danlos syndrome progeroid type? Ehlers-Danlos syndrome progeroid type is caused by changes (mutations) in both of an individual's copies of the B4GALT7 gene, which is located on chromosome 5. This g...
4. `medquad_ans_0005116-1` (score=0.6122) - Proteus-like syndrome describes people who do not meet the diagnostic criteria for Proteus syndrome but who share many of the characteristic signs and symptoms associated with the condition. Affected people may experi...
5. `medquad_ans_0000313-5` (score=0.5272) - These resources address the diagnosis or management of Ehlers-Danlos syndrome: - Gene Review: Gene Review: Ehlers-Danlos Syndrome, Classic Type - Gene Review: Gene Review: Ehlers-Danlos Syndrome, Hypermobility Type - ...

**Answer**:

Ehlers-Danlos syndrome progeroid type is a genetic disorder of the connective tissue that primarily affects the skin, hair, and skeletal system. Symptoms usually appear by childhood or adolescence and include unusually flexible joints, loose elastic skin, and easy scarring. Unique features of this type include sparse scalp hair and eyebrows, and loose elastic skin on the face, which causes affected individuals to look older than their age. Additional symptoms may include bone weakness, weak muscle tone, mild intellectual disability, and delayed growth in children. This type is caused by mutations in both copies of the B4GALT7 gene, located on chromosome 5, which leads to reduced production of an enzyme involved in collagen formation, resulting in improperly formed collagen and weak connective tissue [medquad_ans_0002061-1][medquad_ans_0002061-3]. It is inherited in an autosomal recessive pattern

**Supported citations**: `medquad_ans_0002061-1`, `medquad_ans_0002061-3`
**Unsupported citations**: _none_

### Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=0.900, groundedness=0.967, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000416-3` (score=0.7000) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
2. `medquad_ans_0000416-2` (score=0.6723) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
3. `medquad_ans_0002601-1` (score=0.6619) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...
4. `medquad_ans_0000416-5` (score=0.6619) - These resources address the diagnosis or management of glutamate formiminotransferase deficiency: - Baby's First Test - Genetic Testing Registry: Glutamate formiminotransferase deficiency These resources from MedlineP...
5. `medquad_ans_0000416-1` (score=0.6538) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with ...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the metabolism of the amino acid histidine and in the production of folate [medquad_ans_0000416-3]. The FTCD gene provides instructions for making this enzyme [medquad_ans_0000416-3].

**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

### Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Scores**: context_relevance=1.000, groundedness=0.941, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0001169-1` (score=0.7000) - Chondrodysplasia punctata 1, X-linked recessive (CDPX1) is a genetic disorder present from birth that affects bone and cartilage development. On x-ray, affected infants have characteristic spots at the ends of their b...
2. `medquad_ans_0001047-1` (score=0.6710) - X-linked chondrodysplasia punctata 1 is a disorder of cartilage and bone development that occurs almost exclusively in males. Chondrodysplasia punctata is an abnormality that appears on x-rays as spots (stippling) nea...
3. `medquad_ans_0001169-2` (score=0.6524) - What are the signs and symptoms of Chondrodysplasia punctata 1, X-linked recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 1, X-linked recessive. I...
4. `medquad_ans_0001048-2` (score=0.6360) - X-linked chondrodysplasia punctata 2 has been estimated to affect fewer than 1 in 400,000 newborns. However, the disorder may actually be more common than this estimate because it is likely underdiagnosed, particularl...
5. `medquad_ans_0001047-2` (score=0.6322) - The prevalence of X-linked chondrodysplasia punctata 1 is unknown. Several dozen affected males have been reported in the scientific literature.

**Answer**:

The symptoms of Chondrodysplasia punctata 1, X-linked recessive include characteristic spots (stippled epiphyses) on x-ray at the ends of bones, which typically disappear between ages 2 and 3. Additional common features are shortened fingers, a flat nose, breathing abnormalities, hearing loss, abnormalities of the spinal bones in the neck, and delayed intellectual development. Some individuals may experience serious or life-threatening complications such as airway stenosis leading to restricted breathing, spinal cord compression causing pain, numbness, and weakness, delayed development, vision abnormalities, and heart defects. Most affected individuals have a normal lifespan, although complications can be life-threatening [medquad_ans_0001169-1][medquad_ans_0001047-1].

**Supported citations**: `medquad_ans_0001169-1`, `medquad_ans_0001047-1`
**Unsupported citations**: _none_

### Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Scores**: context_relevance=0.857, groundedness=0.935, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000132-11` (score=0.7000) - - Whipple disease is a rare bacterial infection that primarily affects the small intestine. Left untreated, Whipple disease gets worse and is usually life threatening. - Bacteria called Tropheryma whipplei (T. whipple...
2. `medquad_ans_0000132-5__dup1` (score=0.6979) - Signs and symptoms of Whipple disease can vary widely from person to person. The most common symptoms of Whipple disease are - diarrhea - weight loss caused by malabsorption A person may not have diarrhea. Instead, ot...
3. `medquad_ans_0000132-1__dup3` (score=0.6585) - Whipple disease is a rare bacterial infection that primarily affects the small intestine. The infection may spread to any organ in the body; however, it more commonly affects the - joints - central nervous system, whi...
4. `medquad_ans_0000132-6` (score=0.5799) - People with Whipple disease may have complications caused by malnutrition, which is due to damaged villi in the small intestine. As a result of delayed diagnosis or treatment, people may experience the following compl...
5. `medquad_ans_0000132-7` (score=0.5729) - A health care provider may use several tests and exams to diagnose Whipple disease, including the following: - medical and family history - physical exam - blood tests - upper GI endoscopy and enteroscopy A patient ma...

**Answer**:

The treatments for Whipple disease involve antibiotics. The healthcare provider prescribes antibiotics to destroy the Tropheryma whipplei bacteria and treat the disease. Initially, intravenous (IV) antibiotics are prescribed for the first 2 weeks of treatment. Most patients feel relief from symptoms within the first week or two. After completing the IV antibiotics, the healthcare provider will prescribe long-term oral antibiotics. Most patients have good outcomes with an early diagnosis and complete treatment [medquad_ans_0000132-11].

**Supported citations**: `medquad_ans_0000132-11`
**Unsupported citations**: _none_

### Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Scores**: context_relevance=0.857, groundedness=0.875, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000676-4__dup1` (score=0.7000) - How might Bell's palsy be treated?
2. `medquad_ans_0000042-3__dup2` (score=0.6976) - The prognosis for individuals with Bell's palsy is generally very good. The extent of nerve damage determines the extent of recovery. With or without treatment, most individuals begin to get better within 2 weeks afte...
3. `medquad_ans_0000676-2__dup1` (score=0.6883) - What are the symptoms of Bell's palsy?
4. `medquad_ans_0000086-1__dup2` (score=0.6808) - Bell's palsy is the most common cause of facial paralysis. It usually affects just one side of the face. Symptoms appear suddenly and are at their worst about 48 hours after they start. They can range from mild to sev...
5. `medquad_ans_0000676-1__dup1` (score=0.6689) - Bell's palsy is a form of temporary facial paralysis which results from damage or trauma to one of the facial nerves. This disorder is characterized by the sudden onset of facial paralysis that often affects just one ...

**Answer**:

Treatments for Bell's palsy may include steroids such as prednisone to reduce inflammation and swelling of the facial nerve, antiviral medications like acyclovir to combat potential viral infections, and pain relievers such as aspirin, acetaminophen, or ibuprofen to manage pain. Additionally, physical therapy, facial massage, and acupuncture have been used as supportive treatments [medquad_ans_0000676-1__dup1].

**Supported citations**: `medquad_ans_0000676-1__dup1`
**Unsupported citations**: _none_
