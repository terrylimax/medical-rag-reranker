# Generation Evaluation

## Summary

- `avg_answer_chars`: 568.0000
- `avg_answer_relevance`: 0.7363
- `avg_citation_presence_rate`: 0.8600
- `avg_context_relevance`: 0.7986
- `avg_empty_answer`: 0.0000
- `avg_end_to_end_latency_ms`: 8237.5309
- `avg_generation_latency_ms`: 6078.3478
- `avg_groundedness`: 0.7339
- `avg_insufficient_context`: 0.0367
- `avg_num_retrieved_docs`: 5.0000
- `avg_rerank_latency_ms`: 2090.1370
- `avg_retrieval_latency_ms`: 68.8993
- `avg_supported_citation_rate`: 0.7133
- `avg_unsupported_citation_rate`: 0.1467
- `generation_remote_concurrency`: 4.0000
- `num_examples`: 300
- `reranker_enabled_rate`: 1.0000

## Examples

### Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.974, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006510-1` (score=0.9994) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the informatio...
2. `medquad_ans_0000995-2` (score=0.0001) - This condition occurs in about 1 in 1,000 newborn girls. Five to 10 girls with triple X syndrome are born in the United States each day.
3. `medquad_ans_0000307-4` (score=0.0001) - This condition is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each cell. In males...
4. `medquad_ans_0001046-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
5. `medquad_ans_0000158-4` (score=0.0001) - This condition is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each cell. In femal...

**Answer**:

The symptoms of X-linked lymphoproliferative syndrome 1 include cellular immunodeficiency (90%), decreased antibody level in blood (50%), hepatomegaly (50%), lymphadenopathy (50%), lymphoma (50%), splenomegaly (50%), anemia (7.5%), encephalitis, fulminant hepatitis, hepatic encephalopathy, IgG deficiency, immunodeficiency, increased IgM level, meningitis, pancytopenia, recurrent pharyngitis, reduced natural killer cell activity, thrombocytopenia, and X-linked recessive inheritance [medquad_ans_0006510-1].

**Supported citations**: `medquad_ans_0006510-1`
**Unsupported citations**: _none_

### Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.955, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000312-4` (score=0.9879) - Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, which means one of the two copies of the gene is altered in each cell. Many people who have a mutation in this gene are not affected by the d...
2. `medquad_ans_0000859-4` (score=0.7779) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered ATP1A3 gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits a mutation fro...
3. `medquad_ans_0000309-5` (score=0.0001) - These resources address the diagnosis or management of dystonia 6: - Gene Review: Gene Review: Dystonia Overview - Genetic Testing Registry: Dystonia 6, torsion These resources from MedlinePlus offer information about...
4. `medquad_ans_0000112-1__dup1` (score=0.0001) - The dystonias are movement disorders in which sustained muscle contractions cause twisting and repetitive movements or abnormal postures. The movements, which are involuntary and sometimes painful, may affect a single...
5. `medquad_ans_0005768-2` (score=0.0001) - What are the signs and symptoms of Spinocerebellar ataxia 2? Early symptoms of spinocerebellar ataxia may include uncoordinated movement (ataxia) and leg cramps. Other symptoms may include tremor; decreased muscle ton...

**Answer**:

Yes, early-onset primary dystonia is inherited. Everyone who has been diagnosed with early-onset primary dystonia has inherited a TOR1A mutation from one parent, and the condition is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder [medquad_ans_0000312-4].

**Supported citations**: `medquad_ans_0000312-4`
**Unsupported citations**: _none_

### Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Scores**: context_relevance=0.714, groundedness=0.962, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000397-2__dup1` (score=0.9876) - The tapeworms that cause taeniasis (Taenia saginata, T. solium, and T. asiatica) are found worldwide. Eating raw or undercooked beef or pork is the primary risk factor for acquiring taeniasis. Persons who don't eat ra...
2. `medquad_ans_0000397-1__dup2` (score=0.0824) - Taeniasis in humans is a parasitic infection caused by the tapeworm species Taenia saginata (beef tapeworm), Taenia solium (pork tapeworm), and Taenia asiatica (Asian tapeworm). Humans can become infected with these t...
3. `medquad_ans_0000265-4__dup2` (score=0.0167) - Loiasis is an infection caused by the parasitic worm Loa loa.
4. `medquad_ans_0000432-1__dup3` (score=0.0054) - Whipworm (Trichuris trichiura) is an intestinal parasite of humans. The larvae and adult worms live in the intestine of humans and can cause intestinal disease. The name is derived from the worm’s distinctive whip-lik...
5. `medquad_ans_0000121-2__dup3` (score=0.0040) - Cysticercosis is an infection caused by the larvae of the parasite Taenia solium. This infection occurs after a person swallows tapeworm eggs. The larvae get into tissues such as muscle and brain, and form cysts there...

**Answer**:

People at risk for taeniasis include those who eat raw or undercooked beef or pork. Infections with Taenia saginata occur wherever contaminated raw beef is eaten, particularly in Eastern Europe, Russia, eastern Africa, and Latin America. Taenia solium infections are more prevalent in under-developed communities with poor sanitation and where people eat raw or undercooked pork, with higher rates seen in Latin America, Eastern Europe, sub-Saharan Africa, India, and Asia. Taeniasis due to T. solium is seen in the United States, typically among Latin American immigrants. Taenia asiatica is limited to Asia and is seen mostly in the Republic of Korea, China, Taiwan, Indonesia, and Thailand. Persons who do not eat raw or undercooked beef or pork are not likely to get taeniasis [medquad_ans_0000397-2__dup1].

**Supported citations**: `medquad_ans_0000397-2__dup1`
**Unsupported citations**: _none_

### Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.975, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000194-2__dup1` (score=0.9994) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
2. `medquad_ans_0000029-4` (score=0.0001) - Aicardi-Goutieres syndrome can have different inheritance patterns. In most cases it is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an i...
3. `medquad_ans_0000195-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000197-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
5. `medquad_ans_0000194-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy [medquad_ans_0000194-2__dup1].

**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

### Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003421-1` (score=0.9993) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0000172-3__dup1` (score=0.0057) - There is no cure for Klver-Bucy syndrome. The disorder is not life-threatening, but the patient can be difficult to manage. With treatment, symptoms may slowly decline.
3. `medquad_ans_0000920-2` (score=0.0001) - Smith-Magenis syndrome affects at least 1 in 25,000 individuals worldwide. Researchers believe that many people with this condition are not diagnosed, however, so the true prevalence may be closer to 1 in 15,000 indiv...
4. `medquad_ans_0000556-2` (score=0.0001) - KBG syndrome is a rare disorder that has been reported in around 60 individuals. For unknown reasons, males are affected more often than females. Doctors think the disorder is underdiagnosed because the signs and symp...
5. `medquad_ans_0000543-2` (score=0.0001) - Joubert syndrome is estimated to affect between 1 in 80,000 and 1 in 100,000 newborns. However, this estimate may be too low because Joubert syndrome has such a large range of possible features and is likely underdiag...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

### Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.962, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000335-4` (score=0.9960) - Factor V deficiency is inherited in an autosomal recessive pattern, which means both copies of the F5 gene in each cell have mutations. Individuals with a mutation in a single copy of the F5 gene have a reduced amount...
2. `medquad_ans_0000337-4` (score=0.9957) - When this condition is caused by mutations in the F10 gene, it is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an auto...
3. `medquad_ans_0000450-4` (score=0.9842) - Hemophilia A and hemophilia B are inherited in an X-linked recessive pattern. The genes associated with these conditions are located on the X chromosome, which is one of the two sex chromosomes. In males (who have onl...
4. `medquad_ans_0000139-2__dup2` (score=0.9745) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...
5. `medquad_ans_0006255-4` (score=0.0138) - Is plasminogen deficiency, type 1 inherited? If so, in what manner? Plasminogen deficiency, type 1 is inherited in an autosomal recessive fashion, which means that an individual must inherit two disease-causing mutate...

**Answer**:

Yes, hemophilia is inherited. Hemophilia A and hemophilia B are inherited in an X-linked recessive pattern, which means the genes associated with these conditions are located on the X chromosome [medquad_ans_0000450-4]. In males, who have only one X chromosome, one altered copy of the gene is sufficient to cause the condition. In females, who have two X chromosomes, a mutation would need to occur in both copies to cause the disorder, making it very rare for females to have hemophilia [medquad_ans_0000450-4]. Fathers cannot pass X-linked traits to their sons, and females who carry one altered copy are typically carriers with about half the usual amount of coagulation factor VIII or IX, which is generally sufficient for normal blood clotting [medquad_ans_0000450-4].

**Supported citations**: `medquad_ans_0000450-4`
**Unsupported citations**: _none_

### Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.970, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.9994) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often t...
2. `medquad_ans_0000045-2` (score=0.0043) - More than 900 people with Alstrm syndrome have been reported worldwide.
3. `medquad_ans_0000087-3__dup1` (score=0.0027) - The prognosis is dependent upon the specific syndrome, however, some of the syndromes are fatal if left untreated.
4. `medquad_ans_0000062-3__dup1` (score=0.0006) - Central pain syndrome is not a fatal disorder, but the syndrome causes disabling chronic pain and suffering among the majority of individuals who have it.
5. `medquad_ans_0000954-2` (score=0.0003) - Stickler syndrome affects an estimated 1 in 7,500 to 9,000 newborns. Type I is the most common form of the condition.

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
3. `medquad_ans_0001062-2` (score=0.0071) - This form of anemia is uncommon. However, researchers believe that it may not be as rare as they once thought. Increased awareness of the disease has led to more frequent diagnoses.
4. `medquad_ans_0000412-2` (score=0.0015) - GPI deficiency is a rare cause of hemolytic anemia; its prevalence is unknown. About 50 cases have been described in the medical literature.
5. `medquad_ans_0000347-2` (score=0.0007) - Familial erythrocytosis is a rare condition; its prevalence is unknown.

**Answer**:

Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes), which protect the body from infection; and platelets (thrombocytes), which are involved in blood clotting. PNH can occur at any age but is usually diagnosed in young adulthood and affects both sexes equally. The disorder is caused by acquired, rather than inherited, mutations in the PIGA gene and is not passed down to children of affected individuals. People with PNH experience sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses on the body, such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to hemolytic anemia and signs and symptoms such as fatigue,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.974, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000010-20` (score=0.8451) - Yes. In July 2008, the U.S. Preventive Services Task Force made its strongest ever recommendation for colorectal cancer screening: it suggested that all adults between ages 50 and 75 get screened, or tested, for the d...
2. `medquad_ans_0000116-10` (score=0.7744) - No. Unlike the other inflammatory bowel diseases, such as Crohns disease and ulcerative colitis, microscopic colitis does not increase a persons risk of getting colon cancer.
3. `medquad_ans_0000010-10` (score=0.6004) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the li...
4. `medquad_ans_0000010-17` (score=0.0222) - Parents, siblings, or children of a person who has had colorectal cancer are somewhat more likely to develop this type of cancer themselves. This is especially true if the relative had the cancer at a young age. If ma...
5. `medquad_ans_0000037_4-1` (score=0.0212) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...

**Answer**:

Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. The colon and the rectum are parts of the large intestine, which is part of the digestive system. Cancer that begins in the colon is called colon cancer, and cancer that begins in the rectum is called rectal cancer. Cancer that begins in either of these organs may also be called colorectal cancer [medquad_ans_0000010-10][medquad_ans_0000037_4-1]. Colorectal cancer is the second leading cause of death from cancer in the United States [medquad_ans_0000037_4-1].

**Supported citations**: `medquad_ans_0000010-10`, `medquad_ans_0000037_4-1`
**Unsupported citations**: _none_

### Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Scores**: context_relevance=0.833, groundedness=0.962, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003317-1` (score=0.9712) - Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz ...
2. `medquad_ans_0000544-1` (score=0.9670) - Junctional epidermolysis bullosa (JEB) is one of the major forms of epidermolysis bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. Blisters and skin erosions form in...
3. `medquad_ans_0002121-1` (score=0.0042) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main ty...
4. `medquad_ans_0002122-1` (score=0.0017) - Epidermolysis bullosa acquisita (EBA) is a rare autoimmune disorder that causes the skin to blister in response to minor injury. Common areas of blistering include the hands, feet, knees, elbows, and buttocks. It can ...
5. `medquad_ans_0000433-2` (score=0.0004) - Gorlin syndrome affects an estimated 1 in 31,000 people. While more than 1 million new cases of basal cell carcinoma are diagnosed each year in the United States, fewer than 1 percent of these skin cancers are related...

**Answer**:

Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz type and the Non-Herlitz type. The Herlitz type is very severe, with individuals often not surviving infancy, while the Non-Herlitz type includes several subtypes that cause mild to severe blistering present at birth or shortly thereafter. JEB is inherited in an autosomal recessive pattern and is caused by mutations in the LAMB3, COL17A1, or LAMC2, and LAMA3 genes. There is no cure for JEB; treatment is focused on managing blistering and preventing secondary infections [medquad_ans_0003317-1]. [medquad_ans_00005

**Supported citations**: `medquad_ans_0003317-1`
**Unsupported citations**: _none_

### Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Scores**: context_relevance=0.900, groundedness=0.973, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000668-3` (score=0.9991) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same ...
2. `medquad_ans_0000650-3` (score=0.0001) - Mutations in the MUT, MMAA, MMAB, MMADHC, and MCEE genes cause methylmalonic acidemia. The long term effects of methylmalonic acidemia depend on which gene is mutated and the severity of the mutation. About 60 percent...
3. `medquad_ans_0000416-3` (score=0.0001) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
4. `medquad_ans_0000326-4` (score=0.0001) - Some cases of erythromelalgia occur in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some of these instances, an affected person inherits ...
5. `medquad_ans_0000396-1__dup2` (score=0.0001) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...

**Answer**:

The genetic changes related to molybdenum cofactor deficiency are mutations in the MOCS1, MOCS2, or GPHN gene. Mutations in the MOCS1 gene cause type A, mutations in the MOCS2 gene cause type B, and mutations in the GPHN gene cause type C [medquad_ans_0000668-3]. These mutations reduce or eliminate the function of the associated protein, impairing molybdenum cofactor biosynthesis, which is essential for the function of metabolic enzymes that break down toxic substances in the body [medquad_ans_0000668-3].

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
5. `medquad_ans_0000065-2` (score=0.0004) - Arginase deficiency is a very rare disorder; it has been estimated to occur once in every 300,000 to 1,000,000 individuals.

**Answer**:

Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The nervous system is especially sensitive to the effects of excess ammonia. Ornithine translocase deficiency varies widely in its severity and age of onset. An infant with ornithine translocase deficiency may be lacking in energy (lethargic) or refuse to eat, or have poorly controlled breathing or body temperature. Some babies with this disorder may experience seizures or unusual body movements, or go into a coma. Episodes of illness may coincide with the introduction of high-protein formulas or solid foods into the diet. In most affected individuals, signs and symptoms of ornithine translocase deficiency do not appear until later in life. Later-onset forms of ornithine translocase deficiency are usually less severe than the

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Scores**: context_relevance=0.857, groundedness=0.885, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000586-1__dup2` (score=0.9970) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state ha...
2. `medquad_ans_0000362-1__dup2` (score=0.9954) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
3. `medquad_ans_0000590-1__dup2` (score=0.9948) - Summary : Part D is the name of Medicare's prescription drug coverage. It's insurance that helps people pay for prescription drugs. It is available to everyone who has Medicare. It provides protection if you pay high ...
4. `medquad_ans_0000431-1__dup1` (score=0.9945) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...
5. `medquad_ans_0000338-1__dup1` (score=0.9728) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the s...

**Answer**:

Yes, financial assistance is available for individuals who do not have health insurance or who need help with medical costs not covered by insurance. Certain government programs and nonprofit organizations can provide such assistance. Additionally, you can discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2]. Medicaid is a government health insurance program that helps low-income individuals pay their medical bills, and eligibility may depend on income, resources, age, pregnancy, disability, or blindness [medquad_ans_0000586-1__dup2]. The Affordable Care Act also expands health insurance coverage for many people in the U.S., which may indirectly support access to financial assistance [medquad_ans_0000431-1__dup1].

**Supported citations**: `medquad_ans_0000362-1__dup2`, `medquad_ans_0000586-1__dup2`, `medquad_ans_0000431-1__dup1`
**Unsupported citations**: _none_

### Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Scores**: context_relevance=0.800, groundedness=0.984, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000258-2__dup3` (score=0.9836) - LCMV is most commonly recognized as causing neurological disease, as its name implies, though infection without symptoms or mild febrile illnesses are more common clinical manifestations. For infected persons who do b...
2. `medquad_ans_0004105-1` (score=0.8755) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About hal...
3. `medquad_ans_0000165-3__dup1` (score=0.0126) - Generally, long-term antibiotic treatment to destroy the bacteria can relieve symptoms and cure the disease. If left untreated, the disease is progressive and fatal. Individuals with involvement of the central nervous...
4. `medquad_ans_0000599-1__dup1` (score=0.0084) - Meningococci are a type of bacteria that cause serious infections. The most common infection is meningitis, which is an inflammation of the thin tissue that surrounds the brain and spinal cord. Meningococci can also c...
5. `medquad_ans_0000262-3__dup1` (score=0.0065) - There is no cure for these headaches. The disorder is not fatal but can cause considerable discomfort.

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
3. `medquad_ans_0000045-2` (score=0.0072) - More than 900 people with Alstrm syndrome have been reported worldwide.
4. `medquad_ans_0000055-4__dup1` (score=0.0060) - The NINDS supports a broad spectrum of research on motor neuron diseases, such as Kennedy's disease. Much of this research is aimed at increasing scientific understanding of these diseases and, ultimately, finding way...
5. `medquad_ans_0000025-2` (score=0.0022) - ALSP is thought to be a rare disorder, although the prevalence is unknown. Because it can be mistaken for other disorders with similar symptoms, ALSP may be underdiagnosed.

**Answer**:

Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases, the underlying gene abnormality cannot be determined. Juvenile ALS may be inherited in an autosomal dominant (as in ALS type 4) or autosomal recessive (as in ALS type 2) fashion [medquad_ans_0003319-3].

**Supported citations**: `medquad_ans_0003319-3`
**Unsupported citations**: _none_

### Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Scores**: context_relevance=0.889, groundedness=0.958, answer_relevance=0.711

**Top docs**:

1. `medquad_ans_0000219-11` (score=0.9849) - - Erectile dysfunction (ED) is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last...
2. `medquad_ans_0000219-1__dup4` (score=0.9725) - Erectile dysfunction\* is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last long ...
3. `medquad_ans_0000058-19` (score=0.9633) - Yes. Any of the symptoms caused by prostate cancer may also be due to enlargement of the prostate, which is not cancer. If you have any of the symptoms mentioned in question #10, see your doctor or a urologist to find...
4. `medquad_ans_0000219-7` (score=0.9372) - To find the cause of your ED, your doctor may - take your medical and sexual history - ask you questions about your mental health - give you a physical exam - test your blood - give you a nighttime erection test - per...
5. `medquad_ans_0000058-24` (score=0.8807) - Surgery, radiation therapy, and hormonal therapy all have the potential to disrupt sexual desire or performance for a short while or permanently. Discuss your concerns with your health care provider. Several options a...

**Answer**:

To learn about erectile dysfunction (ED), you should know that it is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last long enough for sex - are unable to get an erection at all [medquad_ans_0000219-11]. An erection occurs when blood flow into the penis increases, making the penis larger and firmer, and involves the interaction of hormones, blood vessels, nerves, and muscles [medquad_ans_0000219-11]. ED often has multiple causes, and many diseases can damage nerves, arteries, and muscles [medquad_ans_0000219-11].

To find the cause of ED, your doctor may take your medical and sexual history, ask about your mental health, perform a

**Supported citations**: `medquad_ans_0000219-11`
**Unsupported citations**: _none_

### Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.954, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0006267-3` (score=0.9702) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
2. `medquad_ans_0000171-4__dup1` (score=0.0222) - The NINDS supports research on congenital disorders such as KTS with the goal of finding new means to treat and prevent them.
3. `medquad_ans_0000201-4__dup1` (score=0.0187) - The NINDS supports a broad range of research on neuromuscular disorders with the goal of finding ways to prevent, treat, and, ultimately, cure them.
4. `medquad_ans_0000981-5` (score=0.0061) - These resources address the diagnosis or management of tibial muscular dystrophy: - Gene Review: Gene Review: Udd Distal Myopathy - Genetic Testing Registry: Distal myopathy Markesbery-Griggs type These resources from...
5. `medquad_ans_0000205-4__dup1` (score=0.0031) - The NINDS supports a broad program of research studies on MD. The goals of these studies are to understand MD and to develop techniques to diagnose, treat, prevent, and ultimately cure the disorder. The NINDS is a mem...

**Answer**:

Physical therapy, including early mobilization, regular stretching, and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation may become necessary in the first or second decade of life. Prevention of chest infections can be achieved with the use of antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that people with Ullrich congenital muscular dystrophy may benefit from cyclosporin A, although more studies are needed to confirm its benefits [medquad_ans_0006267-3].

**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

### Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Scores**: context_relevance=0.375, groundedness=0.058, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0004039-4` (score=0.0482) - Is microhydranencephaly inherited? Most cases of microhydranencephaly occur sporadically in people with no family history of the condition. However, the condition can rarely affect more than one family member and be i...
2. `medquad_ans_0000042-4__dup1` (score=0.0069) - How is achondrogenesis inherited? Achondrogenesis type 1A and type 1B are believed to be inherited in an autosomal recessive pattern. Autosomal recessive inheritance means both copies of the gene in each cell have mut...
3. `medquad_ans_0000508-4` (score=0.0014) - The severe forms of hypophosphatasia that appear early in life are inherited in an autosomal recessive pattern. Autosomal recessive inheritance means that two copies of the gene in each cell are altered. Most often, t...
4. `medquad_ans_0000503-2` (score=0.0006) - Hypochromic microcytic anemia with iron overload is likely a rare disorder; at least five affected families have been reported in the scientific literature.
5. `medquad_ans_0000102-5` (score=0.0001) - These resources address the diagnosis or management of MCPH: - Gene Review: Gene Review: Primary Autosomal Recessive Microcephalies and Seckel Syndrome Spectrum Disorders - Genetic Testing Registry: Primary autosomal ...

**Answer**:

The provided context does not contain information about the inheritance pattern of microcephalic osteodysplastic primordial dwarfism type II. Therefore, based on the given context, it is not possible to determine how this condition is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000136-4` (score=0.7393) - BOR/BO syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In about 90 percent of cases, an affected person inherits the m...
2. `medquad_ans_0006111-4` (score=0.0050) - Is Tietze syndrome inherited? Tietze syndrome is not thought to be inherited. Most cases occur sporadically in people with no family history of the condition.
3. `medquad_ans_0000912-4` (score=0.0049) - Silver syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In these cases, the affected person inherits the mutation from ...
4. `medquad_ans_0001091-2` (score=0.0001) - 3-M syndrome is a rare disorder. About 50 individuals with this disorder have been identified worldwide.
5. `medquad_ans_0000227-2` (score=0.0001) - CHILD syndrome is a rare disorder; it has been reported in about 60 people worldwide. This condition occurs almost exclusively in females.

**Answer**:

The provided context does not contain any information about Ochoa syndrome or its inheritance pattern. Therefore, it is not possible to determine whether Ochoa syndrome is inherited based on the given text.

Answer: Insufficient context [medquad_ans_0000136-4, medquad_ans_0006111-4, medquad_ans_0000912-4, medquad_ans_0001091-2, medquad_ans_0000227-2].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000136-4, medquad_ans_0006111-4, medquad_ans_0000912-4, medquad_ans_0001091-2, medquad_ans_0000227-2`

### Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Scores**: context_relevance=0.625, groundedness=0.788, answer_relevance=0.725

**Top docs**:

1. `medquad_ans_0000531-5` (score=0.9960) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management...
2. `medquad_ans_0001086-2` (score=0.0002) - At least 500 cases of 22q13.3 deletion syndrome are known.
3. `medquad_ans_0000531-2` (score=0.0001) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
4. `medquad_ans_0006209-1` (score=0.0001) - Triploidy is a chromosome abnormality that occurs when there is an extra set of chromosomes present in each cell. Most pregnancies affected by triploidy are lost through early miscarriage. However, reports exist of so...
5. `medquad_ans_0000531-3` (score=0.0001) - Isodicentric chromosome 15 syndrome results from the presence of an abnormal extra chromosome, called an isodicentric chromosome 15, in each cell. An isodicentric chromosome contains mirror-image segments of genetic m...

**Answer**:

The provided context does not specify treatments for isodicentric chromosome 15 syndrome. While it mentions that resources from MedlinePlus cover diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care in relation to the diagnosis and management of health conditions, it does not explicitly state which of these apply to isodicentric chromosome 15 syndrome [medquad_ans_0000531-5]. Therefore, based on the given information, the treatments for isodicentric chromosome 15 syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000531-5`
**Unsupported citations**: _none_

### Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.895, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006224-3` (score=0.9931) - What causes Troyer syndrome? Troyer syndrome is caused by mutations in the SPG20 gene. This gene gives the body instructions to make a protein called spartin, which is present in many body tissues, including those of ...
2. `medquad_ans_0000272-4__dup1` (score=0.0002) - The NINDS supports research on genetic disorders such as the hereditary spastic paraplegias. A gene for Troyer syndrome has been identified and others may be identified in the future. Understanding how these genes cau...
3. `medquad_ans_0000403-2` (score=0.0001) - Genitopatellar syndrome is estimated to occur in fewer than 1 per million people. At least 18 cases have been reported in the medical literature.
4. `medquad_ans_0000964-2` (score=0.0001) - Swyer syndrome occurs in approximately 1 in 80,000 people.
5. `medquad_ans_0000998-2` (score=0.0001) - About 20 cases of Troyer syndrome have been reported in the Old Order Amish population of Ohio. It has not been found outside this population.

**Answer**:

Troyer syndrome is caused by mutations in the SPG20 gene. This gene provides instructions for making a protein called spartin, which is present in many body tissues, including those of the nervous system. It is thought that the syndrome is caused by a loss of function of the spartin protein, although the exact role of this protein in cellular functions is not fully understood [medquad_ans_0006224-3].

**Supported citations**: `medquad_ans_0006224-3`
**Unsupported citations**: _none_

### Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.955, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000983-3` (score=0.9990) - Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. This gene provides instructions for making a protein that acts as a channel across cell membranes. This channel, known as CaV1....
2. `medquad_ans_0000902-3` (score=0.0013) - Mutations in the KCNH2, KCNJ2, and KCNQ1 genes can cause short QT syndrome. These genes provide instructions for making channels that transport positively charged atoms (ions) of potassium out of cells. In cardiac mus...
3. `medquad_ans_0000174-3` (score=0.0004) - Mutations in the CHD7 gene cause more than half of all cases of CHARGE syndrome. The CHD7 gene provides instructions for making a protein that most likely regulates gene activity (expression) by a process known as chr...
4. `medquad_ans_0000910-3` (score=0.0003) - Sick sinus syndrome can result from genetic or environmental factors. In many cases, the cause of the condition is unknown. Genetic changes are an uncommon cause of sick sinus syndrome. Mutations in two genes, SCN5A a...
5. `medquad_ans_0000140-3` (score=0.0002) - Brugada syndrome can be caused by mutations in one of several genes. The most commonly mutated gene in this condition is SCN5A, which is altered in approximately 30 percent of affected individuals. This gene provides ...

**Answer**:

Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. These mutations change the structure of CaV1.2 channels, causing them to stay open much longer than usual, which leads to an abnormal flow of calcium ions into cells. This calcium overload alters the way cardiac muscle cells beat and can cause arrhythmia [medquad_ans_0000983-3].

**Supported citations**: `medquad_ans_0000983-3`
**Unsupported citations**: _none_

### Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Scores**: context_relevance=1.000, groundedness=0.933, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000061-3__dup3` (score=0.9930) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by i...
2. `medquad_ans_0000061-11` (score=0.9910) - Rheumatoid arthritis can be difficult to diagnose in its early stages because the full range of symptoms develops over time, and only a few symptoms may be present in the early stages. As part of the diagnosis, your d...
3. `medquad_ans_0000061-10` (score=0.0139) - Rheumatoid arthritis is characterized by inflammation of the joint lining. This inflammation causes warmth, redness, swelling, and pain around the joints. The pain of rheumatoid arthritis varies greatly from person to...
4. `medquad_ans_0000061-13` (score=0.0024) - Medication, exercise, and, in some cases, surgery are common treatments for this disease. Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation...
5. `medquad_ans_0000847-2` (score=0.0006) - Psoriatic arthritis affects an estimated 24 in 10,000 people. Between 5 and 10 percent of people with psoriasis develop psoriatic arthritis, according to most estimates. Some studies suggest a figure as high as 30 per...

**Answer**:

The symptoms of rheumatoid arthritis include pain and stiffness in the joints, warmth, redness, swelling, and pain around the joints. People also often feel sick, tired, and may experience fever. Rheumatoid arthritis generally occurs in a symmetrical pattern, meaning if one knee or hand is affected, the other is also likely to be affected [medquad_ans_0000061-3__dup3].

**Supported citations**: `medquad_ans_0000061-3__dup3`
**Unsupported citations**: _none_

### Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Scores**: context_relevance=0.800, groundedness=0.521, answer_relevance=0.480

**Top docs**:

1. `medquad_ans_0002796-1` (score=0.9718) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is no...
2. `medquad_ans_0000149-1__dup3` (score=0.9287) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, d...
3. `medquad_ans_0004053-1` (score=0.0025) - Microscopic polyangiitis (MPA) is a disorder that causes blood vessel inflammation (vasculitis), which can lead to organ damage. The kidneys, lungs, nerves, skin, and joints are the most commonly affected areas of the...
4. `medquad_ans_0004874-1` (score=0.0021) - Pigmented purpuric eruption is a condition that causes reddish-brown skin lesions, most commonly on the lower legs. In some cases, the skin lesions cause severe itching. The skin lesions may spread over time, or clear...
5. `medquad_ans_0000663-2` (score=0.0020) - MPAN is a rare condition that is estimated to affect less than 1 in 1 million people.

**Answer**:

Henoch-Schönlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). It may develop as an immune response to an infection, and while the exact cause is not fully understood, it is usually seen in children but can affect people of any age. Most cases go away on their own without treatment, and the main goal of treatment, when needed, is to relieve symptoms such as joint pain, abdominal pain, or swelling. Over-the-counter medicines can be used in many cases, and in some patients with severe arthritis, prednisone, a steroid medicine, may be prescribed. HSP can also cause inflammation of small blood vessels, leading to a rash that resembles many small raised bruises. In adults, HSP can lead to chronic kidney disease (CKD) and kidney failure, described as

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Scores**: context_relevance=1.000, groundedness=0.967, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0003166-1` (score=0.9994) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information ...
2. `medquad_ans_0001025-2` (score=0.0002) - Waldenstrm macroglobulinemia affects an estimated 3 per million people each year in the United States. Approximately 1,500 new cases of the condition are diagnosed each year in this country, and whites are more common...
3. `medquad_ans_0000483-2` (score=0.0002) - Histiocytosis-lymphadenopathy plus syndrome is a rare disorder, affecting approximately 100 individuals worldwide.
4. `medquad_ans_0003163-1` (score=0.0001) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information ...
5. `medquad_ans_0002992-4` (score=0.0001) - Is hyper IgD syndrome inherited? Hyper IgD syndrome is inherited in an autosomal recessive manner, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recess...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [medquad_ans_0003166-1].

**Supported citations**: `medquad_ans_0003166-1`
**Unsupported citations**: _none_

### Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.960, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000896-3` (score=0.9991) - Senior-Lken syndrome can be caused by mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopic, finger...
2. `medquad_ans_0000033-3__dup1` (score=0.0462) - Average lifespan has been improving for years, for unknown reasons, and varies with the severity of the underlying mutations, ATM (ataxia-telangiectasia mutated) protein levels, and residual ATM kinase activity. Some ...
3. `medquad_ans_0000017-3` (score=0.0003) - In very few families, people develop AD in their 30s, 40s, and 50s. This is known as "early onset" AD. These individuals have a mutation in one of three different inherited genes that causes the disease to begin at an...
4. `medquad_ans_0000896-2` (score=0.0001) - Senior-Lken syndrome is a rare disorder, with an estimated prevalence of about 1 in 1 million people worldwide. Only a few families with the condition have been described in the medical literature.
5. `medquad_ans_0002013-2` (score=0.0001) - How is early-onset, autosomal dominant Alzheimer disease inherited? Early-onset, autosomal dominant Alzheimer disease is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a...

**Answer**:

Genetic changes related to Senior-Lken syndrome include mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia, which are involved in signaling pathways between cells. Mutations in these genes likely lead to problems with the structure and function of cilia, disrupting important chemical signaling pathways. However, it remains unclear how these defects specifically lead to nephronophthisis and Leber congenital amaurosis. In some cases, individuals with Senior-Lken syndrome do not have identified mutations in these five genes, and the genetic cause in those cases is unknown [medquad_ans_0000896-3].

**Supported citations**: `medquad_ans_0000896-3`
**Unsupported citations**: _none_

### Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.931, answer_relevance=0.350

**Top docs**:

1. `medquad_ans_0000454-2` (score=0.9991) - Hereditary angioedema is estimated to affect 1 in 50,000 people. Type I is the most common, accounting for 85 percent of cases. Type II occurs in 15 percent of cases, and type III is very rare.
2. `medquad_ans_0000610-2` (score=0.9988) - The prevalence of lymphedema-distichiasis syndrome is unknown. Because the extra eyelashes can be overlooked during a medical examination, researchers believe that some people with this condition may be misdiagnosed a...
3. `medquad_ans_0000644-2` (score=0.0064) - The prevalence of Meige disease is unknown. Collectively, the many types of primary lymphedema affect an estimated 1 in 100,000 people younger than 20; Meige disease is the most common type of primary lymphedema. For ...
4. `medquad_ans_0003626-1` (score=0.0001) - Lipedema is a syndrome characterized by symmetric enlargement of the legs due to deposits of fat beneath the skin, which is often painful. It is a common condition affecting up to 11% of women The underlying cause is ...
5. `medquad_ans_0000572-1__dup1` (score=0.0001) - Lymphedema is the name of a type of swelling. It happens when lymph builds up in your body's soft tissues. Lymph is a fluid that contains white blood cells that defend against germs. It can build up when the lymph sys...

**Answer**:

The prevalence of lymphedema-distichiasis syndrome is unknown [medquad_ans_0000610-2].

**Supported citations**: `medquad_ans_0000610-2`
**Unsupported citations**: _none_

### Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Scores**: context_relevance=1.000, groundedness=0.845, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000399-1__dup1` (score=0.9977) - Summary : Genetic tests are tests on blood and other tissue to find genetic disorders. Over 2000 tests are available. Doctors use genetic tests for several reasons. These include - Finding genetic diseases in unborn b...
2. `medquad_ans_0000397-1__dup1` (score=0.1860) - Summary : Genetic counseling provides information and support to people who have, or may be at risk for, genetic disorders. A genetic counselor meets with you to discuss genetic risks. The counseling may be for yourse...
3. `medquad_ans_0000013-25` (score=0.0037) - Genetic testing may be helpful whether the test identifies a mutation or not. Test results can - serve as a relief, eliminating some of the uncertainty about a health condition - help doctors make recommendations for ...
4. `medquad_ans_0000427-1__dup2` (score=0.0030) - Regular health exams and tests can help find problems before they start. They also can help find problems early, when your chances for treatment and cure are better. Which exams and screenings you need depends on your...
5. `medquad_ans_0005122-3` (score=0.0001) - Is genetic testing available for Cushing's symphalangism? GeneTests lists the names of laboratories that are performing genetic testing for Cushing's symphalangism. To view the contact information for the clinical lab...

**Answer**:

Yes, I have information about genetic testing. Genetic tests are conducted on blood and other tissues to identify genetic disorders. Over 2000 such tests are available, and doctors use them for various purposes, including detecting genetic diseases in unborn babies, determining if individuals carry a gene that could be passed to their children, screening embryos for disease, testing adults before symptoms appear, making a diagnosis in someone with symptoms, and determining the most suitable type or dose of medication for a person [medquad_ans_0000399-1__dup1]. Test results can provide relief by reducing uncertainty about a health condition, assist doctors in making treatment or monitoring recommendations, support individuals in making health and family-related decisions, encourage preventive actions such as earlier screenings or lifestyle changes, and help people make informed choices about future decisions like having a child [medquad_ans_0000013-25]. Genetic counseling is often provided to

**Supported citations**: `medquad_ans_0000399-1__dup1`, `medquad_ans_0000013-25`
**Unsupported citations**: _none_

### Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.982, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000149-1` (score=0.8681) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxy...
2. `medquad_ans_0003238-1` (score=0.0137) - Intracranial arteriovenous malformations (AVMs) are abnormal connections between the arteries and veins in the brain. Most people with brain or spinal AVMs experience few, if any, major symptoms. About 12 percent of p...
3. `medquad_ans_0000060-3__dup1` (score=0.0044) - Rebleeding from a cavernous angioma is common, it is not predictable, and individuals frequently have multiple CCMs found via magnetic resonance imaging. Individuals with CCM are faced with a diagnosis that imparts ri...
4. `medquad_ans_0000168-1` (score=0.0039) - Cerebral cavernous malformations are collections of small blood vessels (capillaries) in the brain that are enlarged and irregular in structure. These capillaries have abnormally thin walls, and they lack other suppor...
5. `medquad_ans_0000056-1__dup3` (score=0.0021) - Arteriovenous malformations (AVMs) are defects in your vascular system. The vascular system includes arteries, veins, and capillaries. Arteries carry blood away from the heart to other organs; veins carry blood back t...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin, appearing as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs, and may be visible from birth or develop during childhood [medquad_ans_0000149-1]. In most cases, these malformations do not cause health problems and may be the only sign of the disorder. However, some individuals also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs), which are abnormal connections between arteries, veins, and capillaries that can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure,

**Supported citations**: `medquad_ans_0000149-1`
**Unsupported citations**: _none_

### Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Scores**: context_relevance=0.833, groundedness=0.015, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000112-3__dup1` (score=0.9985) - The initial symptoms can be very mild and may be noticeable only after prolonged exertion, stress, or fatigue. Dystonias often progress through various stages. Initially, dystonic movements are intermittent and appear...
2. `medquad_ans_0000971-2` (score=0.0056) - Task-specific focal dystonia affects an estimated 7 to 69 per million people in the general population. Musician's dystonia that is severe enough to impact performance occurs in about 1 percent of musicians.
3. `medquad_ans_0000299-2` (score=0.0027) - Dopa-responsive dystonia is estimated to affect 1 per million people worldwide. However, the disorder is likely underdiagnosed because the condition may not be identified in people with mild symptoms, or it may be mis...
4. `medquad_ans_0000704-2` (score=0.0005) - The prevalence of myoclonus-dystonia is unknown. This condition has been described in people worldwide.
5. `medquad_ans_0000015-2` (score=0.0004) - The prevalence of ADCY5-related dyskinesia is unknown. At least 50 affected individuals have been described in the medical literature.

**Answer**:

The provided context does not contain information about the symptoms of Dystonia 16. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Scores**: context_relevance=0.714, groundedness=0.931, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000899-5` (score=0.9950) - These resources address the diagnosis or management of septo-optic dysplasia: - Genetic Testing Registry: Septo-optic dysplasia sequence - MedlinePlus Encyclopedia: Growth Hormone Deficiency - MedlinePlus Encyclopedia...
2. `medquad_ans_0005548-5` (score=0.9652) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not tr...
3. `medquad_ans_0001478-6` (score=0.0185) - How might congenital anosmia be treated? Unfortunately, there is currently no known cure or treatment for congenital anosmia.
4. `medquad_ans_0000189-2__dup1` (score=0.0081) - There is no standard treatment for megalencephaly. Treatment will depend upon the disorder with which the megalencephaly is associated and will address individual symptoms and disabilities.
5. `medquad_ans_0000924-5` (score=0.0069) - These resources address the diagnosis or management of SOX2 anophthalmia syndrome: - Gene Review: Gene Review: SOX2-Related Eye Disorders - Genetic Testing Registry: Microphthalmia syndromic 3 - MedlinePlus Encycloped...

**Answer**:

There is no cure for septo-optic dysplasia. Treatment is symptomatic and focuses on managing hormone deficiencies with hormone replacement therapy. Optical problems are generally not treatable, and vision, physical, and occupational therapies may be required [medquad_ans_0005548-5].

**Supported citations**: `medquad_ans_0005548-5`
**Unsupported citations**: _none_

### Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Scores**: context_relevance=0.900, groundedness=0.970, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000208-7` (score=0.9931) - The complications of benign prostatic hyperplasia may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney da...
2. `medquad_ans_0000208-11` (score=0.9898) - The complications of benign prostatic hyperplasia treatment depend on the type of treatment. Medications Medications used to treat benign prostatic hyperplasia may have side effects that sometimes can be serious. Men ...
3. `medquad_ans_0000058-19` (score=0.9847) - Yes. Any of the symptoms caused by prostate cancer may also be due to enlargement of the prostate, which is not cancer. If you have any of the symptoms mentioned in question #10, see your doctor or a urologist to find...
4. `medquad_ans_0000036_3-3` (score=0.0282) - Different factors increase or decrease the risk of developing prostate cancer. Anything that increases a person's chance of developing a disease is called a risk factor. Anything that decreases your chance of getting ...
5. `medquad_ans_0000213-9` (score=0.0052) - Complications of urinary retention and its treatments may include - UTIs - bladder damage - kidney damage - urinary incontinence after prostate, tumor, or cancer surgery UTIs. Urine is normally sterile, and the normal...

**Answer**:

The complications of benign prostatic hyperplasia (prostate enlargement) may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney damage - bladder stones. Most men with benign prostatic hyperplasia do not develop these complications. However, kidney damage in particular can be a serious health threat when it occurs [medquad_ans_0000208-7].

Complications related to treatment include side effects from medications such as hives, rash, itching, shortness of breath, rapid or irregular heartbeat, painful erection lasting hours, swelling of the face, lips, tongue, throat, or extremities, difficulty breathing or swallowing, chest pain, dizziness or fainting upon standing, sudden vision or hearing loss, blurred vision, and chest pain, dizziness, or nausea during sexual activity, mostly associated with phosphodiesterase

**Supported citations**: `medquad_ans_0000208-7`
**Unsupported citations**: _none_

### Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Scores**: context_relevance=0.444, groundedness=0.000, answer_relevance=0.711

**Top docs**:

1. `medquad_ans_0000004-2__dup1` (score=0.9962) - Treatment for LKS usually consists of medications, such as anticonvulsants and corticosteroids, and speech therapy, which should be started early. A controversial treatment option involves a surgical technique called ...
2. `medquad_ans_0000117-2__dup2` (score=0.9951) - Generally, surgery is performed during infancy to place the protruding tissues back into the skull, remove the sac, and correct the associated craniofacial abnormalities. Even large protrusions can often be removed wi...
3. `medquad_ans_0000150-2__dup1` (score=0.9929) - There is no standard course of treatment for holoprosencephaly. Treatment is symptomatic and supportive.
4. `medquad_ans_0000019-2__dup1` (score=0.0442) - There is no cure or standard treatment for anencephaly. Treatment is supportive.
5. `medquad_ans_0001478-6` (score=0.0141) - How might congenital anosmia be treated? Unfortunately, there is currently no known cure or treatment for congenital anosmia.

**Answer**:

The provided context does not contain any information about the treatments for absence of the septum pellucidum. Therefore, based on the given text, it is not possible to answer the question. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Scores**: context_relevance=1.000, groundedness=0.517, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.9992) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (...
2. `medquad_ans_0000480-2` (score=0.0009) - Hidradenitis suppurativa was once thought to be a rare condition because only the most severe cases were reported. However, recent studies have shown that the condition affects at least 1 in 100 people when milder cas...
3. `medquad_ans_0000122-5__dup1` (score=0.0006) - The most common signs and symptoms of ulcerative colitis are diarrhea with blood or pus and abdominal discomfort. Other signs and symptoms include - an urgent need to have a bowel movement - feeling tired - nausea or ...
4. `medquad_ans_0000107-2` (score=0.0005) - Baraitser-Winter syndrome is a rare condition. Fewer than 50 cases have been reported in the medical literature.
5. `medquad_ans_0000342-2` (score=0.0001) - Familial cold autoinflammatory syndrome is a very rare condition, believed to have a prevalence of less than 1 per million people.

**Answer**:

The symptoms of Febrile Ulceronecrotic Mucha-Habermann disease (FUMHD) include red scaly skin lesions (papules) that ulcerate, break down, form open sores, and then develop a red-brown crust (i.e., PLEVA). The lesions can progress to large, destructive ulcers with extensive, painful loss of skin tissue. Skin lesions may become infected, leading to pus and a putrid odor. FUMHD is often associated with high fever (up to 104°F), which may be persistent or intermittent. Other symptoms may include feeling ill, sore throat, congestion, muscle soreness or pain, joint pain, diarrhea, central nervous system symptoms, abdominal pain, enlarged spleen, arthritis, megaloblastic anemia, interstitial pneumonitis, lymphocytic (viral) myocarditis, and sepsis. FUMHD can become life

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Scores**: context_relevance=1.000, groundedness=0.954, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000873-4` (score=0.9909) - Most cases of retroperitoneal fibrosis are sporadic, which means that they occur in people with no apparent history of the disorder in their family. In rare cases, the condition has been reported to occur in a few mem...
2. `medquad_ans_0002833-4` (score=0.9639) - How is hereditary leiomyomatosis and renal cell cancer inherited? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the ...
3. `medquad_ans_0005300-4` (score=0.0130) - Is a renal oncocytoma inherited? Most renal oncocytomas are not inherited. They usually occur sporadically in people with no family history of tumors. However, in rare cases, they can occur in people with certain gene...
4. `medquad_ans_0005299-3` (score=0.0033) - Is renal nutcracker syndrome inherited? Renal nutcracker syndrome is not inherited. Most cases occur sporadically in people with no family history of the condition. Although more than one family member may rarely be a...
5. `medquad_ans_0002833-1` (score=0.0001) - Hereditary leiomyomatosis and renal cell cancer (HLRCC) is a condition that causes benign tumors of smooth muscle tissue in the skin (cutaneous leiomyomas) and in the uterus in females (uterine leiomyomas, or fibroids...

**Answer**:

Yes, hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the gene in each cell is enough to cause symptoms of the condition [medquad_ans_0002833-4]. This inheritance pattern applies regardless of which parent has the condition, and each child of an affected individual has a 50% (1 in 2) risk to inherit the mutated gene [medquad_ans_0002833-4]. HLRCC is caused by mutations in the FH gene [medquad_ans_0002833-1].

**Supported citations**: `medquad_ans_0002833-4`, `medquad_ans_0002833-1`
**Unsupported citations**: _none_

### Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002445-1` (score=0.9994) - What are the signs and symptoms of Frontometaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontometaphyseal dysplasia. If the information is available, the tabl...
2. `medquad_ans_0000385-2` (score=0.0017) - Fraser syndrome affects an estimated 1 in 200,000 newborns. The condition occurs in approximately 1 in 10,000 fetuses that do not survive to birth.
3. `medquad_ans_0000501-2` (score=0.0017) - Hypochondrogenesis and achondrogenesis, type 2 (a similar skeletal disorder) together affect 1 in 40,000 to 60,000 newborns.
4. `medquad_ans_0000757-2` (score=0.0015) - This condition affects an estimated 6 to 7 per 100,000 people worldwide. Types I and IV are the most common forms of osteogenesis imperfecta, affecting 4 to 5 per 100,000 people.
5. `medquad_ans_0000386-2` (score=0.0013) - Frasier syndrome is thought to be a rare condition; approximately 50 cases have been described in the scientific literature.

**Answer**:

The symptoms of Frontometaphyseal dysplasia include abnormality of dental morphology, abnormality of frontal sinus, abnormality of the metaphyses, bowing of the long bones, camptodactyly of finger, craniofacial hyperostosis, hypertelorism, limitation of joint mobility, prominent supraorbital ridges, abnormal form of the vertebral bodies, abnormality of the palate, accelerated skeletal maturation, aplasia/hypoplasia of the thumb, arachnodactyly, conductive hearing impairment, elbow dislocation, scoliosis, sensorineural hearing impairment, skeletal muscle atrophy, synostosis of carpal bones, ulnar deviation of finger, abnormality of the larynx, abnormality of the urethra, complete atrioventricular canal defect, craniosynostosis, tracheal stenosis, ureteral stenosis, ankle

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Scores**: context_relevance=0.818, groundedness=0.891, answer_relevance=0.673

**Top docs**:

1. `medquad_ans_0000097-9` (score=0.9854) - Diverticulitis can attack suddenly and cause complications, such as - an abscessa painful, swollen, pus-filled area just outside the colon wallcaused by infection - a perforationa small tear or hole in the diverticula...
2. `medquad_ans_0000128-7` (score=0.9832) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding,...
3. `medquad_ans_0000128-9` (score=0.9815) - A doctor may prescribe antibiotics and recommend following a liquid diet. Most people get better with this treatment. Some people may need surgery and other treatments. - Surgery. Serious problems from diverticulitis ...
4. `medquad_ans_0000097-8` (score=0.9577) - A health care provider may treat the symptoms of diverticulosis with a high-fiber diet or fiber supplements, medications, and possibly probiotics. Treatment for diverticular disease varies, depending on whether a pers...
5. `medquad_ans_0000128-11` (score=0.9136) - - Diverticular disease is more common in people as they grow older. - A low-fiber diet is the most likely cause of the disease. - Most people are treated with a high-fiber diet and pain medication. - Add whole grain f...

**Answer**:

The treatments for diverticular disease, particularly when complications arise, include:

- **Abscess**: Antibiotic treatment is usually used to prevent or treat an abscess. If the abscess is large or does not respond to antibiotics, it may need to be drained. This procedure involves inserting a needle through the skin to the abscess under imaging guidance (abdominal ultrasound or CT scan) to drain the fluid using a catheter [medquad_ans_0000097-9].

- **Perforation**: Surgery is typically required to repair the tear or hole. In some cases, a portion of the intestine may need to be removed if the perforation cannot be repaired [medquad_ans_0000097-9].

- **Peritonitis**: Immediate surgery is required to clean the abdominal cavity, and possibly a colon resection after antibiotic treatment. The patient may also need a blood transfusion if significant blood

**Supported citations**: `medquad_ans_0000097-9`
**Unsupported citations**: _none_

### Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Scores**: context_relevance=0.556, groundedness=0.000, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000556-2` (score=0.0006) - KBG syndrome is a rare disorder that has been reported in around 60 individuals. For unknown reasons, males are affected more often than females. Doctors think the disorder is underdiagnosed because the signs and symp...
2. `medquad_ans_0000226-3__dup1` (score=0.0003) - The prognosis for individuals with Parry-Romberg syndrome varies. In some cases, the atrophy ends before the entire face is affected. In mild cases, the disorder usually causes no disability other than cosmetic effects.
3. `medquad_ans_0000172-3__dup1` (score=0.0002) - There is no cure for Klver-Bucy syndrome. The disorder is not life-threatening, but the patient can be difficult to manage. With treatment, symptoms may slowly decline.
4. `medquad_ans_0000748-2` (score=0.0001) - The Maat-Kievit-Brunner type of Ohdo syndrome is a very rare condition, with only a few affected individuals reported in the medical literature.
5. `medquad_ans_0000894-2` (score=0.0001) - Schinzel-Giedion syndrome is very rare, although the exact prevalence is unknown.

**Answer**:

The provided context does not contain any information about the symptoms of Baraitser Brett Piesowicz syndrome. Therefore, it is not possible to answer the question based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Scores**: context_relevance=0.667, groundedness=0.075, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000052-2__dup1` (score=0.9958) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
2. `medquad_ans_0000060-2__dup1` (score=0.9927) - The primary treatment option for a CCM is surgical removal. Radiation therapy has not been shown to be effective. The decision to operate is made based upon the risk of approaching the lesion. For example, symptomatic...
3. `medquad_ans_0000234-2__dup1` (score=0.9922) - Generally, treatment depends on the type of tumor, the size of the tumor, whether the tumor has invaded or pressed on surrounding structures, such as the brain and visual pathways, and the individuals age and overall ...
4. `medquad_ans_0000001-8__dup1` (score=0.9895) - No single treatment is effective for all patients. Treatment should be individualized, and often combined, depending on patient characteristics such as age and tumor size. If the tumor has not yet invaded surrounding ...
5. `medquad_ans_0000027-2__dup1` (score=0.9833) - There has been active debate about how to treat arachnoid cysts. The need for treatment depends mostly upon the location and size of the cyst. If the cyst is small, not disturbing surrounding tissue, and not causing s...

**Answer**:

The provided context does not contain information about the treatments for meningioma. Therefore, it is not possible to answer the question based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000006_7-8` (score=0.9738) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a te...
2. `medquad_ans_0000015_1-8` (score=0.9636) - Key Points - There are different types of treatment for children with extracranial germ cell tumors. - Children with extracranial germ cell tumors should have their treatment planned by a team of health care providers...
3. `medquad_ans_0000006_2-8` (score=0.6526) - Key Points - There are different types of treatment for children with brain and spinal cord tumors. - Children with brain or spinal cord tumors should have their treatment planned by a team of health care providers wh...
4. `medquad_ans_0000006_5-7` (score=0.0911) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the...
5. `medquad_ans_0000006_5-9` (score=0.0390) - Key Points - There are different types of treatment for children with brain stem glioma. - Children with brain stem glioma should have their treatment planned by a team of health care providers who are experts in trea...

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
3. `medquad_ans_0000568-2` (score=0.0025) - Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, where approximately 1 in 12,500 individuals have the condition. Kufs disease is thought to repres...
4. `medquad_ans_0000516-2` (score=0.0014) - Although the prevalence of IBMPFD is unknown, this condition is rare. It has been identified in about 26 families.
5. `medquad_ans_0000575-2` (score=0.0008) - Although Laing distal myopathy is thought to be rare, its prevalence is unknown. Several families with the condition have been identified worldwide.

**Answer**:

Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (myopathy). This condition is caused by mutations in the PNPLA2 gene and is inherited in an autosomal recessive pattern. Other features may include a fatty liver, a weakened and enlarged heart (cardiomyopathy), inflammation of the pancreas (pancreatitis), reduced thyroid activity (hypothyroidism), and type 2 diabetes mellitus. Signs and symptoms vary greatly among affected individuals. There is currently no treatment to correct the underlying metabolic problem [medquad_ans_0004401-1][medquad_ans_0000729-1].

**Supported citations**: `medquad_ans_0004401-1`, `medquad_ans_0000729-1`
**Unsupported citations**: _none_

### Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000001-7__dup1` (score=0.9207) - Topics
2. `medquad_ans_0000981-1__dup1` (score=0.4161) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been repo...
3. `medquad_ans_0006546-1` (score=0.0613) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. ...
4. `medquad_ans_0000073-3__dup1` (score=0.0054) - COFS is a fatal disease. Most children do not live beyond five years.
5. `medquad_ans_0000092-4__dup1` (score=0.0010) - The National Institute of Neurological Disorders and Stroke (NINDS), and other institutes of the National Institutes of Health (NIH), conduct research related to CMV infection in laboratories at the NIH, and support a...

**Answer**:

Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been reports that the virus has spread through blood transfusions. Most people who get the virus do not get sick. One in five people do get symptoms, which can include a fever, rash, joint pain, and conjunctivitis (pinkeye). Symptoms are usually mild, and start 2 to 7 days after being bitten by an infected mosquito. A blood test can tell whether you have the infection. There are no vaccines or medicines to treat it. Drinking lots of fluids, resting, and taking acetaminophen might help. Zika can cause microcephaly (a serious birth defect of the brain) and other problems in babies whose mothers were infected while pregnant. The Centers for Disease Control and Prevention recommends that pregnant women

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000019-3__dup2` (score=0.9850) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack...
2. `medquad_ans_0000034-12` (score=0.0337) - Heart failure is caused by other diseases and conditions that damage the heart muscle. It is most commonly caused by coronary artery disease, including heart attack. Diabetes and high blood pressure also contribute to...
3. `medquad_ans_0000033-14` (score=0.0126) - Very common. Each year, more than 1 million people in the U.S. have a heart attack and about half of them die. About one-half of those who die do so within 1 hour of the start of symptoms and before reaching the hospi...
4. `medquad_ans_0006111-3` (score=0.0122) - What causes Tietze syndrome? The exact underlying cause of Tietze syndrome is currently unknown. Some researchers have speculated that small injuries to the anterior chest wall may contribute to the development of the...
5. `medquad_ans_0000019-8` (score=0.0004) - Researchers are still learning about broken heart syndrome, and no treatments have been shown to prevent it. For people who have experienced the condition, the risk of recurrence is low. An emotionally upsetting or se...

**Answer**:

Broken heart syndrome is believed to be caused by extreme emotional or physical stress, which triggers a surge of fight or flight hormones such as adrenaline and noradrenaline. This hormone surge overwhelms and "stuns" the heart muscle, leading to temporary changes in heart muscle cells or coronary blood vessels. Potential causes include a hormone surge, coronary artery spasm, or microvascular dysfunction [medquad_ans_0000019-3__dup2].

**Supported citations**: `medquad_ans_0000019-3__dup2`
**Unsupported citations**: _none_

### Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Scores**: context_relevance=0.600, groundedness=0.618, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000005-1__dup5` (score=0.4499) - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. An autoimmune disease is a disorder in which the bodys immune system attacks the bodys own cells and...
2. `medquad_ans_0002675-1` (score=0.0146) - Graves' disease is an autoimmune disorder that leads to overactivity of the thyroid gland (hyperthyroidism). It is caused by an abnormal immune system response that causes the thyroid gland to produce too much thyroid...
3. `medquad_ans_0002735-1` (score=0.0051) - Hashimoto's encephalitis (HE) is a condition characterized by onset of confusion with altered level of consciousness; seizures; and jerking of muscles (myoclonus). Psychosis, including visual hallucinations and parano...
4. `medquad_ans_0000436-2` (score=0.0033) - Graves disease affects about 1 in 200 people. The disease occurs more often in women than in men, which may be related to hormonal factors. Graves disease is the most common cause of thyroid overactivity (hyperthyroid...
5. `medquad_ans_0000448-2` (score=0.0024) - Hashimoto thyroiditis affects 1 to 2 percent of people in the United States. It occurs more often in women than in men, which may be related to hormonal factors. The condition is the most common cause of thyroid under...

**Answer**:

Hashimoto thyroiditis, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease in which the body's immune system attacks the thyroid gland, causing inflammation and interfering with its ability to produce thyroid hormones. Large numbers of white blood cells called lymphocytes accumulate in the thyroid and produce antibodies that initiate the autoimmune process. Hashimoto thyroiditis often leads to reduced thyroid function, or hypothyroidism, a condition in which the thyroid does not produce enough thyroid hormone. Thyroid hormones regulate metabolism and affect nearly every organ in the body; without sufficient hormone levels, many bodily functions slow down. Hashimoto thyroiditis is the most common cause of hypothyroidism in the United States [medquad_ans_0000005-1]. It affects 1 to 2 percent of people in the United States [medquad_ans_0000448-2].

**Supported citations**: `medquad_ans_0000448-2`
**Unsupported citations**: `medquad_ans_0000005-1`

### Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Scores**: context_relevance=0.800, groundedness=0.980, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0006463-1` (score=0.6708) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include...
2. `medquad_ans_0000969-2` (score=0.0021) - Tangier disease is a rare disorder with approximately 100 cases identified worldwide. More cases are likely undiagnosed. This condition is named after an island off the coast of Virginia where the first affected indiv...
3. `medquad_ans_0001037-2` (score=0.0014) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
4. `medquad_ans_0000545-2` (score=0.0012) - Juvenile Batten disease is the most common type of NCL, but its exact prevalence is unknown. Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, wher...
5. `medquad_ans_0000002-2__dup1` (score=0.0009) - Enzyme replacement therapy for both Wolman's and cholesteryl ester storage disease is currently under investigation. Certain drugs may be given to help with adrenal gland production, and children may need to be fed in...

**Answer**:

Wolman disease is a type of lysosomal storage disorder that is an inherited condition causing a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include enlarged liver and spleen, poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor absorption of nutrients from food. It is caused by mutations in the LIPA gene and is inherited in an autosomal recessive manner. The condition is severe and life-threatening, although new therapies such as bone marrow transplantation have shown promise in improving the outlook of children with this disease. Enzyme replacement therapy is also being developed [medquad_ans_0006463-1].

**Supported citations**: `medquad_ans_0006463-1`
**Unsupported citations**: _none_

### Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002753-3` (score=0.9955) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma a...
2. `medquad_ans_0000006_3-2` (score=0.9927) - The cause of most childhood brain tumors is not known.
3. `medquad_ans_0000006_5-2` (score=0.9924) - The cause of most childhood brain tumors is unknown.
4. `medquad_ans_0000006_9-2` (score=0.9924) - The cause of most childhood brain tumors is unknown.
5. `medquad_ans_0000060-3__dup1` (score=0.0038) - Rebleeding from a cavernous angioma is common, it is not predictable, and individuals frequently have multiple CCMs found via magnetic resonance imaging. Individuals with CCM are faced with a diagnosis that imparts ri...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a possible relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

### Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Scores**: context_relevance=0.750, groundedness=0.927, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000655-3` (score=0.9980) - Microphthalmia may be caused by changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or...
2. `medquad_ans_0000593-3` (score=0.0024) - Mutations in at least two genes on the X chromosome are thought to be responsible for Lenz microphthalmia syndrome. Only one of these genes, BCOR, has been identified. The BCOR gene provides instructions for making a ...
3. `medquad_ans_0000631-2` (score=0.0001) - McCune-Albright syndrome occurs in between 1 in 100,000 and 1 in 1,000,000 people worldwide.
4. `medquad_ans_0000248-4` (score=0.0001) - Craniofacial microsomia most often occurs in a single individual in a family and is not inherited. If the condition is caused by a chromosomal abnormality, it may be inherited from one affected parent or it may result...
5. `medquad_ans_0000655-2` (score=0.0001) - Microphthalmia occurs in approximately 1 in 10,000 individuals.

**Answer**:

Genetic changes related to microphthalmia include mutations in many genes involved in the early development of the eye, most of which have not been identified. In the case of Lenz microphthalmia syndrome, mutations in at least two genes on the X chromosome are thought to be responsible, with only one gene, BCOR, having been identified. The BCOR gene provides instructions for making a protein called the BCL6 corepressor, which helps regulate the activity of other genes and appears to play an important role in early embryonic development. A mutation in the BCOR gene changes the structure of the BCL6 corepressor protein, disrupting normal eye development and other organ development before birth [medquad_ans_0000593-3]. Most genetic changes associated with isolated microphthalmia have been identified in only a small number of affected individuals [medquad_ans_0000655-3]. Additionally, microphthalmia

**Supported citations**: `medquad_ans_0000593-3`, `medquad_ans_0000655-3`
**Unsupported citations**: _none_

### Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Scores**: context_relevance=0.833, groundedness=0.739, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000203-7` (score=0.9979) - People with kidney stones may have pain while urinating, see blood in the urine, or feel a sharp pain in the back or lower abdomen. The pain may last for a short or long time. People may experience nausea and vomiting...
2. `medquad_ans_0000221-6` (score=0.9973) - You may have a kidney stone if you - have pain while urinating - see blood in your urine - feel a sharp pain in your back or lower abdomenthe area between your chest and hips The pain may last for a short or long time...
3. `medquad_ans_0000028-18` (score=0.0379) - The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH has more information about gout. Check out the information here. Also, see "Key Words" related to gout.
4. `medquad_ans_0000028-14` (score=0.0125) - To confirm a diagnosis of gout, the doctor inserts a needle into the inflamed joint and draws a sample of synovial fluid, the substance that lubricates a joint. A laboratory technician places some of the fluid on a sl...
5. `medquad_ans_0000028-15` (score=0.0029) - Physicians often prescribe high doses of non-steroidal anti-inflammatory drugs (NSAIDs) or steroids for a sudden attack of gout. NSAIDs are taken by mouth and corticosteroids are either taken by mouth or injected into...

**Answer**:

The provided context does not describe the symptoms of gout. It details the diagnostic process (e.g., synovial fluid analysis for uric acid crystals) and treatments (e.g., NSAIDs, colchicine, medications to lower uric acid levels) for gout, but does not list symptoms [medquad_ans_0000028-14][medquad_ans_0000028-15]. Therefore, based on the given information, the symptoms of gout cannot be answered.

**Supported citations**: `medquad_ans_0000028-14`, `medquad_ans_0000028-15`
**Unsupported citations**: _none_

### Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Scores**: context_relevance=0.333, groundedness=0.000, answer_relevance=0.333

**Top docs**:

1. `medquad_ans_0000112-3__dup1` (score=0.0006) - The initial symptoms can be very mild and may be noticeable only after prolonged exertion, stress, or fatigue. Dystonias often progress through various stages. Initially, dystonic movements are intermittent and appear...
2. `medquad_ans_0000363-2` (score=0.0002) - Familial paroxysmal kinesigenic dyskinesia is estimated to occur in 1 in 150,000 individuals. For unknown reasons, this condition affects more males than females.
3. `medquad_ans_0000971-2` (score=0.0001) - Task-specific focal dystonia affects an estimated 7 to 69 per million people in the general population. Musician's dystonia that is severe enough to impact performance occurs in about 1 percent of musicians.
4. `medquad_ans_0000704-2` (score=0.0001) - The prevalence of myoclonus-dystonia is unknown. This condition has been described in people worldwide.
5. `medquad_ans_0000299-2` (score=0.0001) - Dopa-responsive dystonia is estimated to affect 1 per million people worldwide. However, the disorder is likely underdiagnosed because the condition may not be identified in people with mild symptoms, or it may be mis...

**Answer**:

The provided context does not contain information about the cause of Dystonia 8. Therefore, the cause cannot be determined from the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Scores**: context_relevance=0.556, groundedness=0.976, answer_relevance=0.378

**Top docs**:

1. `medquad_ans_0000612-2` (score=0.9993) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unk...
2. `medquad_ans_0000554-2` (score=0.9298) - Kallmann syndrome is estimated to affect 1 in 10,000 to 86,000 people and occurs more often in males than in females. Kallmann syndrome 1 is the most common form of the disorder.
3. `medquad_ans_0000264-2` (score=0.7338) - Cystinuria affects approximately 1 in 10,000 people.
4. `medquad_ans_0000837-2` (score=0.0040) - Mild protein C deficiency affects approximately 1 in 500 individuals. Severe protein C deficiency is rare and occurs in an estimated 1 in 4 million newborns.
5. `medquad_ans_0000573-2` (score=0.0033) - Lactose intolerance in infancy resulting from congenital lactase deficiency is a rare disorder. Its incidence is unknown. This condition is most common in Finland, where it affects an estimated 1 in 60,000 newborns. A...

**Answer**:

Lysinuric protein intolerance affects 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations, the condition occurs less frequently, but the exact incidence is unknown [medquad_ans_0000612-2].

**Supported citations**: `medquad_ans_0000612-2`
**Unsupported citations**: _none_

### Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Scores**: context_relevance=0.375, groundedness=0.000, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000154-6` (score=0.9986) - Many babies with kidney dysplasia in only one kidney have no signs of the condition. In some cases, the affected kidney may be enlarged at birth and may cause pain.
2. `medquad_ans_0000123-3__dup1` (score=0.0204) - Individuals with Fabry disease often die prematurely of complications from strokes, heart disease, or kidney failure.
3. `medquad_ans_0000038-12` (score=0.0134) - Kidney disease is most often caused by diabetes or high blood pressure. Each kidney contains about one million tiny filters made up of blood vessels. These filters are called glomeruli. Diabetes and high blood pressur...
4. `medquad_ans_0000054-8` (score=0.0010) - The most common cause of P.A.D. is atherosclerosis, a buildup of plaque in the arteries. Over time, plaque can harden and narrow the arteries. This limits the flow of oxygen-rich blood to your organs and other parts o...
5. `medquad_ans_0000571-2` (score=0.0009) - LADD syndrome appears to be a rare condition; at least 60 cases have been described in the scientific literature.

**Answer**:

The provided context does not contain information about the symptoms of Renal Artery Stenosis. Therefore, based on the given text, it is not possible to answer the question. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Scores**: context_relevance=0.800, groundedness=0.772, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000712-5` (score=0.9908) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis -...
2. `medquad_ans_0002122-5` (score=0.4729) - How might epidermolysis bullosa acquisita be treated?
3. `medquad_ans_0000190-4__dup1` (score=0.0006) - The NINDS supports research on neurological disorders such as Melkersson-Rosenthal syndrome. Much of this research is aimed at increasing knowledge of these disorders and finding ways to treat, prevent, and ultimately...
4. `medquad_ans_0000510-5` (score=0.0001) - These resources address the diagnosis or management of ichthyosis with confetti: - Foundation for Ichthyosis and Related Skin Types (FIRST): Skin Care Tips - Foundation for Ichthyosis and Related Skin Types (FIRST): T...
5. `medquad_ans_0000083-4__dup1` (score=0.0001) - The NINDS conducts and supports a broad range of research on neurogenetic disorders, including Moebius syndrome. The goals of these studies are to develop improved techniques to diagnose, treat, and eventually cure th...

**Answer**:

The provided context does not specify the treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis. While it mentions that the Genetic Testing Registry includes resources for these conditions and that MedlinePlus offers information on diagnosis and management of various health conditions—including drug therapy, genetic counseling, and palliative care—no specific treatments are detailed [medquad_ans_0000712-5]. Therefore, based on the available information, the treatments for these syndromes cannot be determined.

**Supported citations**: `medquad_ans_0000712-5`
**Unsupported citations**: _none_

### Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.9994) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0000015-1__dup2` (score=0.5374) - The adrenal glands are small glands located on top of each kidney. They produce hormones that you can't live without, including sex hormones and cortisol. Cortisol helps you respond to stress and has many other import...
3. `medquad_ans_0000451-2` (score=0.0018) - At least 50 cases of Hennekam syndrome have been reported worldwide.
4. `medquad_ans_0000051-3__dup1` (score=0.0010) - The prognosis for individuals with orthostatic hypotension depends on the underlying cause of the condition.
5. `medquad_ans_0000109-2` (score=0.0003) - Bart-Pumphrey syndrome is a rare disorder; its exact prevalence is unknown. Only a few affected families and individual cases have been identified.

**Answer**:

The symptoms of Familial hyperaldosteronism type III include hypercalciuria (5%), metabolic acidosis (5%), polydipsia (5%), polyuria (5%), adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

**Supported citations**: `medquad_ans_0002259-1`
**Unsupported citations**: _none_

### Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Scores**: context_relevance=1.000, groundedness=0.952, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0005761-3` (score=0.9580) - How is spinocerebellar ataxia type 11 inherited? SCA11 is inherited in an autosomal dominant manner. The rate of de novo mutations is not known. Each child of an individual with SCA11 has a 50% chance of inheriting th...
2. `medquad_ans_0002439-4` (score=0.0721) - How is Friedreich ataxia inherited? Friedreich ataxia is inherited in an autosomal recessive manner. This means that to be affected, a person must have a mutation in both copies of the responsible gene in each cell. T...
3. `medquad_ans_0000033-3__dup1` (score=0.0625) - Average lifespan has been improving for years, for unknown reasons, and varies with the severity of the underlying mutations, ATM (ataxia-telangiectasia mutated) protein levels, and residual ATM kinase activity. Some ...
4. `medquad_ans_0000103-5` (score=0.0001) - These resources address the diagnosis or management of ARSACS: - Gene Review: Gene Review: ARSACS - Genetic Testing Registry: Spastic ataxia Charlevoix-Saguenay type These resources from MedlinePlus offer information ...
5. `medquad_ans_0000032-1__dup1` (score=0.0001) - Ataxia often occurs when parts of the nervous system that control movement are damaged. People with ataxia experience a failure of muscle control in their arms and legs, resulting in a lack of balance and coordination...

**Answer**:

Yes, Spinocerebellar ataxia type 11 (SCA11) is inherited in an autosomal dominant manner. Each child of an individual with SCA11 has a 50% chance of inheriting the mutation [medquad_ans_0005761-3].

**Supported citations**: `medquad_ans_0005761-3`
**Unsupported citations**: _none_

### Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Scores**: context_relevance=1.000, groundedness=0.935, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0003040-1` (score=0.9993) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload....
2. `medquad_ans_0001062-2` (score=0.0207) - This form of anemia is uncommon. However, researchers believe that it may not be as rare as they once thought. Increased awareness of the disease has led to more frequent diagnoses.
3. `medquad_ans_0000503-1` (score=0.0001) - Hypochromic microcytic anemia with iron overload is a condition that impairs the normal transport of iron in cells. Iron is an essential component of hemoglobin, which is the substance that red blood cells use to carr...
4. `medquad_ans_0000026-4` (score=0.0001) - African iron overload seems to run in families, and high iron in a family's diet seems to be the major contributor to development of the condition. There also may be a genetic contribution, but the inheritance pattern...
5. `medquad_ans_0000529-2` (score=0.0001) - Although iron deficiency anemia is relatively common, the prevalence of the iron-refractory form of the disease is unknown. At least 50 cases have been described in the medical literature. Researchers suspect that iro...

**Answer**:

The symptoms of Hypochromic microcytic anemia with iron overload include pale skin (pallor), tiredness (fatigue), and slow growth. Additionally, the condition leads to anemia characterized by abnormally small (microcytic) and pale (hypochromic) red blood cells. Iron that is not used by red blood cells accumulates in the liver, which can impair liver function over time, with liver problems typically becoming apparent in adolescence or early adulthood [medquad_ans_0000503-1].

**Supported citations**: `medquad_ans_0000503-1`
**Unsupported citations**: _none_

### Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000639-4` (score=0.9988) - This condition is inherited in an autosomal dominant pattern, which means one copy of an altered KRT12 or KRT3 gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the cond...
2. `medquad_ans_0000617-4` (score=0.9983) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0001001-4` (score=0.9979) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. However, some people who inherit the altered gene never develop ...
4. `medquad_ans_0000235-4` (score=0.9976) - This condition is most commonly inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each car...
5. `medquad_ans_0000796-2` (score=0.9965) - PRS superactivity is believed to be a rare disorder. Approximately 30 families with the condition have been reported. More than two thirds of these families are affected by the milder form of the disease.

**Answer**:

The provided context does not mention benign chronic pemphigus or its mode of inheritance. Therefore, it is not possible to determine based on the given information whether benign chronic pemphigus is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.960, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000107-4` (score=0.9988) - This condition is described as autosomal dominant, which means one copy of the altered gene in each cell is sufficient to cause the disorder. The condition almost always results from new (de novo) mutations in the ACT...
2. `medquad_ans_0000916-4__dup1` (score=0.9533) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is...
3. `medquad_ans_0001084-3__dup1` (score=0.0003) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
4. `medquad_ans_0000915-1__dup1` (score=0.0001) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
5. `medquad_ans_0000916-1__dup1` (score=0.0001) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...

**Answer**:

Yes, Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a mutation in only one copy of the responsible gene in each cell is sufficient to cause the condition. In some cases, the mutation is inherited from an affected parent, while in other cases, it occurs for the first time in the individual (de novo mutation) with no family history of the disorder [medquad_ans_0000916-4__dup1]. The condition is caused by a mutation in the TGFB1 gene, which is inherited in an autosomal dominant fashion [medquad_ans_0000915-1__dup1].

**Supported citations**: `medquad_ans_0000916-4__dup1`, `medquad_ans_0000915-1__dup1`
**Unsupported citations**: _none_

### Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.760, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000580-3` (score=0.0144) - Somatic mutations in the BRAF gene have been identified in the Langerhans cells of about half of individuals with Langerhans cell histiocytosis. Somatic gene mutations are acquired during a person's lifetime and are p...
2. `medquad_ans_0000352-3` (score=0.0002) - Familial hemophagocytic lymphohistiocytosis may be caused by mutations in any of several genes. These genes provide instructions for making proteins that help destroy or deactivate lymphocytes that are no longer neede...
3. `medquad_ans_0000324-4` (score=0.0001) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
4. `medquad_ans_0000483-2` (score=0.0001) - Histiocytosis-lymphadenopathy plus syndrome is a rare disorder, affecting approximately 100 individuals worldwide.
5. `medquad_ans_0000136-4` (score=0.0001) - BOR/BO syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In about 90 percent of cases, an affected person inherits the m...

**Answer**:

The provided context does not contain information about the genetic changes related to histiocytosis-lymphadenopathy plus syndrome. [medquad_ans_0000483-2] mentions that the condition is rare and affects approximately 100 individuals worldwide, but it does not specify any genetic changes associated with the syndrome. Therefore, based on the available information, the genetic changes related to histiocytosis-lymphadenopathy plus syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000483-2`
**Unsupported citations**: _none_

### Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.933, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0006446-4` (score=0.9827) - Is Williams syndrome inherited?
2. `medquad_ans_0004450-3` (score=0.9363) - How is Noonan syndrome inherited? Noonan syndrome is inherited in an autosomal dominant manner. This means that having one changed (mutated) copy of the responsible gene in each cell is enough to cause the condition. ...
3. `medquad_ans_0004451-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004456-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004450-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

Yes, Noonan syndrome is typically inherited in an autosomal dominant manner, meaning that having one changed (mutated) copy of the responsible gene in each cell is enough to cause the condition. Each child of a person with Noonan syndrome has a 50% (1 in 2) chance to inherit the condition. However, many cases are due to a new mutation and are not inherited from an affected parent [medquad_ans_0004450-3][medquad_ans_0004451-1][medquad_ans_0004456-1].

**Supported citations**: `medquad_ans_0004450-3`, `medquad_ans_0004451-1`, `medquad_ans_0004456-1`
**Unsupported citations**: _none_

### Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Scores**: context_relevance=0.500, groundedness=0.666, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001093-5` (score=0.9952) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Aci...
2. `medquad_ans_0001092-5` (score=0.9949) - These resources address the diagnosis or management of 3-MCC deficiency: - Baby's First Test - Genetic Testing Registry: 3 Methylcrotonyl-CoA carboxylase 1 deficiency - Genetic Testing Registry: 3-methylcrotonyl CoA c...
3. `medquad_ans_0001089-5` (score=0.2636) - These resources address the diagnosis or management of HMG-CoA lyase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of hydroxymethylglutaryl-CoA lyase These resources from MedlinePlus offer inf...
4. `medquad_ans_0000244-5` (score=0.0382) - These resources address the diagnosis or management of Costeff syndrome: - Baby's First Test - Gene Review: Gene Review: OPA3-Related 3-Methylglutaconic Aciduria - Genetic Testing Registry: 3-Methylglutaconic aciduria...
5. `medquad_ans_0000960-5` (score=0.0039) - These resources address the diagnosis or management of succinyl-CoA:3-ketoacid CoA transferase deficiency: - Genetic Testing Registry: Succinyl-CoA acetoacetate transferase deficiency - MedlinePlus Encyclopedia: Keton...

**Answer**:

The provided context does not specify any treatments for 3-methylglutaconyl-CoA hydratase deficiency. While resources such as "Diagnostic Tests," "Genetic Counseling," and "Palliative Care" are mentioned in relation to the diagnosis and management of various conditions, including 3-methylglutaconic aciduria, no specific therapeutic interventions are detailed for 3-methylglutaconyl-CoA hydratase deficiency [medquad_ans_0001093-5]. Therefore, based on the given information, the treatments for this deficiency cannot be determined.

**Supported citations**: `medquad_ans_0001093-5`
**Unsupported citations**: _none_

### Example 61 (`0003000-1`)

**Question**: What is (are) Hyperkalemic periodic paralysis ?

**Scores**: context_relevance=0.833, groundedness=0.994, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000168-3__dup1` (score=0.9736) - KSS is a slowly progressive disorder. The prognosis for individuals with KSS varies depending on the severity and the number of organs involved. Early diagnosis and periodic electrocardiogram (ECG) are important since...
2. `medquad_ans_0003000-1` (score=0.9120) - Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move...
3. `medquad_ans_0000505-1` (score=0.7986) - Hypokalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness typically beginning in childhood or adolescence. Most often, these episodes involve a temporary inability to move muscles i...
4. `medquad_ans_0000494-1` (score=0.7255) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to...
5. `medquad_ans_0003059-1` (score=0.6926) - Hypokalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness typically beginning in childhood or adolescence. Most often, these episodes involve a temporary inability to move muscles i...

**Answer**:

Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms and legs. Episodes tend to increase in frequency until about age 25, after which they may occur less frequently. Factors that can trigger attacks include rest after exercise, potassium-rich foods, stress, fatigue, and long periods without food. Muscle strength improves between attacks, although many affected people continue to experience mild stiffness, particularly in muscles of the face and hands. This condition is caused by mutations in the SCN4A gene and is inherited in an autosomal dominant fashion [medquad_ans_0003000-1].

**Supported citations**: `medquad_ans_0003000-1`
**Unsupported citations**: _none_

### Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Scores**: context_relevance=0.700, groundedness=0.757, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0000054-2__dup1` (score=0.9968) - Generally treatment for individuals with BSS focuses on the underlying cause of the disorder. Early treatment with high-dose steroids may be beneficial in many cases. Other treatment is symptomatic and supportive.
2. `medquad_ans_0000225-5` (score=0.9944) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of...
3. `medquad_ans_0000171-4__dup1` (score=0.0130) - The NINDS supports research on congenital disorders such as KTS with the goal of finding new means to treat and prevent them.
4. `medquad_ans_0000746-5` (score=0.0026) - These resources address the diagnosis or management of oculofaciocardiodental syndrome: - Genetic Testing Registry: Oculofaciocardiodental syndrome These resources from MedlinePlus offer information about the diagnosi...
5. `medquad_ans_0000083-3__dup1` (score=0.0018) - There is no cure for Moebius syndrome. In spite of the impairments that characterize the disorder, proper care and treatment give many individuals a normal life expectancy.

**Answer**:

The provided context does not specify the exact treatments for congenital fibrosis of the extraocular muscles. It mentions that resources are available for the diagnosis or management of this condition, including genetic testing and information on diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care [medquad_ans_0000225-5]. However, no specific treatment details are given. Therefore, based on the available information, the treatments for congenital fibrosis of the extraocular muscles cannot be determined.

**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

### Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Scores**: context_relevance=0.800, groundedness=0.508, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000491-1__dup2` (score=0.6731) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms...
2. `medquad_ans_0002944-1` (score=0.4746) - Huntington disease (HD) is an inherited condition that causes progressive degeneration of neurons in the brain. Signs and symptoms usually develop between ages 35 to 44 years and may include uncontrolled movements, lo...
3. `medquad_ans_0000098-3__dup1` (score=0.0155) - There are many disorders that can cause dementia. Some, such as Alzheimers disease or Huntingtons disease, lead to a progressive loss of mental functions. But other types of dementia can be halted or reversed with app...
4. `medquad_ans_0000152-3__dup1` (score=0.0110) - Huntingtons disease causes disability that gets worse over time. People with this disease usually die within 15 to 20 years following diagnosis. At this time, no treatment is available to slow, stop or reverse the cou...
5. `medquad_ans_0002944-5` (score=0.0062) - Is genetic testing available for Huntington disease? Yes. Testing of adults at risk for Huntington disease (HD) who have no symptoms of the disease is called predictive testing. Whether to have predictive testing requ...

**Answer**:

Huntington's disease (HD) is an inherited condition that causes progressive degeneration of neurons in the brain. It is caused by changes (mutations) in the HTT gene and is inherited in an autosomal dominant manner. Signs and symptoms usually develop between ages 35 to 44 years and may include uncontrolled movements, loss of intellectual abilities, and various emotional and psychiatric problems. Early symptoms may include uncontrolled movements, clumsiness, and balance problems, while later symptoms can involve loss of the ability to walk, talk, and swallow, and in some cases, a loss of recognition of family members. People with HD usually live for about 15 to 20 years after the condition begins. There is no cure, and while medicines can help manage symptoms, they cannot slow down or stop the disease. A blood test can detect the presence of the HD gene, and genetic counseling is recommended to help individuals weigh the risks and benefits

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=0.778, groundedness=0.950, answer_relevance=0.511

**Top docs**:

1. `medquad_ans_0000711-2` (score=0.9993) - N-acetylglutamate synthase deficiency is a very rare disorder. Only a few cases have been reported worldwide, and the overall incidence is unknown.
2. `medquad_ans_0000419-2` (score=0.9993) - Glutathione synthetase deficiency is very rare. This disorder has been described in more than 70 people worldwide.
3. `medquad_ans_0000066-2` (score=0.9992) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
4. `medquad_ans_0000416-2` (score=0.9992) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
5. `medquad_ans_0000442-2` (score=0.9986) - Guanidinoacetate methyltransferase deficiency is a very rare disorder. About 80 affected individuals have been described in the medical literature. Of these, approximately one-third are of Portuguese origin.

**Answer**:

Approximately 20 affected individuals have been identified with glutamate formiminotransferase deficiency [medquad_ans_0000416-2].

**Supported citations**: `medquad_ans_0000416-2`
**Unsupported citations**: _none_

### Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.931, answer_relevance=0.475

**Top docs**:

1. `medquad_ans_0001036-2` (score=0.9993) - Williams syndrome affects an estimated 1 in 7,500 to 10,000 people.
2. `medquad_ans_0000738-2` (score=0.9992) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
3. `medquad_ans_0000058-2` (score=0.9991) - Angelman syndrome affects an estimated 1 in 12,000 to 20,000 people.
4. `medquad_ans_0004453-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004452-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people [medquad_ans_0000738-2].

**Supported citations**: `medquad_ans_0000738-2`
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

**Scores**: context_relevance=1.000, groundedness=0.934, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0005571-3` (score=0.9688) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
2. `medquad_ans_0000860-2__dup1` (score=0.9474) - How might Brown-Sequard syndrome be treated?
3. `medquad_ans_0006423-3` (score=0.9413) - How might Wernicke-Korsakoff syndrome be treated?
4. `medquad_ans_0000124-2__dup1` (score=0.1886) - There is no cure for Fahr's Syndrome, nor is there a standard course of treatment. Treatment addresses symptoms on an individual basis.
5. `medquad_ans_0000275-2` (score=0.0175) - There is no cure for Williams syndrome, nor is there a standard course of treatment. Because WS is an uncommon and complex disorder, multidisciplinary clinics have been established at several centers in the United Sta...

**Answer**:

Treatment options attempted for Shapiro syndrome include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, or sympathectomy. However, due to the limited number of diagnosed cases and the periodic nature of the disease, evaluating effective treatment options is difficult. It is recommended that treatment be discussed with a healthcare provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

### Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.514, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000373-1` (score=0.6256) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects ...
2. `medquad_ans_0000407-2` (score=0.0038) - Gilbert syndrome is a common condition that is estimated to affect 3 to 7 percent of Americans.
3. `medquad_ans_0000385-2` (score=0.0015) - Fraser syndrome affects an estimated 1 in 200,000 newborns. The condition occurs in approximately 1 in 10,000 fetuses that do not survive to birth.
4. `medquad_ans_0000954-2` (score=0.0014) - Stickler syndrome affects an estimated 1 in 7,500 to 9,000 newborns. Type I is the most common form of the condition.
5. `medquad_ans_0000912-2` (score=0.0014) - Although Silver syndrome appears to be a rare condition, its exact prevalence is unknown.

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. It affects intelligence and behavior, with almost everyone having intellectual disability ranging from mild to severe. Affected individuals tend to be friendly, inquisitive, and hyperactive, with a short attention span. Socialization and daily living skills are generally strong, while verbal communication and language skills are typically weaker. Physical features include weak muscle tone (hypotonia), broad thumbs, wide first (big) toes, and a distinctive facial appearance such as small underdeveloped ears, a tall prominent forehead, and down-slanting palpebral fissures. Abnormalities of the corpus callosum are common, and most individuals have constipation, with some having an imperforate anus. Other reported health issues include heart defects, seizures, undescended testes (cryptorch

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.036, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000172-3__dup1` (score=0.0025) - There is no cure for Klver-Bucy syndrome. The disorder is not life-threatening, but the patient can be difficult to manage. With treatment, symptoms may slowly decline.
2. `medquad_ans_0000634-2` (score=0.0006) - Meckel syndrome affects 1 in 13,250 to 1 in 140,000 people worldwide. It is more common in certain populations; for example, the condition affects about 1 in 9,000 people of Finnish ancestry and about 1 in 3,000 peopl...
3. `medquad_ans_0000553-2` (score=0.0002) - Kabuki syndrome occurs in approximately 1 in 32,000 newborns.
4. `medquad_ans_0000684-2` (score=0.0002) - Muenke syndrome occurs in about 1 in 30,000 newborns. This condition accounts for an estimated 8 percent of all cases of craniosynostosis.
5. `medquad_ans_0000560-2` (score=0.0001) - The prevalence of Kleefstra syndrome is unknown. Only recently has testing become available to distinguish it from other disorders with similar features.

**Answer**:

The provided context does not mention "Tukel syndrome" or describe any symptoms associated with it. Therefore, no information can be given about the symptoms of Tukel syndrome based on the available content. [No relevant context]

**Supported citations**: _none_
**Unsupported citations**: `No relevant context`

### Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Scores**: context_relevance=0.333, groundedness=0.479, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0000674-5` (score=0.9859) - These resources address the diagnosis or management of mucolipidosis II alpha/beta: - Gene Review: Gene Review: Mucolipidosis II - Genetic Testing Registry: I cell disease - MedlinePlus Encyclopedia: Clubfoot - Medlin...
2. `medquad_ans_0000675-5` (score=0.8807) - These resources address the diagnosis or management of mucolipidosis III alpha/beta: - Gene Review: Gene Review: Mucolipidosis III Alpha/Beta - Genetic Testing Registry: Pseudo-Hurler polydystrophy - MedlinePlus Encyc...
3. `medquad_ans_0005612-3` (score=0.0981) - How might sialidosis type I be treated? There is no specific treatment for sialidosis. Management should be multidisciplinary and directed at supportive care and symptomatic relief. Overall health maintenance should b...
4. `medquad_ans_0000002-3__dup1` (score=0.0576) - Wolmans disease is usually fatal by age 1. The onset and course of cholesteryl ester storage disease varies, and individuals may live into adulthood.
5. `medquad_ans_0000676-5` (score=0.0573) - These resources address the diagnosis or management of mucolipidosis III gamma: - Gene Review: Gene Review: Mucolipidosis III Gamma - Genetic Testing Registry: Mucolipidosis III Gamma - MedlinePlus Encyclopedia: Cloud...

**Answer**:

There is no specific treatment for mucolipidosis II alpha/beta, mucolipidosis III alpha/beta, or mucolipidosis III gamma. Management is primarily supportive and directed at symptomatic relief, including diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care [medquad_ans_0000674-5, medquad_ans_0000675-5, medquad_ans_0000676-5]. For sialidosis type I, which is a related condition, there is no specific treatment, and management focuses on supportive care and symptomatic relief, such as seizure control [medquad_ans_0005612-3]. However, specific treatment details for mucolipidosis types are not provided in the context.

**Supported citations**: `medquad_ans_0005612-3`
**Unsupported citations**: `medquad_ans_0000674-5, medquad_ans_0000675-5, medquad_ans_0000676-5`

### Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.953, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0005084-2` (score=0.9994) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis...
2. `medquad_ans_0000824-2` (score=0.0339) - An estimated 1 in 10,000 people have primary sclerosing cholangitis, and the condition is diagnosed in approximately 1 in 100,000 people per year worldwide.
3. `medquad_ans_0000183-2__dup3` (score=0.0002) - Biliary atresia is rare and only affects about one out of every 18,000 infants.1 The disease is more common in females, premature babies, and children of Asian or African American heritage.
4. `medquad_ans_0001037-2` (score=0.0001) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
5. `medquad_ans_0005084-1` (score=0.0001) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching...

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 (PFIC2) include severe itching (pruritus), jaundice (intermittent jaundice), failure to thrive, portal hypertension, hepatosplenomegaly (enlarged liver and spleen), conjugated hyperbilirubinemia, diarrhea, fat malabsorption, hepatomegaly, and infantile onset [medquad_ans_0005084-2]. These symptoms typically develop during infancy and may progress to liver failure in the first few years of life [medquad_ans_0005084-1].

**Supported citations**: `medquad_ans_0005084-2`, `medquad_ans_0005084-1`
**Unsupported citations**: _none_

### Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Scores**: context_relevance=0.600, groundedness=0.958, answer_relevance=0.440

**Top docs**:

1. `medquad_ans_0000211-6` (score=0.9968) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he i...
2. `medquad_ans_0000061-5__dup3` (score=0.9840) - Risk factors are conditions that increase the chances of getting a particular disease. The more risk factors people have, the greater their chances of developing that disease or condition. Diabetic neuropathy and rela...
3. `medquad_ans_0000219-4__dup3` (score=0.0513) - Erectile dysfunction affects men of all races and in all parts of the world. Men are more likely to have ED as they get older. For example, ED occurs in - about 12 percent of men younger than 60 - 22 percent of men ag...
4. `medquad_ans_0000213-4__dup2` (score=0.0061) - Urinary retention in men becomes more common with age. - In men 40 to 83 years old, the overall incidence of urinary retention is 4.5 to 6.8 per 1,000 men.2 - For men in their 70s, the overall incidence increases to 1...
5. `medquad_ans_0000219-5__dup2` (score=0.0019) - Having ED can cause you to feel depressed or anxious. ED may also cause low self-esteem. When you have ED, you may not have a satisfying sex life. You may not feel as close with your sexual partner, which may strain y...

**Answer**:

Urinary incontinence occurs in 11 to 34 percent of older men [medquad_ans_0000211-6]. Two to 11 percent of older men report daily urinary incontinence [medquad_ans_0000211-6].

**Supported citations**: `medquad_ans_0000211-6`
**Unsupported citations**: _none_

### Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.532, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000882-1` (score=0.2402) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
2. `medquad_ans_0000979-2` (score=0.0010) - TAR syndrome is a rare disorder, affecting fewer than 1 in 100,000 newborns.
3. `medquad_ans_0000135-2` (score=0.0006) - Branchio-oculo-facial syndrome is a rare condition, although the prevalence is unknown.
4. `medquad_ans_0000964-2` (score=0.0004) - Swyer syndrome occurs in approximately 1 in 80,000 people.
5. `medquad_ans_0000881-2` (score=0.0002) - Roberts syndrome is a rare disorder; approximately 150 affected individuals have been reported.

**Answer**:

Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome, distinguished by the severity of their signs and symptoms and by their pattern of inheritance—autosomal recessive or autosomal dominant. Autosomal recessive Robinow syndrome is characterized by skeletal abnormalities such as shortening of the long bones in the arms and legs, particularly the forearms; abnormally short fingers and toes (brachydactyly); wedge-shaped spinal bones (hemivertebrae) leading to abnormal spinal curvature (kyphoscoliosis); fused or missing ribs; and short stature. Affected individuals also have distinctive facial features, including a broad forehead, prominent and widely spaced eyes, a short nose with an upturned tip, a wide nasal bridge, and a broad, triangle-shaped mouth, which are sometimes described as "fetal facies" due to their resemblance

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 74 (`0000551-1`)

**Question**: What is (are) juvenile primary lateral sclerosis ?

**Scores**: context_relevance=0.857, groundedness=0.986, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000551-1` (score=0.9132) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. The features of this disorder are caused by damage to moto...
2. `medquad_ans_0003331-1` (score=0.8675) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
3. `medquad_ans_0000055-3__dup1` (score=0.2352) - Kennedy's disease is slowly progressive. Individuals tend to remain ambulatory until late in the disease, although some may be wheelchair-bound during later stages. The life span of individuals with Kennedy's disease ...
4. `medquad_ans_0000238-1__dup1` (score=0.1857) - Primary lateral sclerosis (PLS) is a rare neuromuscular disease with slowly progressive weakness in voluntary muscle movement. PLS belongs to a group of disorders known as motor neuron diseases. PLS affects the upper ...
5. `medquad_ans_0003319-1` (score=0.0260) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. U...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. It is caused by damage to motor neurons, which are specialized nerve cells in the brain and spinal cord that control muscle movement. Symptoms begin in early childhood and progress slowly over many years. Early symptoms include clumsiness, muscle weakness, and spasticity in the legs, and difficulty with balance. As the disorder progresses, spasticity spreads to the arms and hands, and individuals may develop slurred speech, drooling, difficulty swallowing, and an inability to walk. Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene and is inherited in an autosomal recessive pattern [medquad_ans_0000551-1][medquad_ans_0003331-1].

**Supported citations**: `medquad_ans_0000551-1`, `medquad_ans_0003331-1`
**Unsupported citations**: _none_

### Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Scores**: context_relevance=0.750, groundedness=0.935, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000040-11` (score=0.9829) - There are four common types of leukemia. They are chronic lymphocytic leukemia, chronic myeloid leukemia, acute myeloid leukemia, and acute lymphocytic leukemia. Chronic lymphocytic leukemia, chronic myeloid leukemia,...
2. `medquad_ans_0000040-9` (score=0.9663) - Leukemia is a cancer of the blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white ...
3. `medquad_ans_0000554-1__dup2` (score=0.7928) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells c...
4. `medquad_ans_0000040-1__dup3` (score=0.4616) - Leukemia is a cancer of the blood cells. It is the most common type of blood cancer and affects 10 times as many adults as children. Most people diagnosed with leukemia are over 50 years old. Leukemia Starts in Bone M...
5. `medquad_ans_0000040-5__dup1` (score=0.0984) - Unlike other types of cancer, leukemia isn't a tumor that your doctor can surgically remove. Leukemia cells are produced in the bone marrow and travel throughout the body. The Goal of Treatment The goal of treatment f...

**Answer**:

Leukemia is a cancer of the blood cells that usually begins in the bone marrow, where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. These abnormal cells crowd out healthy blood cells, making it difficult for the blood to carry out its normal functions [medquad_ans_0000040-9]. It is the most common type of blood cancer and affects 10 times as many adults as children, with most diagnoses occurring in individuals over 50 years old [medquad_ans_0000040-1__dup3]. There are four common types of leukemia: chronic lymphocytic leukemia, chronic myeloid leukemia, acute myeloid leukemia, and acute lymphocytic leukemia [medquad_ans_0000040-11]. Chronic leukemia grows slowly, while acute leukemia involves rapid increases in very abnormal cells, with children most often having

**Supported citations**: `medquad_ans_0000040-9`, `medquad_ans_0000040-1__dup3`, `medquad_ans_0000040-11`
**Unsupported citations**: _none_

### Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Scores**: context_relevance=0.750, groundedness=0.972, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0002485-1` (score=0.7954) - Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. Gangliocytomas may occur in all age groups but most often occur in people between the ages of 10 and 30. The most common si...
2. `medquad_ans_0000006_1-3` (score=0.0020) - The cause of most adult brain and spinal cord tumors is not known.
3. `medquad_ans_0000006_3-2` (score=0.0017) - The cause of most childhood brain tumors is not known.
4. `medquad_ans_0000006_5-2` (score=0.0013) - The cause of most childhood brain tumors is unknown.
5. `medquad_ans_0000006_9-2` (score=0.0013) - The cause of most childhood brain tumors is unknown.

**Answer**:

Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. It may occur in all age groups but most often occurs in people between the ages of 10 and 30. The most common site is the temporal lobe of the brain, but it can arise anywhere in the CNS, including the cerebellum, brainstem, floor of the third ventricle, and spinal cord. Gangliocytomas are among the most frequent tumors associated with epilepsy. Signs and symptoms depend on the tumor's location and may include seizures (most commonly), increased brain pressure, endocrine disorders, and focal symptoms. These tumors are generally slow-growing and usually do not become malignant. Treatment involves surgical removal of the tumor [medquad_ans_0002485-1].

**Supported citations**: `medquad_ans_0002485-1`
**Unsupported citations**: _none_

### Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.985, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000130-1__dup1` (score=0.7352) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of ...
2. `medquad_ans_0000229-3__dup1` (score=0.0052) - In acute neuropathies, such as Guillain-Barr syndrome, symptoms appear suddenly, progress rapidly, and resolve slowly as damaged nerves heal. In chronic forms, symptoms begin subtly and progress slowly. Some people ma...
3. `medquad_ans_0000267-4__dup1` (score=0.0018) - The National Institute of Neurological Disorders and Stroke (NINDS) and other institutes at the National Institutes of Health support research in TOS through grants to major medical research institutions across the co...
4. `medquad_ans_0000226-4__dup1` (score=0.0013) - The NINDS supports research on neurological disorders such as Parry-Romberg syndrome with the goal of finding ways to prevent, treat, and cure them.
5. `medquad_ans_0000130-4__dup1` (score=0.0011) - The NINDS supports research aimed at discovering new ways to diagnose, treat, and, ultimately, cure neuropathies such as Miller Fisher syndrome.

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of tendon reflexes. Like Guillain-Barr syndrome, symptoms may be preceded by a viral illness. Additional symptoms include generalized muscle weakness and respiratory failure. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder [medquad_ans_0000130-1__dup1]. The National Institute of Neurological Disorders and Stroke (NINDS) supports research aimed at discovering new ways to diagnose, treat, and, ultimately, cure neuropathies such as Miller Fisher syndrome [medquad_ans_0000130-4__dup1].

**Supported citations**: `medquad_ans_0000130-1__dup1`, `medquad_ans_0000130-4__dup1`
**Unsupported citations**: _none_

### Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.538, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006570-3` (score=0.9947) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location desig...
2. `medquad_ans_0001086-2` (score=0.0002) - At least 500 cases of 22q13.3 deletion syndrome are known.
3. `medquad_ans_0001084-3` (score=0.0001) - Most people with 22q11.2 deletion syndrome are missing a sequence of about 3 million DNA building blocks (base pairs) on one copy of chromosome 22 in each cell. This region contains 30 to 40 genes, many of which have ...
4. `medquad_ans_0001087-2` (score=0.0001) - 2q37 deletion syndrome appears to be a rare condition, although its exact prevalence is unknown. Approximately 100 cases have been reported worldwide.
5. `medquad_ans_0001084-2` (score=0.0001) - 22q11.2 deletion syndrome affects an estimated 1 in 4,000 people. However, the condition may actually be more common than this estimate because doctors and researchers suspect it is underdiagnosed due to its variable ...

**Answer**:

22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell, specifically near the middle of the chromosome at a location designated q11.2. Most people with this syndrome are missing a sequence of about 3 million DNA building blocks (base pairs) on one copy of chromosome 22, which contains about 30 to 40 genes, many of which have not been well characterized. A small percentage of affected individuals have smaller deletions. The deletion of a particular gene, TBX1, is probably responsible for many of the syndrome's characteristic signs, such as heart defects, a cleft palate, distinctive facial features, hearing loss, and low calcium levels. Loss of this gene may also contribute to behavioral problems. The loss of another gene, COMT, may also increase the risk of behavioral problems and mental illness. The loss of additional genes in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Scores**: context_relevance=1.000, groundedness=0.902, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000062-12` (score=0.9888) - Shingles is caused by a virus called the varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores and other symptoms healed, but the virus remained....
2. `medquad_ans_0000062-2__dup3` (score=0.9619) - Caused By A Virus Shingles is caused by a virus called varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores healed and the other symptoms went a...
3. `medquad_ans_0000148-3__dup2` (score=0.0060) - For most healthy people who receive treatment soon after the outbreak of blisters, the lesions heal, the pain subsides within 3 to 5 weeks, and the blisters often leave no scars. However, shingles is a serious threat ...
4. `medquad_ans_0000062-1__dup5` (score=0.0012) - Shingles is a painful rash that develops on one side of the face or body. The rash forms blisters that typically scab over in 7 to 10 days and clear up within 2 to 4 weeks. Most commonly, the rash occurs in a single s...
5. `medquad_ans_0000149-4__dup1` (score=0.0006) - The NINDS supports research on shingles and shingles-related conditions. Current studies focus on the relationship between the persistence of neurotropic viruses and development of neurological diseases including herp...

**Answer**:

Shingles is caused by the varicella-zoster virus, the same virus that causes chickenpox. After recovering from chickenpox, the virus remains in the body for life, hiding in nerve cells, usually in the spine. It can become active again when the immune system weakens, which may happen due to aging, certain medical conditions (such as cancer, HIV/AIDS, or transplant operations), or stress. When reactivated, the virus travels along a nerve path (dermatome) to the skin, causing a painful rash [medquad_ans_0000062-12][medquad_ans_0000062-2__dup3].

**Supported citations**: `medquad_ans_0000062-12`, `medquad_ans_0000062-2__dup3`
**Unsupported citations**: _none_

### Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.925, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0002456-1` (score=0.9994) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the ta...
2. `medquad_ans_0000123-3__dup1` (score=0.0043) - Individuals with Fabry disease often die prematurely of complications from strokes, heart disease, or kidney failure.
3. `medquad_ans_0000124-2__dup1` (score=0.0029) - There is no cure for Fahr's Syndrome, nor is there a standard course of treatment. Treatment addresses symptoms on an individual basis.
4. `medquad_ans_0000568-2` (score=0.0001) - Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, where approximately 1 in 12,500 individuals have the condition. Kufs disease is thought to repres...
5. `medquad_ans_0000556-2` (score=0.0001) - KBG syndrome is a rare disorder that has been reported in around 60 individuals. For unknown reasons, males are affected more often than females. Doctors think the disorder is underdiagnosed because the signs and symp...

**Answer**:

The symptoms of Fryns Hofkens Fabry syndrome include ulnar deviation of the finger (seen in 90% of patients), distal ulnar hypoplasia, dysplastic radii, hypoplasia of the radius, mesomelic arm shortening, radial bowing, and ulnar deviation of the hand. These signs and symptoms are based on data from the Human Phenotype Ontology, which compiles information on the frequency of features in this condition [medquad_ans_0002456-1].

**Supported citations**: `medquad_ans_0002456-1`
**Unsupported citations**: _none_

### Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Scores**: context_relevance=1.000, groundedness=0.810, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000207-1__dup2` (score=0.9963) - Summary : Clinical trials are research studies that test how well new medical approaches work in people. Each study answers scientific questions and tries to find better ways to prevent, screen for, diagnose, or treat...
2. `medquad_ans_0000019_3-7` (score=0.9917) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, ta...
3. `medquad_ans_0000050-8` (score=0.9897) - Scientists are pursuing a wide range of basic and clinical studies on osteoporosis. Significant advances in preventing and treating osteoporosis continue to be made. Such advances are the direct result of research foc...
4. `medquad_ans_0000002-14` (score=0.9449) - Some people with Addisons disease who are aldosterone deficient can benefit from following a diet rich in sodium. A health care provider or a dietitian can give specific recommendations on appropriate sodium sources a...
5. `medquad_ans_0000027_4-4` (score=0.0006) - Cancer prevention clinical trials are used to study ways to prevent cancer. Cancer prevention clinical trials are used to study ways to lower the risk of developing certain types of cancer. Some cancer prevention tria...

**Answer**:

The provided context does not mention any specific research or clinical trials being conducted for osteoporosis. Therefore, based on the given information, it is not possible to answer the question. [medquad_ans_0000207-1__dup2] [medquad_ans_0000019_3-7]

**Supported citations**: `medquad_ans_0000207-1__dup2`, `medquad_ans_0000019_3-7`
**Unsupported citations**: _none_

### Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Scores**: context_relevance=0.857, groundedness=0.910, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000338-1__dup1` (score=0.9973) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the s...
2. `medquad_ans_0000430-1__dup1` (score=0.9966) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. ...
3. `medquad_ans_0000693-1__dup2` (score=0.9966) - Summary : As a patient, you have certain rights. Some are guaranteed by federal law, such as the right to get a copy of your medical records, and the right to keep them private. Many states have additional laws protec...
4. `medquad_ans_0000694-1__dup1` (score=0.9945) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of me...
5. `medquad_ans_0000067-17` (score=0.9927) - It is important to keep a copy of your medical records to share with any new doctors that you see. This information should contain the type of cancer you were diagnosed with, test results, and treatment details. It is...

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. It often includes red flags such as "miracle cure," "quick fix," "ancient remedy," "secret ingredient," or "scientific breakthrough." These scams can be dangerous, waste money, and may prevent individuals from receiving necessary treatments. The Food and Drug Administration (FDA) warns that such practices are common, especially for conditions like cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. Before taking unproven treatments, it is recommended to consult a doctor or healthcare professional, especially when considering prescription drugs [medquad_ans_0000430-1__dup1].

**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

### Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Scores**: context_relevance=0.833, groundedness=0.306, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000055-3__dup2` (score=0.9946) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision -...
2. `medquad_ans_0000043-2__dup4` (score=0.9731) - Many people have no visible signs or symptoms of diabetes. Symptoms can also be so mild that you might not notice them. More than 5 million people in the United States have type 2 diabetes and do not know it. - increa...
3. `medquad_ans_0000343-1__dup1` (score=0.6822) - Some eye problems are minor and don't last long. But some can lead to a permanent loss of vision. Common eye problems include - Refractive errors - Cataracts - clouded lenses - Glaucoma - a disorder caused by damage t...
4. `medquad_ans_0000275-1__dup1` (score=0.0254) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
5. `medquad_ans_0000022-4__dup3` (score=0.0050) - The signs and symptoms of type 2 diabetes can be so mild that you might not even notice them. Nearly 7 million people in the United States have type 2 diabetes and dont know they have the disease. Many have no signs o...

**Answer**:

The provided context does not contain information about symptoms of preventing diabetes problems or specifically about "keeping your eyes healthy" in terms of preventive symptoms. The context discusses symptoms of diabetes-related eye problems, such as blurry or double vision, flashing lights, dark or floating spots, and pain or pressure in the eyes, but does not outline symptoms associated with prevention of diabetes eye problems. Therefore, based on the given information, it is not possible to answer the question.

Answer: Insufficient context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.900, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000107-4` (score=0.9988) - This condition is described as autosomal dominant, which means one copy of the altered gene in each cell is sufficient to cause the disorder. The condition almost always results from new (de novo) mutations in the ACT...
2. `medquad_ans_0000392-4` (score=0.9972) - Fryns syndrome appears to be inherited in an autosomal recessive pattern, which means both copies of a gene in each cell have mutations. However, no associated gene has been identified. The parents of an individual wi...
3. `medquad_ans_0000001-4` (score=0.9843) - Aarskog-Scott syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chro...
4. `medquad_ans_0006446-4` (score=0.9560) - Is Williams syndrome inherited?
5. `medquad_ans_0000108-4` (score=0.0157) - Bardet-Biedl syndrome is typically inherited in an autosomal recessive pattern, which means both copies of a BBS gene in each cell have mutations. The parents of an individual with an autosomal recessive condition eac...

**Answer**:

Yes, Aarskog-Scott syndrome is inherited in an X-linked recessive pattern [medquad_ans_0000001-4].

**Supported citations**: `medquad_ans_0000001-4`
**Unsupported citations**: _none_

### Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Scores**: context_relevance=0.900, groundedness=0.935, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0001062-3` (score=0.9992) - Mutations in the ALAS2 gene cause X-linked sideroblastic anemia. The ALAS2 gene provides instructions for making an enzyme called erythroid ALA-synthase, which plays a critical role in the production of heme (a compon...
2. `medquad_ans_0001063-3` (score=0.9982) - Mutations in the ABCB7 gene cause X-linked sideroblastic anemia and ataxia. The ABCB7 gene provides instructions for making a protein that is critical for heme production. Heme is a component of the hemoglobin protein...
3. `medquad_ans_0000026-4` (score=0.0013) - African iron overload seems to run in families, and high iron in a family's diet seems to be the major contributor to development of the condition. There also may be a genetic contribution, but the inheritance pattern...
4. `medquad_ans_0000040-3` (score=0.0004) - Alpha thalassemia X-linked intellectual disability syndrome results from mutations in the ATRX gene. This gene provides instructions for making a protein that plays an essential role in normal development. Although th...
5. `medquad_ans_0001063-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...

**Answer**:

Genetic changes related to X-linked sideroblastic anemia include mutations in the ALAS2 gene and mutations in the ABCB7 gene. Mutations in the ALAS2 gene impair the activity of erythroid ALA-synthase, disrupting heme production and leading to reduced hemoglobin synthesis and iron accumulation in erythroblasts [medquad_ans_0001062-3]. Mutations in the ABCB7 gene alter the structure of the ABCB7 protein, disrupting its role in heme production and iron homeostasis in developing red blood cells, which results in anemia [medquad_ans_0001063-3]. Additionally, individuals with mutations in both ALAS2 and HFE genes may experience a more severe form of the disorder due to increased iron overload [medquad_ans_0001062-3].

**Supported citations**: `medquad_ans_0001062-3`, `medquad_ans_0001063-3`
**Unsupported citations**: _none_

### Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Scores**: context_relevance=0.833, groundedness=0.951, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000602-1` (score=0.3290) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifi...
2. `medquad_ans_0003601-1` (score=0.1475) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
3. `medquad_ans_0003610-1` (score=0.0108) - Limb-girdle muscular dystrophy type 2C (LGMD2C) is a condition that affects the muscles and is caused by mutations in the gamma-sarcoglycan gene. This condition belongs to a group of muscle disorders called limb-girdl...
4. `medquad_ans_0000205-1__dup2` (score=0.0097) - The muscular dystrophies (MD) are a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or ...
5. `medquad_ans_0003604-1` (score=0.0029) - Limb-girdle muscular dystrophy type 2A (LGMD2A) is an autosomal recessive limb-girdle muscular dystrophy characterized by progressive, symmetrical weakness of the proximal limb and girdle muscles (mainly those around ...

**Answer**:

Limb-girdle muscular dystrophy is a group of diseases that cause weakness and wasting of the muscles in the arms and legs, particularly affecting the proximal muscles closest to the body, such as those in the shoulders, upper arms, pelvic area, and thighs [medquad_ans_0000602-1]. It is a progressive condition leading to a loss of muscle strength and bulk over time, with onset possibly occurring in childhood, adolescence, young adulthood, or later in life [medquad_ans_0003601-1]. The severity, age of onset, and features vary among the subtypes and may differ even within the same family [medquad_ans_0000602-1]. Most forms are inherited in an autosomal recessive manner, while several rare forms are inherited in an autosomal dominant pattern [medquad_ans_0003601-1].

**Supported citations**: `medquad_ans_0000602-1`, `medquad_ans_0003601-1`
**Unsupported citations**: _none_

### Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Scores**: context_relevance=0.833, groundedness=0.869, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000016-24` (score=0.9810) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even ...
2. `medquad_ans_0000042-7` (score=0.9672) - Low vision can result from a variety of diseases, disorders, and injuries that affect the eye. Many people with low vision have age-related macular degeneration, cataracts, glaucoma, or diabetic retinopathy. Age-relat...
3. `medquad_ans_0000275-1__dup1` (score=0.4466) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
4. `medquad_ans_0000343-1__dup1` (score=0.2204) - Some eye problems are minor and don't last long. But some can lead to a permanent loss of vision. Common eye problems include - Refractive errors - Cataracts - clouded lenses - Glaucoma - a disorder caused by damage t...
5. `medquad_ans_0000016-8` (score=0.2198) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a com...

**Answer**:

Diabetic eye problems refer to eye conditions that can develop as a result of high blood glucose levels in people with diabetes. The most common problem is diabetic retinopathy, which damages the tiny blood vessels in the retina, the light-sensitive tissue at the back of the eye. This condition can progress without symptoms and may lead to vision loss or even blindness if not detected and treated early. Other eye issues associated with diabetes include cataracts, which involve a clouding of the eye's lens, and glaucoma, which occurs when pressure builds up in the eye and damages the main nerve [medquad_ans_0000275-1__dup1]. Macular edema, which is swelling in the macula (the part of the retina responsible for central vision), can also develop without symptoms at any stage of diabetic retinopathy [medquad_ans_0000016-24]. These conditions can be

**Supported citations**: `medquad_ans_0000275-1__dup1`, `medquad_ans_0000016-24`
**Unsupported citations**: _none_

### Example 88 (`0006449-1`)

**Question**: What is (are) Wilson disease ?

**Scores**: context_relevance=0.800, groundedness=0.490, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000133-1__dup3` (score=0.2184) - Wilson disease is a genetic disease that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filt...
2. `medquad_ans_0006449-1` (score=0.0779) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver,...
3. `medquad_ans_0000276-3__dup1` (score=0.0043) - Early onset of the disease may foretell a worse prognosis than later onset. If the disorder is detected early and treated appropriately, an individual with WD can usually enjoy normal health and a normal lifespan. If ...
4. `medquad_ans_0000133-8` (score=0.0041) - People who have Wilson disease that is not treated or diagnosed early can have serious complications, such as - cirrhosisscarring of the liver - kidney damageas liver function decreases, the kidneys may be damaged - p...
5. `medquad_ans_0000276-4__dup1` (score=0.0029) - The National Institute of Neurological Disorders and Stroke, the Eunice Kennedy Shriver National Institute of Child Health and Human Development, and other institutes of the National Institutes of Health (NIH) conduct...

**Answer**:

Wilson disease is a rare inherited disorder characterized by the accumulation of copper in the body due to a mutation of the ATP7B gene, which is inherited in an autosomal recessive manner. The body normally filters extra copper through the liver and releases it into bile, but in Wilson disease, the liver does not filter copper correctly, leading to copper buildup in the liver, brain, eyes, and other organs. High levels of copper are toxic and can cause damage to the liver, brain, and eyes. Signs and symptoms include chronic liver disease, central nervous system abnormalities, and psychiatric disturbances. If left untreated, the disease can lead to serious complications such as cirrhosis, kidney damage, persistent nervous system problems, liver cancer, liver failure, and death. Early detection and appropriate treatment can allow individuals to enjoy normal health and a normal lifespan, with lifelong treatment required [medquad*ans_0006449-1, medquad_ans*

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 89 (`0001503-2`)

**Question**: What are the symptoms of Congenital dyserythropoietic anemia type 2 ?

**Scores**: context_relevance=0.444, groundedness=0.000, answer_relevance=0.711

**Top docs**:

1. `medquad_ans_0001062-2` (score=0.9425) - This form of anemia is uncommon. However, researchers believe that it may not be as rare as they once thought. Increased awareness of the disease has led to more frequent diagnoses.
2. `medquad_ans_0000529-2` (score=0.0003) - Although iron deficiency anemia is relatively common, the prevalence of the iron-refractory form of the disease is unknown. At least 50 cases have been described in the medical literature. Researchers suspect that iro...
3. `medquad_ans_0000283-2` (score=0.0003) - Diamond-Blackfan anemia affects approximately 5 to 7 per million liveborn infants worldwide.
4. `medquad_ans_0000503-2` (score=0.0002) - Hypochromic microcytic anemia with iron overload is likely a rare disorder; at least five affected families have been reported in the scientific literature.
5. `medquad_ans_0002164-4` (score=0.0001) - How is erythropoietic protoporphyria (EPP) inherited? EPP is inherited in an autosomal recessive manner. In most cases, affected individuals have one severe (loss-of-function) mutation that is inherited from one paren...

**Answer**:

The provided context does not contain information about the symptoms of Congenital dyserythropoietic anemia type 2. Therefore, the question cannot be answered based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Scores**: context_relevance=0.800, groundedness=0.838, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000058-11__dup1` (score=0.9837) - Kidney failure, also called end-stage kidney disease or ESRD, means your kidneys no longer work well enough to do their job. You will need treatment to replace the work your kidneys have stopped doing.
2. `medquad_ans_0000155-3__dup3` (score=0.9495) - Chronic kidney disease means you have damaged kidneys that cannot filter blood normally. Wastes then build up in your blood, harming your body. Kidney disease usually does not get better and may lead to kidney failure...
3. `medquad_ans_0000140-2__dup2` (score=0.9286) - Chronic kidney disease is kidney damage that occurs slowly over many years, often due to diabetes or high blood pressure. Once damaged, the kidneys cant filter blood as they should. This damage can cause wastes to bui...
4. `medquad_ans_0000155-1__dup4` (score=0.8638) - If you have advanced chronic kidney disease (CKD), you may soon need treatment to do the work your kidneys can no longer do. Learning about your treatment options for kidney failure will help you make the best choice ...
5. `medquad_ans_0000038-14` (score=0.1635) - Kidney disease is often called a "silent" disease, because most people have no symptoms in early kidney disease. In fact, you might feel just fine until your kidneys have almost stopped working. Do NOT wait for sympto...

**Answer**:

Kidney disease refers to damage in the kidneys that impairs their ability to filter blood normally, leading to the buildup of wastes in the body, which can harm overall health. Chronic kidney disease develops slowly over many years, often due to conditions like diabetes or high blood pressure, and may not show symptoms until the kidneys have significantly damaged [medquad_ans_0000140-2__dup2]. In some cases, kidney disease may progress to kidney failure, where the kidneys no longer work well enough to perform their essential functions, requiring treatment to replace their work [medquad_ans_0000058-11__dup1]. Kidney disease is often called a "silent" disease because most people have no symptoms in early stages, and blood and urine tests are the only reliable way to detect it or measure kidney function [medquad_ans_0000038-14].

**Supported citations**: `medquad_ans_0000140-2__dup2`, `medquad_ans_0000058-11__dup1`, `medquad_ans_0000038-14`
**Unsupported citations**: _none_

### Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Scores**: context_relevance=0.857, groundedness=0.616, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000836-5` (score=0.9860) - These resources address the diagnosis or management of prostate cancer: - American College of Radiology: Prostate Cancer Radiation Treatment - Genetic Testing Registry: Familial prostate cancer - Genetic Testing Regis...
2. `medquad_ans_0000058-24` (score=0.9795) - Surgery, radiation therapy, and hormonal therapy all have the potential to disrupt sexual desire or performance for a short while or permanently. Discuss your concerns with your health care provider. Several options a...
3. `medquad_ans_0000058-28` (score=0.9740) - Hormonal therapy deprives cancer cells of the male hormones they need to grow and survive. This treatment is often used for prostate cancer that has spread to other parts of the body. Sometimes doctors use hormonal th...
4. `medquad_ans_0000058-4__dup3` (score=0.9707) - If tests show that you have cancer, you should talk with your doctor in order to make treatment decisions. Working With a Team of Specialists A team of specialists often treats people with cancer. The team will keep t...
5. `medquad_ans_0000058-30` (score=0.8510) - Through research, doctors are trying to find new, more effective ways to treat prostate cancer. Cryosurgery -- destroying cancer by freezing it -- is under study as an alternative to surgery and radiation therapy. To ...

**Answer**:

The treatments for prostate cancer include surgery, radiation therapy, and hormonal therapy. Surgery, such as radical prostatectomy, and radiation therapy, including brachytherapy, are used to treat prostate cancer. Hormonal therapy is often used for prostate cancer that has spread to other parts of the body or to prevent recurrence after surgery or radiation. These treatments may have side effects such as impotence, loss of sexual desire, and hot flashes [medquad_ans_0000058-24, medquad_ans_0000058-28]. Additionally, clinical trials may offer new treatment options for prostate cancer patients [medquad_ans_0000058-4__dup3].

**Supported citations**: `medquad_ans_0000058-4__dup3`
**Unsupported citations**: `medquad_ans_0000058-24, medquad_ans_0000058-28`

### Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Scores**: context_relevance=0.857, groundedness=0.900, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000586-1__dup2` (score=0.9973) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state ha...
2. `medquad_ans_0000431-1__dup1` (score=0.9972) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...
3. `medquad_ans_0000362-1__dup2` (score=0.9971) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
4. `medquad_ans_0000590-1__dup2` (score=0.9927) - Summary : Part D is the name of Medicare's prescription drug coverage. It's insurance that helps people pay for prescription drugs. It is available to everyone who has Medicare. It provides protection if you pay high ...
5. `medquad_ans_0000338-1__dup1` (score=0.9707) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the s...

**Answer**:

Yes, there is information about health insurance. Health insurance is a contract between you and your insurance company, where you buy a plan and the company agrees to pay part of your medical expenses when you need care [medquad_ans_0000431-1__dup1]. Many people obtain health insurance through their employers, and employer-sponsored plans are often managed care plans that contract with healthcare providers to offer care at reduced costs [medquad_ans_0000431-1__dup1]. Individuals can also purchase health insurance on their own or qualify for government health insurance such as Medicare and Medicaid [medquad_ans_0000431-1__dup1]. Medicaid is a government health insurance program that assists low-income individuals in paying medical bills, with state-specific rules regarding income, age, disability, pregnancy, and immigration status [medquad_ans_0000586-1\_\_

**Supported citations**: `medquad_ans_0000431-1__dup1`
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

**Scores**: context_relevance=0.667, groundedness=0.859, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000028_3-5` (score=0.9974) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer (the size of the tumor and wheth...
2. `medquad_ans_0000028_1-4` (score=0.9973) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) depends mostly on the stage of the cancer and the type of treatment used to remove the cancer. Treatment ...
3. `medquad_ans_0000017_1-5` (score=0.9940) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - How the melanoma cells look under a microscope. - The si...
4. `medquad_ans_0000063-30` (score=0.0762) - The National Cancer Institute has developed a comprehensive online cancer database called the Physician Data Query (PDQ) to present evidence from the most recent research on melanoma and other skin cancers. Click here...
5. `medquad_ans_0000063-15` (score=0.0225) - Melanoma is the fastest growing and most invasive type of skin cancer. This cancer arises from overgrowth of melanocytes. Melanocytes are specialized skin cells that produce a pigment called melanin.

**Answer**:

The outlook for melanoma depends on several factors, including the stage of the cancer (such as tumor size and whether it has spread to lymph nodes or other parts of the body), the location of the cancer in the body, whether the cancer is newly diagnosed or has recurred, the patient's age and general health, and how deeply the tumor has grown into the skin [medquad_ans_0000028_3-5]. Additionally, for melanoma specifically, the outlook also depends on how the melanoma cells look under a microscope, the size and thickness of the tumor, the part of the eye the tumor is in (if applicable), whether the tumor has spread within the eye or to other parts of the body, and the presence of certain gene changes linked to intraocular melanoma [medquad_ans_0000017_1-5]. The prognosis is also influenced by the type of treatment used to

**Supported citations**: `medquad_ans_0000028_3-5`, `medquad_ans_0000017_1-5`
**Unsupported citations**: _none_

### Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000136-4` (score=0.9979) - BOR/BO syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In about 90 percent of cases, an affected person inherits the m...
2. `medquad_ans_0000010-4` (score=0.0375) - Acromicric dysplasia is an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene and occur in people ...
3. `medquad_ans_0000144-4` (score=0.0297) - Campomelic dysplasia is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in or near the SOX9 g...
4. `medquad_ans_0000838-3__dup1` (score=0.0137) - How is branchiooculofacial syndrome (BOFS) inherited? Although some cases can be sporadic, most of the reported cases are inherited within families. BOFS is inherited in an autosomal dominant pattern, which means that...
5. `medquad_ans_0000268-4` (score=0.0128) - Czech dysplasia is inherited in an autosomal dominant pattern, which means one copy of the altered COL2A1 gene in each cell is sufficient to cause the disorder. All known individuals with Czech dysplasia inherited the...

**Answer**:

The provided context does not contain any information about boomerang dysplasia or its inheritance pattern. Therefore, it is not possible to determine based on the given text whether boomerang dysplasia is inherited.

Answer: Insufficient context [medquad_ans_0000136-4, medquad_ans_0000010-4, medquad_ans_0000144-4, medquad_ans_0000838-3__dup1, medquad_ans_0000268-4].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000136-4, medquad_ans_0000010-4, medquad_ans_0000144-4, medquad_ans_0000838-3__dup1, medquad_ans_0000268-4`

### Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Scores**: context_relevance=0.700, groundedness=0.468, answer_relevance=0.620

**Top docs**:

1. `medquad_ans_0000466-3` (score=0.9991) - Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and the EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. The two exosto...
2. `medquad_ans_0001099-4` (score=0.0078) - This condition is generally not inherited but arises from a mutation in the body's cells that occurs after conception. This alteration is called a somatic mutation. Affected people typically have no history of the dis...
3. `medquad_ans_0000375-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of fibrodysplasia ossificans progressiva result from ...
4. `medquad_ans_0002838-1` (score=0.0001) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called...
5. `medquad_ans_0002838-3` (score=0.0001) - How is hereditary multiple osteochondromas inherited? HMO is caused by mutations in the EXT1 and EXT2 genes. It is inherited in an autosomal dominant pattern, which means that one copy of the altered gene in each cell...

**Answer**:

Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene provides instructions for producing the exostosin-1 protein, and the EXT2 gene provides instructions for producing the exostosin-2 protein. These two proteins bind together to form a complex in the Golgi apparatus, where they modify heparan sulfate. When there is a mutation in exostosin-1 or exostosin-2, heparan sulfate cannot be processed correctly and becomes nonfunctional. Mutations in the EXT1 gene result in hereditary multiple osteochondromas type 1, and mutations in the EXT2 gene result in hereditary multiple osteochondromas type 2. Mutations in the EXT1 gene are likely responsible for 55 to 75 percent of all cases, and symptoms tend to be more severe in type 1. Approximately 15 percent of individuals

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Scores**: context_relevance=0.250, groundedness=0.000, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000746-2` (score=0.9995) - OFCD syndrome is very rare; the incidence is estimated to be less than 1 in 1 million people.
2. `medquad_ans_0000136-2` (score=0.9995) - Researchers estimate that BOR/BO syndrome affects about 1 in 40,000 people.
3. `medquad_ans_0000174-2` (score=0.9995) - CHARGE syndrome occurs in approximately 1 in 8,500 to 10,000 individuals.
4. `medquad_ans_0000245-2` (score=0.9994) - This condition is very rare; it probably affects 200 to 300 people worldwide. Reported estimates of Costello syndrome prevalence range from 1 in 300,000 to 1 in 1.25 million people.
5. `medquad_ans_0000135-2` (score=0.9991) - Branchio-oculo-facial syndrome is a rare condition, although the prevalence is unknown.

**Answer**:

The provided context does not contain information about the prevalence or number of people affected by oculofaciocardiodental syndrome. Therefore, it is not possible to answer the question based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Scores**: context_relevance=0.750, groundedness=0.550, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000189-1` (score=0.0631) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused b...
2. `medquad_ans_0000445-1__dup1` (score=0.0302) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...
3. `medquad_ans_0001350-1` (score=0.0046) - Citrullinemia type I is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. This condition, also known as classic citrullinemia, belongs to a class of genetic diseases call...
4. `medquad_ans_0004334-1` (score=0.0025) - Neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) is a liver condition is also known as neonatal-onset type II citrullinemia. NICCD blocks the flow of bile (a digestive fluid produced by the liver)...
5. `medquad_ans_0000482-1` (score=0.0017) - Histidinemia is an inherited condition characterized by elevated blood levels of the amino acid histidine, a building block of most proteins. Histidinemia is caused by the shortage (deficiency) of the enzyme that brea...

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

**Scores**: context_relevance=0.429, groundedness=0.622, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000943-5` (score=0.9942) - These resources address the diagnosis or management of spondylocarpotarsal synostosis syndrome: - Gene Review: Gene Review: FLNB-Related Disorders - Genetic Testing Registry: Spondylocarpotarsal synostosis syndrome Th...
2. `medquad_ans_0000949-5` (score=0.9918) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from Medli...
3. `medquad_ans_0000944-5` (score=0.9918) - These resources address the diagnosis or management of spondylocostal dysostosis: - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive - Genetic Testing Registry: Jarcho-Levin syndrome - Genetic ...
4. `medquad_ans_0000187-5` (score=0.9854) - These resources address the diagnosis or management of CHST3-related skeletal dysplasia: - Gene Review: Gene Review: CHST3-Related Skeletal Dysplasia - Genetic Testing Registry: Spondyloepiphyseal dysplasia with conge...
5. `medquad_ans_0000948-5` (score=0.3362) - These resources address the diagnosis or management of spondyloperipheral dysplasia: - Genetic Testing Registry: Spondyloperipheral dysplasia - MedlinePlus Encyclopedia: Nearsightedness These resources from MedlinePlu...

**Answer**:

The provided context does not specify treatments for spondylocostal dysostosis. It lists resources such as gene reviews, genetic testing registries, and MedlinePlus topics like X-ray exams and scoliosis, which may relate to diagnosis or management, but no explicit treatments are mentioned. Therefore, based on the given information, the treatments for spondylocostal dysostosis cannot be determined. [medquad_ans_0000944-5]

**Supported citations**: `medquad_ans_0000944-5`
**Unsupported citations**: _none_

### Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.948, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000145-3__dup3` (score=0.0026) - The symptoms of Goodpasture syndrome may initially include fatigue, nausea, vomiting, and weakness. The lungs are usually affected before or at the same time as the kidneys, and symptoms can include shortness of breat...
2. `medquad_ans_0000791-2` (score=0.0003) - Pfeiffer syndrome affects about 1 in 100,000 individuals.
3. `medquad_ans_0000954-2` (score=0.0001) - Stickler syndrome affects an estimated 1 in 7,500 to 9,000 newborns. Type I is the most common form of the condition.
4. `medquad_ans_0000145-6` (score=0.0001) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing Goodpasture syndrome.
5. `medquad_ans_0000786-3` (score=0.0001) - Perry syndrome results from mutations in the DCTN1 gene. This gene provides instructions for making a protein called dynactin-1, which is involved in the transport of materials within cells. To move materials, dynacti...

**Answer**:

The symptoms of Perry syndrome are due to a gradual loss of neurons in areas of the brain that regulate movement, emotion, and breathing. [medquad_ans_0000786-3]

**Supported citations**: `medquad_ans_0000786-3`
**Unsupported citations**: _none_

### Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Scores**: context_relevance=1.000, groundedness=0.894, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0006398-2` (score=0.9992) - What are the signs and symptoms of Warfarin sensitivity? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin sensitivity. If the information is available, the table below includ...
2. `medquad_ans_0000195-2__dup2` (score=0.9310) - Because there is no way to tell whether symptoms are from a TIA or an acute stroke, patients should assume that all stroke-like symptoms signal an emergency and should not wait to see if they go away. A prompt evaluat...
3. `medquad_ans_0001011-2` (score=0.0023) - UV-sensitive syndrome appears to be a rare condition; only a small number of affected individuals have been reported in the scientific literature. However, this condition may be underdiagnosed.
4. `medquad_ans_0000034-2__dup2` (score=0.0007) - Within a few hours after onset of a stroke, treatment with drugs or devices that dissolve or break up the clot can restore blood flow to the brain and lead to a better recovery. To prevent strokes related to AF, docto...
5. `medquad_ans_0000034-3__dup2` (score=0.0003) - AF, which affects as many as 2.2 million Americans, increases an individuals risk of stroke by 4 to 6 times on average. The risk increases with age. In people over 80 years old, AF is the direct cause of 1 in 4 stroke...

**Answer**:

The signs and symptoms of Warfarin sensitivity are listed by the Human Phenotype Ontology, which includes abnormalities of blood and blood-forming tissues and abnormalities of metabolism/homeostasis. However, the specific symptoms are not detailed in the provided context. The frequency of these symptoms is estimated as a percentage or fraction of patients, with some entries lacking frequency data. For definitions of medical terms, the MedlinePlus Medical Dictionary can be consulted [medquad_ans_0006398-2].

**Supported citations**: `medquad_ans_0006398-2`
**Unsupported citations**: _none_

### Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.990, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0004262-1` (score=0.7196) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, a...
2. `medquad_ans_0000174-2` (score=0.0015) - CHARGE syndrome occurs in approximately 1 in 8,500 to 10,000 individuals.
3. `medquad_ans_0000771-2` (score=0.0013) - Paramyotonia congenita is an uncommon disorder; it is estimated to affect fewer than 1 in 100,000 people.
4. `medquad_ans_0001036-2` (score=0.0012) - Williams syndrome affects an estimated 1 in 7,500 to 10,000 people.
5. `medquad_ans_0000245-2` (score=0.0010) - This condition is very rare; it probably affects 200 to 300 people worldwide. Reported estimates of Costello syndrome prevalence range from 1 in 300,000 to 1 in 1.25 million people.

**Answer**:

Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement of the muscles (muscle hypertrophy), and/or joint stiffness. Myhre syndrome is caused by mutations in the SMAD4 gene and is inherited in an autosomal dominant pattern. Most cases are due to a new mutation [medquad_ans_0004262-1].

**Supported citations**: `medquad_ans_0004262-1`
**Unsupported citations**: _none_

### Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Scores**: context_relevance=0.714, groundedness=0.000, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000501-2` (score=0.0014) - Hypochondrogenesis and achondrogenesis, type 2 (a similar skeletal disorder) together affect 1 in 40,000 to 60,000 newborns.
2. `medquad_ans_0000290-2` (score=0.0003) - Distal arthrogryposis type 1 affects an estimated 1 in 10,000 people worldwide.
3. `medquad_ans_0000976-4` (score=0.0001) - Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell is sufficient to cause the condition. Virtually all cases of thanatophoric dysplasia are cau...
4. `medquad_ans_0000976-2` (score=0.0001) - This condition occurs in 1 in 20,000 to 50,000 newborns. Type I thanatophoric dysplasia is more common than type II.
5. `medquad_ans_0006057-4` (score=0.0001) - Is thanatophoric dysplasia inherited? Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell causes the condition. However, almost all cases of tha...

**Answer**:

The provided context does not include information about the symptoms of thanatophoric dysplasia. Therefore, the symptoms cannot be determined from the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.894, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000381-3` (score=0.0001) - Mutations in the FMR1 gene cause fragile X syndrome. The FMR1 gene provides instructions for making a protein called FMRP. This protein helps regulate the production of other proteins and plays a role in the developme...
2. `medquad_ans_0002417-4` (score=0.0001) - How is fragile X syndrome inherited? Fragile X syndrome (FXS) is inherited in an X-linked dominant manner. A condition is X-linked if the responsible gene is located on the X chromosome. The inheritance is dominant if...
3. `medquad_ans_0000383-5` (score=0.0001) - These resources address the diagnosis or management of FXTAS: - Fragile X Research Foundation of Canada: FXTAS - Gene Review: Gene Review: FMR1-Related Disorders - Genetic Testing Registry: Fragile X tremor/ataxia syn...
4. `medquad_ans_0000384-1` (score=0.0001) - Fragile XE syndrome is a genetic disorder that impairs thinking ability and cognitive functioning. Most affected individuals have mild intellectual disability. In some people with this condition, cognitive function is...
5. `medquad_ans_0002417-1` (score=0.0001) - Fragile X syndrome is a genetic condition involving changes in part of the X chromosome. This condition causes a range of developmental problems including learning disabilities and cognitive impairment. It is the most...

**Answer**:

The symptoms of fragile X syndrome include disrupted nervous system functions due to a deficiency of the FMRP protein, which results from the silencing of the FMR1 gene caused by an abnormally expanded CGG repeat (more than 200 times). This leads to the signs and symptoms of fragile X syndrome, which may include intellectual disabilities, developmental delays, and behavioral challenges. Additionally, individuals with a premutation (55 to 200 CGG repeats) may experience mild physical features such as prominent ears, emotional problems like anxiety or depression, learning disabilities, or autistic-like behavior [medquad_ans_0000381-3]. However, the specific symptoms of full fragile X syndrome are primarily attributed to the loss of FMRP and its role in synaptic development and regulation [medquad_ans_0000381-3].

**Supported citations**: `medquad_ans_0000381-3`
**Unsupported citations**: _none_

### Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Scores**: context_relevance=0.750, groundedness=0.952, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000068-3__dup1` (score=0.3928) - Cerebral arteriosclerosis can lead to life threatening health events such as ischemic or hemorrhagic strokes. People who survive stroke may have long-term neurological and motor impairments.
2. `medquad_ans_0000056-1__dup1` (score=0.2599) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease that occurs when the thickening of blood vessel walls blocks the fl...
3. `medquad_ans_0000889-1__dup1` (score=0.2255) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited disease of the blood vessels that occurs when the thickening of blood vessel walls blocks the flow ...
4. `medquad_ans_0000166-1` (score=0.1349) - Cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy, usually called CADASIL, is an inherited condition that causes stroke and other impairments. This condition affects blood flow...
5. `medquad_ans_0001051-1__dup1` (score=0.0563) - Cerebral autosomal recessive arteriopathy with subcortical infarcts and leukoencephalopathy, commonly known as CARASIL, is an inherited condition that causes stroke and other impairments. This progressive condition is...

**Answer**:

CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease caused by a mutation in the NOTCH3 gene, which leads to thickening of the blood vessel walls and blocks blood flow to the brain, primarily affecting small blood vessels in the white matter of the brain [medquad_ans_0000056-1__dup1]. It is characterized by migraine headaches, multiple strokes, and progression to dementia, along with other symptoms such as cognitive deterioration, seizures, vision problems, and psychiatric issues like severe depression and changes in behavior and personality [medquad_ans_0000056-1__dup1]. CADASIL is inherited in an autosomal dominant manner, meaning one parent carries and passes on the defective gene, and most individuals have a family history of the disorder [

**Supported citations**: `medquad_ans_0000056-1__dup1`
**Unsupported citations**: _none_

### Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Scores**: context_relevance=0.833, groundedness=0.958, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0004185-1` (score=0.9647) - Multiple endocrine neoplasia, type 1 (MEN1) is an inherited condition that causes tumors of the endocrine system (the body's network of hormone-producing glands). People affected by MEN1 typically develop tumors of th...
2. `medquad_ans_0000010-1__dup6` (score=0.9398) - MEN1 is an inherited disorder that causes tumors in the endocrine glands and the duodenum, the first part of the small intestine. MEN1 is sometimes called multiple endocrine adenomatosis or Wermer's syndrome, after on...
3. `medquad_ans_0000688-1` (score=0.6657) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate...
4. `medquad_ans_0000014-5__dup1` (score=0.0088) - In about 80 percent of people with primary hyperparathyroidism, a benign, or noncancerous, tumor called an adenoma has formed in one of the parathyroid glands.2 The tumor causes the gland to become overactive. In most...
5. `medquad_ans_0000025_2-2` (score=0.0021) - Having certain syndromes can increase the risk of pancreatic NETs. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not havin...

**Answer**:

Multiple endocrine neoplasia (MEN) is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). These disorders typically involve tumors (neoplasia) in at least two endocrine glands, and tumors can also develop in other organs and tissues. The growths can be noncancerous (benign) or cancerous (malignant), and if cancerous, the condition can be life-threatening. The major forms of multiple endocrine neoplasia are type 1, type 2, and type 4, which are distinguished by the genes involved, the types of hormones produced, and the characteristic signs and symptoms [medquad_ans_0000688-1]. Multiple endocrine neoplasia type 1 (MEN1) specifically causes tumors of the parathyroid gland, pituitary gland, and pancreas, and is caused

**Supported citations**: `medquad_ans_0000688-1`
**Unsupported citations**: _none_

### Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Scores**: context_relevance=0.750, groundedness=0.992, answer_relevance=0.150

**Top docs**:

1. `medquad_ans_0004155-3` (score=0.9959) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycos...
2. `medquad_ans_0000682-2` (score=0.0003) - The exact incidence of MPS VI is unknown, although it is estimated to occur in 1 in 250,000 to 600,000 newborns.
3. `medquad_ans_0000681-2` (score=0.0003) - The exact prevalence of MPS IV is unknown, although it is estimated to occur in 1 in 200,000 to 300,000 individuals.
4. `medquad_ans_0000679-3` (score=0.0003) - Mutations in the IDS gene cause MPS II. The IDS gene provides instructions for producing the I2S enzyme, which is involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs). GAGs were original...
5. `medquad_ans_0000678-2` (score=0.0002) - Severe MPS I occurs in approximately 1 in 100,000 newborns. Attenuated MPS I is less common and occurs in about 1 in 500,000 newborns.

**Answer**:

Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs). Mutations in the IDUA gene reduce or completely eliminate the function of the IDUA enzyme. The lack of IDUA enzyme activity leads to the accumulation of GAGs within cells, specifically inside the lysosomes. Lysosomes are compartments in the cell that digest and recycle different types of molecules. Conditions that cause molecules to build up inside the lysosomes, including MPS I, are called lysosomal storage disorders. The accumulation of GAGs increases the size of the lysosomes, which is why many tissues and organs are enlarged in this disorder [medquad_ans_0004155-3].

**Supported citations**: `medquad_ans_0004155-3`
**Unsupported citations**: _none_

### Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Scores**: context_relevance=0.900, groundedness=0.941, answer_relevance=0.580

**Top docs**:

1. `medquad_ans_0000825-3` (score=0.9992) - Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition. The FLCN gene provides instructions for making a protein called follicu...
2. `medquad_ans_0000126-3` (score=0.9992) - Mutations in the FLCN gene cause Birt-Hogg-Dub syndrome. This gene provides instructions for making a protein called folliculin. The normal function of this protein is unknown, but researchers believe that it may act ...
3. `medquad_ans_0000963-4` (score=0.0002) - Surfactant dysfunction can have different inheritance patterns depending on its genetic cause. When caused by mutations in the SFTPB or ABCA3 gene, this condition is inherited in an autosomal recessive pattern, which ...
4. `medquad_ans_0000962-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. However, some people who inherit the altered gene never develop ...
5. `medquad_ans_0000608-4` (score=0.0001) - Most cases of lung cancer are not related to inherited gene changes. These cancers are associated with somatic mutations that occur only in certain cells in the lung. When lung cancer is related to inherited gene chan...

**Answer**:

Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition [medquad_ans_0000825-3]. The FLCN gene provides instructions for making a protein called folliculin, which is found in lung connective tissue cells and in cells lining the alveoli. Researchers believe folliculin may help control cell growth and division and may play a role in repairing and re-forming lung tissue after damage. One theory is that altered folliculin due to FLCN mutations may trigger inflammation in lung tissue, leading to tissue damage and the formation of blebs, which increase the risk of pneumothorax [medquad_ans_0000825-3]. However, the exact mechanism by which FLCN mutations lead to pneumothorax is not fully understood. Primary spontaneous pneumothorax most often occurs in

**Supported citations**: `medquad_ans_0000825-3`
**Unsupported citations**: _none_

### Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Scores**: context_relevance=0.750, groundedness=0.854, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000066-32` (score=0.9961) - For more information on stroke, including research sponsored by the National Institute of Neurological Disorders and Stroke, call 1-800-352-9424 or visit the Web site at www.ninds.nih.gov.
2. `medquad_ans_0000066-18` (score=0.9921) - Transient ischemic attacks, or TIAs, occur when the warning signs of stroke last only a few moments and then disappear. These brief episodes are also sometimes called "mini-strokes." Although brief, they identify an u...
3. `medquad_ans_0000066-13` (score=0.9599) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blo...
4. `medquad_ans_0000261-4__dup2` (score=0.9013) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts stroke research and clinical trials at its laboratories and clinics at the National Institutes of Health (NIH), and through grants to major ...
5. `medquad_ans_0000034-3__dup2` (score=0.8214) - AF, which affects as many as 2.2 million Americans, increases an individuals risk of stroke by 4 to 6 times on average. The risk increases with age. In people over 80 years old, AF is the direct cause of 1 in 4 stroke...

**Answer**:

Stroke is a medical condition characterized by the sudden interruption of blood flow to part of the brain, leading to brain damage. There are two main types: ischemic stroke, which accounts for approximately 80 percent of all strokes and is caused by a blood clot blocking or plugging a blood vessel in the brain, and hemorrhagic stroke, which occurs when a blood vessel in the brain breaks and bleeds into the brain [medquad_ans_0000066-13]. Transient ischemic attacks (TIAs), sometimes referred to as "mini-strokes," are brief episodes of stroke-like symptoms that last only a few moments and then disappear, but they indicate an underlying serious condition that requires medical attention [medquad_ans_0000066-18]. Additionally, atrial fibrillation (AF), which affects up to 2.2 million Americans, increases an individual's risk of stroke by 4 to

**Supported citations**: `medquad_ans_0000066-13`, `medquad_ans_0000066-18`
**Unsupported citations**: _none_

### Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006276-3` (score=0.9953) - What causes uncombable hair syndrome? The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been s...
2. `medquad_ans_0000247-3__dup1` (score=0.0011) - What causes alopecia universalis? The exact underlying cause of alopecia universalis (AU) is not currently known. AU is an advanced form of alopecia areata (AA), a condition that leads to round patches of hair loss. A...
3. `medquad_ans_0006276-6` (score=0.0001) - How might uncombable hair syndrome be treated? There is no definitive treatment for uncombable hair syndrome, but the condition usually improves or resolves on its own with the onset of puberty. Gentle hair care is ge...
4. `medquad_ans_0006276-1` (score=0.0001) - Uncombable hair syndrome (UHS) is a rare disorder of the hair shaft of the scalp. It usually is characterized by silvery-blond or straw-colored hair that is disorderly; stands out from the scalp; and cannot be combed ...
5. `medquad_ans_0002708-1` (score=0.0001) - HAIR-AN syndrome is a condition that affects women. It is characterized by hyperandrogenism, insulin resistance, and acanthosis nigricans. Insulin resistance is a condition in which the body produces insulin but does ...

**Answer**:

The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft seen in cross section, which results from premature keratinization of the inner root sheath. This inner root sheath conforms to the abnormal outline of the hair shaft, forming an irregular, rigid tube that alters the shape of the emerging hair. While UHS is assumed to be autosomal dominant and thus due to mutations in a gene, no responsible gene has been identified [medquad_ans_0006276-3].

**Supported citations**: `medquad_ans_0006276-3`
**Unsupported citations**: _none_

### Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.962, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000912-4` (score=0.9889) - Silver syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In these cases, the affected person inherits the mutation from ...
2. `medquad_ans_0000887-4` (score=0.9860) - Most cases of Russell-Silver syndrome are sporadic, which means they occur in people with no history of the disorder in their family. Less commonly, Russell-Silver syndrome can run in families. In some affected famili...
3. `medquad_ans_0005440-4` (score=0.9638) - Is Russell-Silver syndrome inherited? Most cases of Russell-Silver syndrome (RSS) are sporadic (not inherited), which means they occur in people with no family history of RSS. Less commonly, Russell-Silver syndrome is...
4. `medquad_ans_0006446-4` (score=0.9543) - Is Williams syndrome inherited?
5. `medquad_ans_0000527-4` (score=0.1646) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...

**Answer**:

Yes, Russell-Silver syndrome can be inherited, but most cases are sporadic and not inherited. Most cases occur in people with no family history of the disorder. Less commonly, it is inherited in an autosomal dominant or autosomal recessive pattern. In some families, it appears to be inherited in an autosomal dominant manner, meaning one copy of a genetic change in each cell is sufficient to cause the disorder. In other families, it is inherited in an autosomal recessive manner, meaning both copies of the responsible gene must be altered for the condition to manifest [medquad_ans_0005440-4].

**Supported citations**: `medquad_ans_0005440-4`
**Unsupported citations**: _none_

### Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Scores**: context_relevance=1.000, groundedness=0.946, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.9971) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0000111-1__dup2` (score=0.0073) - Summary : Do you know if your current weight is healthy? "Underweight", "normal", "overweight", and "obese" are all labels for ranges of weight. Obese and overweight mean that your weight is greater than it should be ...
3. `medquad_ans_0000067-29` (score=0.0008) - Certain kinds of chemotherapy and medicines contribute to weight gain. Unfortunately, the usual ways people try to lose weight may not work. Ask your doctor about talking with a nutritionist who can help you plan a he...
4. `medquad_ans_0000015-14` (score=0.0003) - Studies have shown that most people with prediabetes develop type 2 diabetes within a few years, unless they change their lifestyle. Most people with prediabetes dont have any symptoms. Your doctor can test your blood...
5. `medquad_ans_0000022-11` (score=0.0003) - Your eating, diet, and nutrition choices play an important role in preventing or delaying diabetes. Follow the suggestions below to reach and maintain a reasonable weight and make wise food choices most of the time. R...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. It often limits the amount of food you can take in, and some types affect how you digest and absorb nutrients. All types of surgery carry risks and complications, such as infections, hernias, and blood clots. Many people lose weight quickly, but some regain weight later; however, following diet and exercise recommendations can help keep most of the weight off. Ongoing medical follow-up is required for the rest of your life [medquad_ans_0000969-1__dup2].

**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

### Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.777, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000114-3__dup2` (score=0.9956) - ESS is not a life-threatening condition. Most often, and particularly among those with primary ESS, the disorder does not cause health problems and does not affect life expectancy.
2. `medquad_ans_0000889-5` (score=0.9947) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Sae...
3. `medquad_ans_0000124-2__dup1` (score=0.2326) - There is no cure for Fahr's Syndrome, nor is there a standard course of treatment. Treatment addresses symptoms on an individual basis.
4. `medquad_ans_0000071-2__dup1` (score=0.0297) - There is no standard course of treatment for Sotos syndrome. Treatment is symptomatic.
5. `medquad_ans_0002724-5` (score=0.0294) - How is Hanhart syndrome treated? Because Hanhart syndrome affects many different systems of the body, medical management is often provided by a team of doctors and other healthcare professionals. Treatment for this co...

**Answer**:

The provided context does not specify the treatments for Saethre-Chotzen syndrome. While it lists resources that address the diagnosis or management of Saethre-Chotzen syndrome, such as the Gene Review, Genetic Testing Registry, and MedlinePlus Encyclopedia, it does not detail the specific treatments available [medquad_ans_0000889-5]. Therefore, based on the given information, the treatments for Saethre-Chotzen syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000889-5`
**Unsupported citations**: _none_

### Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=1.000, groundedness=0.474, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000078-1__dup2` (score=0.9214) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...
2. `medquad_ans_0000512-1` (score=0.2314) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effect...
3. `medquad_ans_0000758-1__dup2` (score=0.1210) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
4. `medquad_ans_0000522-1__dup1` (score=0.0140) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...
5. `medquad_ans_0000078-6` (score=0.0121) - Idiopathic pulmonary fibrosis (IPF) causes the same kind of scarring and symptoms as some other lung diseases. This makes it hard to diagnose. Seeking medical help as soon as you have symptoms is important. If possibl...

**Answer**:

Idiopathic pulmonary fibrosis (IPF) is a chronic, progressive lung disease in which scar tissue (fibrosis) builds up in the lungs, making them unable to transport oxygen into the bloodstream effectively. It usually affects people between the ages of 50 and 70 and is characterized by symptoms such as shortness of breath, a persistent dry, hacking cough, loss of appetite, gradual weight loss, and in some cases, widened and rounded tips of the fingers and toes (clubbing) due to oxygen shortage. IPF has no known cause in most cases and is referred to as "idiopathic." When more than one family member has the disease, it is called familial IPF. The disease progresses over time, leading to reduced oxygen supply to the body's organs and tissues, and it is associated with serious complications such as respiratory failure, pulmonary hypertension, heart failure, pulmonary embolism, pneumonia, and lung cancer. Currently,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Scores**: context_relevance=0.250, groundedness=0.000, answer_relevance=0.583

**Top docs**:

1. `medquad_ans_0000051-4__dup1` (score=0.9985) - The NINDS supports research on conditions such as neurogenic orthostatic hypotension aimed at increasing scientific understanding of the condition and finding ways to treat and prevent it.
2. `medquad_ans_0000093-4__dup1` (score=0.9985) - The NINDS supports and conducts research on movement disorders such as opsoclonus myoclonus. These studies are aimed at increasing knowledge about these disorders and finding ways to prevent, treat, and cure them.
3. `medquad_ans_0000264-4__dup1` (score=0.9982) - The NINDS conducts and supports a broad range of research on movement disorders including tardive dyskinesia. The goals of this research are to improve understanding of these disorders and to discover ways to treat, p...
4. `medquad_ans_0000224-4__dup1` (score=0.9981) - NINDS supports and conducts research on movement disorders such as paroxysmal choreoathetosis. Much of this research is aimed at finding ways to prevent and treat these disorders.
5. `medquad_ans_0000204-4__dup1` (score=0.9978) - The NINDS supports research on disorders of the autonomic nervous system, including multiple system atrophy with orthostatic hypotension. This research is aimed at developing techniques to diagnose, treat, and prevent...

**Answer**:

The provided context does not mention research or clinical trials for Neuroleptic Malignant Syndrome. Therefore, based on the given information, no specific details about research or clinical trials for this condition are available. [doc_id] not available

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
5. `medquad_ans_0000141-4__dup2` (score=0.0041) - An ectopic kidney may not cause any symptoms and may function normally, even though it is not in its usual position. Many people have an ectopic kidney and do not discover it until they have tests done for other reaso...

**Answer**:

To diagnose kidney stones in adults, a healthcare provider will perform a physical exam and take a medical history, which may include questions about family history, diet, gastrointestinal problems, and other diseases. Diagnostic tests include urinalysis, blood tests, abdominal x-ray, and CT scans. Urinalysis tests a urine sample to detect infection or substances that form stones. Blood tests identify biochemical abnormalities that may lead to kidney stones. An abdominal x-ray uses minimal radiation to show the location of stones in the kidney or urinary tract. CT scans use x-rays and computer technology to create three-dimensional images and can show stone locations and conditions that may have caused the stone to form, possibly with the use of contrast medium [medquad_ans_0000203-8].

**Supported citations**: `medquad_ans_0000203-8`
**Unsupported citations**: _none_

### Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000041_1-1` (score=0.0700) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of ...
2. `medquad_ans_0000036_1-1` (score=0.0131) - Key Points - Prostate cancer is a disease in which malignant (cancer) cells form in the tissues of the prostate. - Signs of prostate cancer include a weak flow of urine or frequent urination. - Tests that examine the ...
3. `medquad_ans_0000211-4__dup2` (score=0.0122) - Urinary incontinence in men results when the brain does not properly signal the bladder, the sphincters do not squeeze strongly enough, or both. The bladder muscle may contract too much or not enough because of a prob...
4. `medquad_ans_0000036_2-1` (score=0.0043) - Key Points - Prostate cancer is a disease in which malignant (cancer) cells form in the tissues of the prostate. - Prostate cancer is the second most common cancer among men in the United States. Prostate cancer is a ...
5. `medquad_ans_0000041_1-5` (score=0.0036) - Key Points - After urethral cancer has been diagnosed, tests are done to find out if cancer cells have spread within the urethra or to other parts of the body. - There are three ways that cancer spreads in the body. -...

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. The urethra is the tube that carries urine from the bladder to outside the body. In women, the urethra is about 1 inch long and is just above the vagina. In men, the urethra is about 8 inches long, and goes through the prostate gland and the penis to the outside of the body. In men, the urethra also carries semen. Urethral cancer is a rare cancer that occurs more often in men than in women. There are different types of urethral cancer that begin in cells that line the urethra, named for the types of cells that become malignant: squamous cell carcinoma, transitional cell carcinoma, and adenocarcinoma. Squamous cell carcinoma is the most common type and forms in cells near the bladder in women and in the lining

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000162-4` (score=0.9987) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with CAV3-relat...
2. `medquad_ans_0000148-4` (score=0.9974) - Cap myopathy is an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases are not inherited; they result from new mutations in the gene and ...
3. `medquad_ans_0000717-4` (score=0.0082) - Nemaline myopathy is usually inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry ...
4. `medquad_ans_0000289-2` (score=0.0012) - DCMA syndrome is a very rare disorder. Approximately 30 cases have been identified in the Dariusleut Hutterite population of the Great Plains region of Canada. Only a few affected individuals have been identified outs...
5. `medquad_ans_0000700-2` (score=0.0005) - The incidence of MYH9-related disorder is unknown. More than 200 affected families have been reported in the scientific literature.

**Answer**:

The provided context does not contain any information about the inheritance pattern of paramyotonia congenita. Therefore, it is not possible to determine based on the given text whether paramyotonia congenita is inherited.

Answer: Insufficient context [medquad_ans_0000162-4, medquad_ans_0000148-4, medquad_ans_0000717-4, medquad_ans_0000289-2, medquad_ans_0000700-2].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000162-4, medquad_ans_0000148-4, medquad_ans_0000717-4, medquad_ans_0000289-2, medquad_ans_0000700-2`

### Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.530, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000735-2__dup1` (score=0.9987) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound bio...
2. `medquad_ans_0000909-2` (score=0.0002) - Fewer than 10 people worldwide have been diagnosed with sialuria. There are probably more people with the disorder who have not been diagnosed, as sialuria can be difficult to detect because of its variable features.
3. `medquad_ans_0000556-2` (score=0.0002) - KBG syndrome is a rare disorder that has been reported in around 60 individuals. For unknown reasons, males are affected more often than females. Doctors think the disorder is underdiagnosed because the signs and symp...
4. `medquad_ans_0000459-2` (score=0.0002) - The prevalence of hereditary folate malabsorption is unknown. Approximately 15 affected families have been reported worldwide. Researchers believe that some infants with this disorder may not get diagnosed or treated,...
5. `medquad_ans_0000125-2` (score=0.0001) - Profound or partial biotinidase deficiency occurs in approximately 1 in 60,000 newborns

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), reduced consciousness/confusion (7.5%), respiratory insufficiency (7.5%), skin ulcer (7.5%), and visual field defect (7.5%). Other associated symptoms may include feeding difficulties in infancy, recurrent skin infections, seborrheic dermatitis, sensorineural hearing impairment, hepatomegaly, hyperammonemia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Scores**: context_relevance=0.571, groundedness=0.943, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000966-1__dup2` (score=0.9971) - Summary : We all need clean water. People need it to grow crops and to operate factories, and for drinking and recreation. Fish and wildlife depend on it to survive. Many different pollutants can harm our rivers, stre...
2. `medquad_ans_0000298-1__dup1` (score=0.0371) - Summary : We all need to drink water. How much you need depends on your size, activity level, and the weather where you live. The water you drink is a combination of surface water and groundwater. Surface water includ...
3. `medquad_ans_0000911-1__dup1` (score=0.0110) - Summary : Traveling can increase your chances of getting sick. A long flight can increase your risk for deep vein thrombosis. Once you arrive, it takes time to adjust to the water, food, and air in another place. Wate...
4. `medquad_ans_0000422-1__dup1` (score=0.0046) - Summary : Even if you use them properly, many chemicals can still harm human health and the environment. When you throw these substances away, they become hazardous waste. Some hazardous wastes come from products in o...
5. `medquad_ans_0000331-1__dup1` (score=0.0012) - Summary : Our environment affects our health. If parts of the environment, like the air, water, or soil become polluted, it can lead to health problems. For example, asthma pollutants and chemicals in the air or in th...

**Answer**:

Yes, there is information about water pollution. The three most common pollutants are soil, nutrients, and bacteria. Rain washes soil into streams and rivers, which can kill tiny animals and fish eggs, clog fish gills, and block light, causing plants to die. Nutrients, often from fertilizers, cause algae growth in lakes, ponds, and reservoirs, turning water green. Bacteria, often from sewage spills, can pollute both fresh and salt water. To help protect water supplies, do not pour household products like cleansers, medicines, or auto fluids down the drain; instead, take them to a hazardous waste collection site. Throw away excess grease, diapers, condoms, and personal hygiene products in the garbage. Also, clean up after pets, as pet waste contains nutrients and germs [medquad_ans_0000966-1__dup2].

**Supported citations**: `medquad_ans_0000966-1__dup2`
**Unsupported citations**: _none_

### Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Scores**: context_relevance=0.714, groundedness=0.973, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000067-24` (score=0.9986) - Most people first notice symptoms, such as tingling or numbness, in their hands or feet. Other common symptoms include sudden or sharp pain sensations, loss of sensation of touch, loss of balance or difficulty walking...
2. `medquad_ans_0000132-3__dup1` (score=0.9937) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is...
3. `medquad_ans_0000067-25` (score=0.9845) - Talk to your doctor when you first notice symptoms of neuropathy. Certain medications and other approaches, such as physical therapy, may help alleviate symptoms. There are some steps you can take yourself. Pay carefu...
4. `medquad_ans_0000074-3__dup1` (score=0.0237) - Onset of symptoms of CMT is most often in adolescence or early adulthood, however presentation may be delayed until mid-adulthood. Progression of symptoms is very gradual. The degeneration of motor nerves results in m...
5. `medquad_ans_0000034-3__dup4` (score=0.0144) - Symptoms depend on the type of neuropathy and which nerves are affected. Some people with nerve damage have no symptoms at all. For others, the first symptom is often numbness, tingling, or pain in the feet. Symptoms ...

**Answer**:

The outlook for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1].

**Supported citations**: `medquad_ans_0000132-3__dup1`
**Unsupported citations**: _none_

### Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000145-1` (score=0.7351) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and h...
2. `medquad_ans_0000915-1__dup1` (score=0.3966) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
3. `medquad_ans_0000916-1__dup1` (score=0.3966) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
4. `medquad_ans_0000051-11` (score=0.0013) - Pain may be a symptom, especially among people with more advanced Paget's disease. Affected bones also can become enlarged, misshapen, and more fragile and likely to break. Misshapen bones tend to be most noticeable i...
5. `medquad_ans_0000513-2` (score=0.0006) - Imerslund-Grsbeck syndrome is a rare condition that was first described in Finland and Norway; in these regions, the condition is estimated to affect 1 in 200,000 people. The condition has also been reported in other ...

**Answer**:

Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are also affected. The thickened bones can lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. An increase in the density of the skull results in increased pressure on the brain and can cause a variety of neurological problems, including headaches, hearing loss, vision problems, dizziness (vertigo), ringing in the ears (tinnitus), and facial paralysis. The added pressure that thickened bones put on the muscular and skeletal systems can cause abnormal curvature of the spine (scoliosis), joint deformities (contractures), knock knees, and flat feet (pes planus). Other features include abnormally long limbs in proportion to height, a decrease in muscle mass and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Scores**: context_relevance=0.889, groundedness=0.762, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000935-2` (score=0.0002) - Spinal muscular atrophy affects 1 in 6,000 to 1 in 10,000 people.
2. `medquad_ans_0000059-2` (score=0.0001) - The prevalence of the X-linked recessive type of EDA-ID is estimated to be 1 in 250,000 individuals. Only a few cases of the autosomal dominant form have been described in the scientific literature.
3. `medquad_ans_0000208-3__dup1` (score=0.0001) - The prognosis for individuals with a myopathy varies. Some individuals have a normal life span and little or no disability. For others, however, the disorder may be progressive, severely disabling, life-threatening, o...
4. `medquad_ans_0000316-4` (score=0.0001) - Emery-Dreifuss muscular dystrophy can have several different patterns of inheritance. When this condition is caused by mutations in the EMD gene, it is inherited in an X-linked recessive pattern. A condition is consid...
5. `medquad_ans_0000294-2` (score=0.0001) - DMD-associated dilated cardiomyopathy appears to be an uncommon condition, although its prevalence is unknown.

**Answer**:

The provided context does not specify the symptoms of Emery-Dreifuss muscular dystrophy, X-linked. It describes the inheritance patterns and genetic causes but does not list the clinical symptoms associated with the X-linked form of the disorder. [medquad_ans_0000316-4] discusses inheritance patterns but does not detail symptoms. Therefore, based on the given information, the symptoms cannot be determined.

**Supported citations**: `medquad_ans_0000316-4`
**Unsupported citations**: _none_

### Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.804, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000072-5` (score=0.9945) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins Mc...
2. `medquad_ans_0000275-2` (score=0.1121) - There is no cure for Williams syndrome, nor is there a standard course of treatment. Because WS is an uncommon and complex disorder, multidisciplinary clinics have been established at several centers in the United Sta...
3. `medquad_ans_0000860-2__dup1` (score=0.0839) - How might Brown-Sequard syndrome be treated?
4. `medquad_ans_0006111-6` (score=0.0412) - How might Tietze syndrome be treated? In some individuals, the pain associated with Tietze syndrome resolves on its own without any treatment. Management options for others may include avoidance of strenuous activity;...
5. `medquad_ans_0001025-5__dup1` (score=0.0074) - How might celiac artery compression syndrome be treated? Surgery is currently the only treatment option for celiac artery compression syndrome. Surgery typically involves decompression of the celiac artery by dividing...

**Answer**:

The provided context does not specify the treatments for arterial tortuosity syndrome. While resources for diagnosis and management are listed, including genetic testing, genetic counseling, and information from the National Heart, Lung, and Blood Institute, no specific treatment options are detailed [medquad_ans_0000072-5]. Therefore, based on the available information, the treatments for arterial tortuosity syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

### Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Scores**: context_relevance=1.000, groundedness=0.929, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001092-1__dup1` (score=0.9994) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
2. `medquad_ans_0000074-3__dup1` (score=0.9939) - Onset of symptoms of CMT is most often in adolescence or early adulthood, however presentation may be delayed until mid-adulthood. Progression of symptoms is very gradual. The degeneration of motor nerves results in m...
3. `medquad_ans_0001090-2__dup1` (score=0.0009) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1A? CMT1 is generally slowly progressive over many years. However, affected individuals often experience long periods without any obvious deteriorati...
4. `medquad_ans_0000172-2` (score=0.0001) - Char syndrome is rare, although its exact incidence is unknown. Only a few families with this condition have been identified worldwide.
5. `medquad_ans_0001105-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is ava...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 1C include decreased motor nerve conduction velocity, distal amyotrophy, distal muscle weakness, distal sensory impairment, hypertrophic nerve changes, hyporeflexia, juvenile onset, onion bulb formation, pes cavus, and segmental peripheral demyelination/remyelination [medquad_ans_0001092-1__dup1]. The frequency of these symptoms is estimated based on studies, with some frequencies listed as percentages or fractions representing the proportion of patients exhibiting the symptom [medquad_ans_0001092-1__dup1].

**Supported citations**: `medquad_ans_0001092-1__dup1`
**Unsupported citations**: _none_

### Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.927, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005321-1` (score=0.9994) - What are the signs and symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal arterial macroaneurysm...
2. `medquad_ans_0000220-3__dup1` (score=0.0050) - There is no cure for OPCA. The disorder is slowly progressive with death usually occurring approximately 20 years after onset.
3. `medquad_ans_0000630-2` (score=0.0023) - MRKH syndrome affects approximately 1 in 4,500 newborn girls.
4. `medquad_ans_0001036-2` (score=0.0012) - Williams syndrome affects an estimated 1 in 7,500 to 10,000 people.
5. `medquad_ans_0000149-2` (score=0.0005) - CM-AVM is thought to occur in at least 1 in 100,000 people of northern European origin. The prevalence of the condition in other populations is unknown.

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis include exudative retinal detachment and pulmonic stenosis [medquad_ans_0005321-1].

**Supported citations**: `medquad_ans_0005321-1`
**Unsupported citations**: _none_

### Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Scores**: context_relevance=0.800, groundedness=0.980, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0001838-1` (score=0.3892) - Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. Symptoms have varied greatly among the few reported cases. Dicarboxylic aminoaciduria is...
2. `medquad_ans_0001081-2` (score=0.0017) - 2-hydroxyglutaric aciduria is a rare disorder. D-2-HGA and L-2-HGA have each been reported to affect fewer than 150 individuals worldwide. Combined D,L-2-HGA appears to be even rarer, with only about a dozen reported ...
3. `medquad_ans_0000959-2` (score=0.0017) - Approximately 350 people with succinic semialdehyde dehydrogenase deficiency have been reported worldwide.
4. `medquad_ans_0000539-2` (score=0.0015) - Isovaleric acidemia is estimated to affect at least 1 in 250,000 people in the United States.
5. `medquad_ans_0000416-2` (score=0.0012) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...

**Answer**:

Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. It is caused by mutations in the SLC1A1 gene and is inherited in an autosomal recessive fashion [medquad_ans_0001838-1].

**Supported citations**: `medquad_ans_0001838-1`
**Unsupported citations**: _none_

### Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Scores**: context_relevance=1.000, groundedness=0.971, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0005860-3` (score=0.9863) - Is spondylothoracic dysostosis genetic? Yes. Spondylothoracic dysostosis is caused by mutations in the MESP2 gene. It is inherited in an autosomal recessive fashion.
2. `medquad_ans_0000944-4` (score=0.9726) - Spondylocostal dysostosis can have different inheritance patterns. Types 1, 2, 3, and 4 are inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of ...
3. `medquad_ans_0000943-4` (score=0.6467) - Spondylocarpotarsal synostosis syndrome caused by FLNB gene mutations is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with ...
4. `medquad_ans_0005840-3` (score=0.0003) - How is spondyloepiphyseal dysplasia congenita inherited? Spondyloepiphyseal dysplasia (SEDC) is typically inherited in an autosomal dominant manner. This means that one altered (mutated) gene in each cell is sufficien...
5. `medquad_ans_0005859-1` (score=0.0001) - Spondylospinal thoracic dysostosis is an extremely rare skeletal disorder characterized by a short, curved spine and fusion of the spinous processes, short thorax with 'crab-like' configuration of the ribs, underdevel...

**Answer**:

Yes, spondylothoracic dysostosis is inherited. It is caused by mutations in the MESP2 gene and is inherited in an autosomal recessive fashion [medquad_ans_0005860-3].

**Supported citations**: `medquad_ans_0005860-3`
**Unsupported citations**: _none_

### Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.971, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000542-1` (score=0.9244) - Jervell and Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). This disorder is a form of long QT syndrome, which is a heart ...
2. `medquad_ans_0003296-1` (score=0.8756) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. I...
3. `medquad_ans_0000038-3__dup1` (score=0.0043) - Early and accurate diagnosis is key to prolonged survival for boys born with Barth syndrome. The disorder was once considered uniformly fatal in infancy, but some individuals are now living much longer. Severe infecti...
4. `medquad_ans_0000058-2` (score=0.0011) - Angelman syndrome affects an estimated 1 in 12,000 to 20,000 people.
5. `medquad_ans_0000701-2` (score=0.0010) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.

**Answer**:

Jervell Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). It is a form of long QT syndrome, which causes the heart muscle to take longer than usual to recharge between beats. Beginning in early childhood, the irregular heartbeats increase the risk of fainting (syncope) and sudden death. The syndrome has two types, type 1 and type 2, and is inherited in an autosomal recessive fashion [medquad_ans_0000542-1][medquad_ans_0003296-1].

**Supported citations**: `medquad_ans_0000542-1`, `medquad_ans_0003296-1`
**Unsupported citations**: _none_

### Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Scores**: context_relevance=0.167, groundedness=0.000, answer_relevance=0.433

**Top docs**:

1. `medquad_ans_0000904-2` (score=0.9995) - This disorder is thought to affect approximately 1 in 35,000 to 50,000 newborns.
2. `medquad_ans_0000146-2` (score=0.9993) - While this condition occurs in people of all ethnic backgrounds, it is most common in people of Ashkenazi (eastern and central European) Jewish heritage. Studies suggest that this disorder affects 1 in 6,400 to 13,500...
3. `medquad_ans_0000631-2` (score=0.9993) - McCune-Albright syndrome occurs in between 1 in 100,000 and 1 in 1,000,000 people worldwide.
4. `medquad_ans_0000245-2` (score=0.9993) - This condition is very rare; it probably affects 200 to 300 people worldwide. Reported estimates of Costello syndrome prevalence range from 1 in 300,000 to 1 in 1.25 million people.
5. `medquad_ans_0000557-2` (score=0.9989) - The prevalence of Kearns-Sayre syndrome is approximately 1 to 3 per 100,000 individuals.

**Answer**:

The provided context does not contain information about the prevalence of congenital cataracts, facial dysmorphism, and neuropathy. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Scores**: context_relevance=0.846, groundedness=0.953, answer_relevance=0.908

**Top docs**:

1. `medquad_ans_0000040-3` (score=0.9988) - Alpha thalassemia X-linked intellectual disability syndrome results from mutations in the ATRX gene. This gene provides instructions for making a protein that plays an essential role in normal development. Although th...
2. `medquad_ans_0000373-3` (score=0.0149) - Researchers have identified changes in five regions of the X chromosome that are linked to FG syndrome in affected families. Mutations in a gene called MED12, which is located in one of these regions, appear to be the...
3. `medquad_ans_0001077-2` (score=0.0004) - 18q deletion syndrome occurs in an estimated 1 in 40,000 newborns. This condition is found in people of all ethnic backgrounds.
4. `medquad_ans_0000030-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered or deleted gene in each cell is sufficient to cause the disorder. In approximately 30 to 50 percent of cases, an affect...
5. `medquad_ans_0001058-4` (score=0.0001) - This condition is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each cell. In males...

**Answer**:

The genetic changes related to alpha thalassemia X-linked intellectual disability syndrome are mutations in the ATRX gene. These mutations change the structure of the ATRX protein, which likely prevents it from effectively regulating gene expression. Reduced activity of the HBA1 and HBA2 genes, which are necessary for normal hemoglobin production, causes alpha thalassemia. Abnormal expression of other genes, whose identities have not been identified, probably leads to developmental delay, distinctive facial features, and other signs and symptoms of the syndrome [medquad_ans_0000040-3].

**Supported citations**: `medquad_ans_0000040-3`
**Unsupported citations**: _none_

### Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Scores**: context_relevance=1.000, groundedness=0.938, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002622-2` (score=0.9994) - What are the signs and symptoms of Glycogen storage disease type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 6. If the information is available, ...
2. `medquad_ans_0000427-1` (score=0.0001) - Glycogen storage disease type VI (also known as GSDVI or Hers disease) is an inherited disorder caused by an inability to break down a complex sugar called glycogen in liver cells. A lack of glycogen breakdown interfe...
3. `medquad_ans_0000425-1` (score=0.0001) - Glycogen storage disease type IX (also known as GSD IX) is a condition caused by the inability to break down a complex sugar called glycogen. The different forms of the condition can affect glycogen breakdown in liver...
4. `medquad_ans_0000424-5` (score=0.0001) - These resources address the diagnosis or management of glycogen storage disease type IV: - Gene Review: Gene Review: Glycogen Storage Disease Type IV - Genetic Testing Registry: Glycogen storage disease, type IV - Med...
5. `medquad_ans_0002622-1` (score=0.0001) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepa...

**Answer**:

The symptoms of Glycogen storage disease type 6 include hypoglycemia (seen in approximately 90% of patients), short stature (seen in approximately 90% of patients), hepatomegaly (enlarged liver), increased hepatic glycogen content, and postnatal growth retardation. These symptoms typically begin in infancy to early childhood, with the first sign often being an enlarged liver. Hypoglycemia and other symptoms may occur during prolonged periods without food (fasting), and the signs and symptoms tend to improve with age [medquad_ans_0002622-2][medquad_ans_0000427-1].

**Supported citations**: `medquad_ans_0002622-2`, `medquad_ans_0000427-1`
**Unsupported citations**: _none_

### Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Scores**: context_relevance=1.000, groundedness=0.881, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000015-16` (score=0.9892) - Type 1 diabetes is an autoimmune disease. In an autoimmune reaction, antibodies, or immune cells, attach to the body's own healthy tissues by mistake, signaling the body to attack them. At present, scientists do not k...
2. `medquad_ans_0000034-12` (score=0.1013) - Heart failure is caused by other diseases and conditions that damage the heart muscle. It is most commonly caused by coronary artery disease, including heart attack. Diabetes and high blood pressure also contribute to...
3. `medquad_ans_0000033-1__dup4` (score=0.0205) - Diabetic kidney disease, also called diabetic nephropathy, is kidney disease caused by diabetes. Even when well controlled, diabetes can lead to chronic kidney disease (CKD) and kidney failure, described as end-stage ...
4. `medquad_ans_0000015-2__dup2` (score=0.0091) - Diabetes is a serious, life-long disease. It can lead to problems such as heart disease, stroke, vision loss, kidney disease, and nerve damage. More than 8 million people in the United States have type 2 diabetes and ...
5. `medquad_ans_0000035-2__dup3` (score=0.0052) - Prediabetes is when blood glucose levels are higher than normal but not high enough for a diagnosis of diabetes. Prediabetes means a person is at increased risk for developing type 2 diabetes, as well as for heart dis...

**Answer**:

The causes of diabetes are not fully known, but for type 1 diabetes, it is believed that both genetic and environmental factors are involved, as the immune system mistakenly attacks the insulin-producing cells in the pancreas [medquad_ans_0000015-16]. For type 2 diabetes, the most common form, it is caused by a combination of factors including insulin resistance, where the body's muscle, fat, and liver cells do not use insulin effectively, and the body's inability to produce enough insulin to compensate for this resistance [medquad_ans_0000015-16]. However, the exact causes of diabetes are not specified in the provided context. [medquad_ans_0000015-2__dup2]

**Supported citations**: `medquad_ans_0000015-16`, `medquad_ans_0000015-2__dup2`
**Unsupported citations**: _none_

### Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Scores**: context_relevance=0.545, groundedness=0.925, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000016-26` (score=0.9974) - National Eye Institute National Institutes of Health 2020 Vision Place Bethesda, MD 20892-3655 301-496-5248 E-mail: 2020@nei.nih.gov www.nei.nih.gov Find eye health organizations that address diabetic eye disease.
2. `medquad_ans_0000016-21` (score=0.9970) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the lab...
3. `medquad_ans_0000027-22` (score=0.9942) - Through studies in the laboratory and with patients, the National Eye Institute is seeking better ways to detect, treat, and prevent vision loss in people with glaucoma. For example, researchers have discovered genes ...
4. `medquad_ans_0000001-22` (score=0.9709) - Once dry AMD reaches the advanced stage, no form of treatment can prevent vision loss. However, treatment can delay and possibly prevent intermediate AMD from progressing to the advanced stage, in which vision loss oc...
5. `medquad_ans_0000016-19` (score=0.0032) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although bo...

**Answer**:

The National Eye Institute (NEI) is conducting and supporting research to seek better ways to detect, treat, and prevent vision loss in people with diabetes. This research includes studies in the laboratory and with patients, such as investigating drugs for the treatment of proliferative retinopathy that may reduce the need for laser surgery [medquad_ans_0000016-21].

**Supported citations**: `medquad_ans_0000016-21`
**Unsupported citations**: _none_

### Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Scores**: context_relevance=0.800, groundedness=0.866, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000012-7__dup1` (score=0.9702) - Chronic obstructive pulmonary disease, or COPD, is a progressive lung disease in which the airways of the lungs become damaged, making it harder to breathe. With COPD, airways become blocked, making it harder to get a...
2. `medquad_ans_0000516-1__dup1` (score=0.9100) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart an...
3. `medquad_ans_0000379-1__dup1` (score=0.8206) - If you've ever gotten a splinter or had sand in your eye, you've had experience with a foreign body. A foreign body is something that is stuck inside you but isn't supposed to be there. You may inhale or swallow a for...
4. `medquad_ans_0000676-1__dup2` (score=0.0673) - Oxygen therapy is a treatment that provides you with extra oxygen. Oxygen is a gas that your body needs to function. Normally, your lungs absorb oxygen from the air you breathe. But some conditions can prevent you fro...
5. `medquad_ans_0000012-15` (score=0.0206) - If you smoke, the most important thing you can do to prevent more lung damage is to stop smoking. It is also important to stay away from people who smoke and places where you know there will be smokers. Avoid exposure...

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles from fires or toxic fumes, which can damage the eyes and respiratory system. These injuries may worsen existing chronic heart and lung diseases. Symptoms include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, the condition may worsen. The best prevention is to limit exposure, especially when smoke is detected or fires are nearby [medquad_ans_0000516-1__dup1].

**Supported citations**: `medquad_ans_0000516-1__dup1`
**Unsupported citations**: _none_

### Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Scores**: context_relevance=0.375, groundedness=0.950, answer_relevance=0.375

**Top docs**:

1. `medquad_ans_0000409-2` (score=0.9994) - Gitelman syndrome affects an estimated 1 in 40,000 people worldwide.
2. `medquad_ans_0000857-2` (score=0.9992) - Rabson-Mendenhall syndrome is estimated to affect less than 1 per million people worldwide. Several dozen cases have been reported in the medical literature.
3. `medquad_ans_0000058-2` (score=0.9991) - Angelman syndrome affects an estimated 1 in 12,000 to 20,000 people.
4. `medquad_ans_0000114-3__dup2` (score=0.9990) - ESS is not a life-threatening condition. Most often, and particularly among those with primary ESS, the disorder does not cause health problems and does not affect life expectancy.
5. `medquad_ans_0000134-4__dup2` (score=0.9982) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has ...

**Answer**:

Zollinger-Ellison syndrome affects about one in every 1 million people worldwide [medquad_ans_0000134-4__dup2].

**Supported citations**: `medquad_ans_0000134-4__dup2`
**Unsupported citations**: _none_

### Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Scores**: context_relevance=0.429, groundedness=0.647, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000974-5` (score=0.9949) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Ge...
2. `medquad_ans_0000482-5` (score=0.0030) - These resources address the diagnosis or management of histidinemia: - Genetic Testing Registry: Histidinemia These resources from MedlinePlus offer information about the diagnosis and management of various health con...
3. `medquad_ans_0000497-5` (score=0.0022) - These resources address the diagnosis or management of hypermethioninemia: - Baby's First Test - Genetic Testing Registry: Glycine N-methyltransferase deficiency - Genetic Testing Registry: Hepatic methionine adenosyl...
4. `medquad_ans_0000459-2` (score=0.0015) - The prevalence of hereditary folate malabsorption is unknown. Approximately 15 affected families have been reported worldwide. Researchers believe that some infants with this disorder may not get diagnosed or treated,...
5. `medquad_ans_0000459-5` (score=0.0014) - These resources address the diagnosis or management of hereditary folate malabsorption: - Gene Review: Gene Review: Hereditary Folate Malabsorption - Genetic Testing Registry: Congenital defect of folate absorption - ...

**Answer**:

The provided context does not specify the treatments for tetrahydrobiopterin deficiency. While it lists diagnostic resources such as genetic testing and serum phenylalanine screening, it does not mention any therapeutic interventions or treatment options [medquad_ans_0000974-5]. Therefore, based on the given information, the treatments for tetrahydrobiopterin deficiency cannot be determined.

**Supported citations**: `medquad_ans_0000974-5`
**Unsupported citations**: _none_

### Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Scores**: context_relevance=0.250, groundedness=0.000, answer_relevance=0.050

**Top docs**:

1. `medquad_ans_0000501-2` (score=0.9993) - Hypochondrogenesis and achondrogenesis, type 2 (a similar skeletal disorder) together affect 1 in 40,000 to 60,000 newborns.
2. `medquad_ans_0001036-2` (score=0.9993) - Williams syndrome affects an estimated 1 in 7,500 to 10,000 people.
3. `medquad_ans_0000248-2` (score=0.9993) - Craniofacial microsomia has been estimated to occur in between 1 in 5,600 and 1 in 26,550 newborns. However, this range may be an underestimate because not all medical professionals agree on the criteria for diagnosis...
4. `medquad_ans_0000752-2` (score=0.9992) - X-linked Opitz G/BBB syndrome is thought to affect 1 in 10,000 to 50,000 males, although it is likely that this condition is underdiagnosed. The incidence of autosomal dominant Opitz G/BBB syndrome is unknown. It is p...
5. `medquad_ans_0000064-2` (score=0.9991) - Apert syndrome affects an estimated 1 in 65,000 to 88,000 newborns.

**Answer**:

The provided context does not contain information about the prevalence or incidence of nonsyndromic holoprosencephaly. Therefore, it is not possible to answer the question based on the given text.

Answer: Insufficient context [doc_id].

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Scores**: context_relevance=0.727, groundedness=0.762, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000525-3` (score=0.0013) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt ex...
2. `medquad_ans_0000069-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means a genetic rearrangement involving one copy of the CYP19A1 gene in each cell is sufficient to cause the disorder. In some cases, an affected per...
3. `medquad_ans_0003240-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy (ICP) is a disorder of the liver that occurs in women during pregnancy. Cholestasis is a condition that impairs the release of bile (a digestive juice) from liver cells. The bile ...
4. `medquad_ans_0000183-4__dup3` (score=0.0001) - Biliary atresia likely has multiple causes, though none are yet proven. Biliary atresia is not an inherited disease, meaning it does not pass from parent to child. Therefore, survivors of biliary atresia are not at ri...
5. `medquad_ans_0000228-3` (score=0.0001) - Syndromes of which congenital hepatic fibrosis is a feature may be caused by changes in many different genes. The gene changes that cause isolated congenital hepatic fibrosis are unknown. Congenital hepatic fibrosis i...

**Answer**:

The provided context does not mention genetic changes related to progressive familial intrahepatic cholestasis. The information discusses genetic changes associated with intrahepatic cholestasis of pregnancy (ICP), specifically involving the ABCB11 and ABCB4 genes, but does not address progressive familial intrahepatic cholestasis. Therefore, based on the given context, the genetic changes related to progressive familial intrahepatic cholestasis cannot be determined.

[medquad_ans_0000525-3]

**Supported citations**: `medquad_ans_0000525-3`
**Unsupported citations**: _none_

### Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.9994) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below in...
2. `medquad_ans_0000630-2` (score=0.0019) - MRKH syndrome affects approximately 1 in 4,500 newborn girls.
3. `medquad_ans_0000543-2` (score=0.0008) - Joubert syndrome is estimated to affect between 1 in 80,000 and 1 in 100,000 newborns. However, this estimate may be too low because Joubert syndrome has such a large range of possible features and is likely underdiag...
4. `medquad_ans_0000631-2` (score=0.0005) - McCune-Albright syndrome occurs in between 1 in 100,000 and 1 in 1,000,000 people worldwide.
5. `medquad_ans_0000560-2` (score=0.0005) - The prevalence of Kleefstra syndrome is unknown. Only recently has testing become available to distinguish it from other disorders with similar features.

**Answer**:

The symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

### Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Scores**: context_relevance=0.571, groundedness=0.000, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0002550-3` (score=0.0268) - How is Gilbert syndrome inherited? Gilbert syndrome is inherited in an autosomal recessive manner, which means both copies of the gene in each cell have mutations. The parents of a person with an autosomal recessive c...
2. `medquad_ans_0000652-2` (score=0.0009) - More than 200 people with mevalonate kinase deficiency have been reported worldwide; the majority of these individuals have HIDS.
3. `medquad_ans_0000452-2` (score=0.0007) - Hepatic lipase deficiency is likely a rare disorder; only a few affected families have been reported in the scientific literature.
4. `medquad_ans_0000851-2` (score=0.0004) - Purine nucleoside phosphorylase deficiency is rare; only about 70 affected individuals have been identified. This disorder accounts for approximately 4 percent of all SCID cases.
5. `medquad_ans_0000459-2` (score=0.0004) - The prevalence of hereditary folate malabsorption is unknown. Approximately 15 affected families have been reported worldwide. Researchers believe that some infants with this disorder may not get diagnosed or treated,...

**Answer**:

The provided context does not contain information about hepatic veno-occlusive disease with immunodeficiency or its inheritance pattern. Therefore, based on the given context, it is not possible to determine whether this condition is inherited.

Answer: Insufficient context [no_doc_id].

**Supported citations**: _none_
**Unsupported citations**: `no_doc_id`

### Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Scores**: context_relevance=1.000, groundedness=0.928, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005902-1` (score=0.9993) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how o...
2. `medquad_ans_0001062-2` (score=0.9733) - This form of anemia is uncommon. However, researchers believe that it may not be as rare as they once thought. Increased awareness of the disease has led to more frequent diagnoses.
3. `medquad_ans_0000909-2` (score=0.0118) - Fewer than 10 people worldwide have been diagnosed with sialuria. There are probably more people with the disorder who have not been diagnosed, as sialuria can be difficult to detect because of its variable features.
4. `medquad_ans_0000529-2` (score=0.0014) - Although iron deficiency anemia is relatively common, the prevalence of the iron-refractory form of the disease is unknown. At least 50 cases have been described in the medical literature. Researchers suspect that iro...
5. `medquad_ans_0000182-2` (score=0.0011) - It is estimated that 500 to 1,000 people worldwide have chorea-acanthocytosis.

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. These signs and symptoms are based on the Human Phenotype Ontology (HPO) data, which compiles information on how often a symptom occurs in individuals with this condition [medquad_ans_0005902-1].

**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

### Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Scores**: context_relevance=0.833, groundedness=0.944, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000159-7` (score=0.9732) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may n...
2. `medquad_ans_0000159-8` (score=0.9253) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed...
3. `medquad_ans_0000144-8` (score=0.0015) - A number of different diseases can result in glomerular disease. It may be the direct result of an infection or a drug toxic to the kidneys, or it may result from a disease that affects the entire body, like diabetes ...
4. `medquad_ans_0000144-9` (score=0.0015) - A number of different diseases can result in glomerular disease. It may be the direct result of an infection or a drug toxic to the kidneys, or it may result from a disease that affects the entire body, like diabetes ...
5. `medquad_ans_0000142-2` (score=0.0011) - C3 glomerulopathy is very rare, affecting 1 to 2 per million people worldwide. It is equally common in men and women.

**Answer**:

For lupus nephritis, treatment involves medications that suppress the immune system to prevent further damage to the kidneys. Standard treatment includes a corticosteroid, usually prednisone, to reduce inflammation in the kidneys, and an immunosuppressive medication such as cyclophosphamide or mycophenolate mofetil, typically used in combination with prednisone [medquad_ans_0000159-8]. If lupus nephritis is causing high blood pressure, medications to lower blood pressure may be needed, as this can significantly slow the progression of kidney disease [medquad_ans_0000159-8]. Additionally, people with kidney disease may need to consult a health care provider about dietary changes, such as reducing sodium intake, especially if they have high blood pressure [medquad_ans_0000159-7]. In many cases, treatment is effective in controlling lup

**Supported citations**: `medquad_ans_0000159-8`, `medquad_ans_0000159-7`
**Unsupported citations**: _none_

### Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.497, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000869-5` (score=0.9811) - These resources address the diagnosis or management of restless legs syndrome: - Agency for Healthcare Research and Quality: Options for Treating Restless Legs Syndrome - Genetic Testing Registry: Restless legs syndro...
2. `medquad_ans_0000114-6` (score=0.9709) - Restless legs syndrome (RLS) has no cure. If a condition or medicine triggers RLS, it may go away or get better if the trigger is relieved or stopped. RLS can be treated. The goals of treatment are to: Prevent or reli...
3. `medquad_ans_0000211-3__dup1` (score=0.0757) - None of the currently available medications enables people with narcolepsy to consistently maintain a fully normal state of alertness. But EDS and cataplexy, the most disabling symptoms of the disorder, can be control...
4. `medquad_ans_0000246-3__dup1` (score=0.0164) - RLS is generally a life-long condition for which there is no cure. Symptoms may gradually worsen with age. Nevertheless, current therapies can control the disorder, minimizing symptoms and increasing periods of restfu...
5. `medquad_ans_0000169-4` (score=0.0108) - NINDS supports a broad range of clinical and basic research on diseases causing sleep disorders in an effort to clarify the mechanisms of these conditions and to develop better treatments for them.

**Answer**:

The treatments for Restless Legs Syndrome (RLS) include lifestyle changes and, in more severe cases, daily medicines. For mild cases, lifestyle changes may be the only treatment needed. These include avoiding substances that trigger symptoms, such as alcohol and tobacco, and informing your doctor about all medications you are taking, as some can cause or worsen RLS. Good sleep habits, such as keeping the sleep environment cool, quiet, comfortable, and dark, maintaining a consistent sleep schedule, and avoiding long periods of lying awake at night, can also help. Engaging in a challenging activity before bedtime, such as solving a crossword puzzle, or using relaxation techniques like focusing on breathing, may ease symptoms. Regular, moderate physical activity can also help limit or prevent symptoms. In more severe cases, daily medicines are typically used. The goals of treatment are to prevent or relieve symptoms, increase sleep duration, improve sleep quality, and treat or correct any underlying conditions that may

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Scores**: context_relevance=0.750, groundedness=0.550, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0002573-1` (score=0.6410) - Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue th...
2. `medquad_ans_0000052-3__dup1` (score=0.1073) - Symptoms of brain and spinal cord tumors generally develop slowly and worsen over time unless they are treated. The tumor may be classified as benign or malignant and given a numbered score that reflects its rate of m...
3. `medquad_ans_0000052-4__dup1` (score=0.0216) - Scientists continue to investigate ways to better understand, diagnose, and treat CNS tumors. Experimental treatment options may include new drugs, gene therapy, surgery , radiation, biologic modulators that enhance t...
4. `medquad_ans_0000006_8-5` (score=0.0071) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The size of the tumor. - Where the tumor is in the brain...
5. `medquad_ans_0000006_1-3` (score=0.0066) - The cause of most adult brain and spinal cord tumors is not known.

**Answer**:

Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue that helps repair brain damage in response to injury. Glioblastomas are often very aggressive and grow into surrounding brain tissue. Signs and symptoms, such as headache, nausea, vomiting, and/or drowsiness, may develop when the tumor begins to put excess pressure on the brain. Affected people may also experience other features depending on the size and location of the tumor. In most cases, the exact underlying cause is unknown; however, they can rarely occur in people with certain genetic syndromes such as neurofibromatosis type 1, Turcot syndrome, and Li Fraumeni syndrome. There is currently no cure for glioblastoma. Treatment is palliative and may include surgery

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Scores**: context_relevance=1.000, groundedness=0.991, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0005496-3` (score=0.9944) - What causes schizencephaly? The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, s...
2. `medquad_ans_0003881-3` (score=0.0013) - What causes megalencephaly, polymicrogyria, and hydrocephalus (MPPH) syndrome? The cause of MPPH syndrome is unknown. Infection during pregnancy or fetal accident is thought to be unlikely.
3. `medquad_ans_0000251-4__dup1` (score=0.0004) - The NINDS conducts and supports a wide range of studies that explore the mechanisms of normal brain development. The knowledge gained from these fundamental studies provides the foundation for understanding how to pre...
4. `medquad_ans_0001070-2` (score=0.0001) - 15q13.3 microdeletion likely occurs in about 1 in 40,000 people in the general population. It appears to be more common in people with intellectual disability, epilepsy, schizophrenia, or autism spectrum disorders.
5. `medquad_ans_0000736-2` (score=0.0001) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.

**Answer**:

The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, schizencephaly can affect more than one family member, which supports a genetic cause in some cases. Schizencephaly has also been linked to a variety of non-genetic factors, including young maternal age and certain medications and infections that can cause vascular disruptions (disruption of blood flow or blood supply) in a developing baby [medquad_ans_0005496-3].

**Supported citations**: `medquad_ans_0005496-3`
**Unsupported citations**: _none_

### Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Scores**: context_relevance=0.800, groundedness=0.454, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000052-3` (score=0.9983) - Mutations in several genes can cause familial ALS and contribute to the development of sporadic ALS. Mutations in the C9orf72 gene account for 30 to 40 percent of familial ALS in the United States and Europe. Worldwid...
2. `medquad_ans_0000551-3` (score=0.0068) - Mutations in the ALS2 gene cause most cases of juvenile primary lateral sclerosis. This gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but its function is not fully u...
3. `medquad_ans_0000052-2` (score=0.0029) - About 5,000 people in the United States are diagnosed with ALS each year. Worldwide, this disorder occurs in 2 to 5 per 100,000 individuals. Only a small percentage of cases have a known genetic cause. Among the Chamo...
4. `medquad_ans_0000015-4__dup1` (score=0.0028) - The NINDS supports research on gene-linked neurodegenerative disorders such as Alpers' disease. The goals of this research are to increase scientific understanding of these disorders, and to find ways to prevent, trea...
5. `medquad_ans_0000217-4` (score=0.0014) - The NINDS supports research on disorders of the brain and nervous system such as neurotoxicity, aimed at learning more about these disorders and finding ways to prevent and treat them. Scientists are investigating the...

**Answer**:

Genetic changes related to amyotrophic lateral sclerosis (ALS) include mutations in several genes such as C9orf72, SOD1, TARDBP, and FUS. Mutations in the C9orf72 gene account for 30 to 40 percent of familial ALS in the United States and Europe, while SOD1 mutations cause 15 to 20 percent of familial ALS worldwide. TARDBP and FUS gene mutations each account for about 5 percent of familial ALS cases. Other genes associated with familial ALS contribute to smaller proportions of cases. Approximately 60 percent of individuals with familial ALS have an identified genetic mutation. These mutations can result in disruptions in axon development, impaired transport of materials within axons, or failure to break down toxic substances, leading to motor neuron damage and death. In some cases, the exact mechanism by which the mutations cause ALS is unknown. Additionally, genetic variations that

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.972, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000581-5` (score=0.9964) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Te...
2. `medquad_ans_0003473-6` (score=0.9531) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injectio...
3. `medquad_ans_0001036-2` (score=0.0002) - Williams syndrome affects an estimated 1 in 7,500 to 10,000 people.
4. `medquad_ans_0003473-2` (score=0.0001) - What are the signs and symptoms of Laron syndrome? Laron syndrome is a rare condition in which the body is unable to use growth hormone. The primary symptom is short stature. Although affected people are generally clo...
5. `medquad_ans_0000550-2` (score=0.0001) - Juvenile polyposis syndrome occurs in approximately 1 in 100,000 individuals worldwide.

**Answer**:

The only specific treatment available for Laron syndrome is subcutaneous injections of insulin-like growth factor 1 (IGF-1), a growth-promoting hormone. IGF-1 stimulates linear growth (height) and also improves brain growth and metabolic abnormalities caused by long-term IGF-1 deficiency. It has been shown to raise blood glucose levels, reduce cholesterol, and increase muscle growth. IGF-1 and GH levels should be closely monitored during treatment because overdosage of IGF-1 can cause various health problems [medquad_ans_0003473-6].

**Supported citations**: `medquad_ans_0003473-6`
**Unsupported citations**: _none_

### Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000001-22` (score=0.9922) - Once dry AMD reaches the advanced stage, no form of treatment can prevent vision loss. However, treatment can delay and possibly prevent intermediate AMD from progressing to the advanced stage, in which vision loss oc...
2. `medquad_ans_0000055-7` (score=0.9745) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need othe...
3. `medquad_ans_0000016-19` (score=0.9650) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although bo...
4. `medquad_ans_0000016-15` (score=0.9647) - During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels of blood s...
5. `medquad_ans_0000016-4__dup2` (score=0.8793) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...

**Answer**:

The treatments for diabetic retinopathy include medicines, laser treatment, and vitrectomy. Medicines involve injections into the eye to block a protein that causes abnormal blood vessel growth and fluid leakage, which helps reduce fluid leakage and restore retinal thickness. Laser treatment includes focal laser treatment for macular edema, where small laser burns are placed near the macula to slow fluid leakage, and scatter laser treatment for new, weak blood vessels, which causes them to shrink and reduces the risk of blindness. However, laser treatment often cannot restore vision that has already been lost. In cases of severe bleeding, a vitrectomy surgery may be needed, where the clouded vitreous gel is removed and replaced with a salt solution [medquad_ans_0000055-7]. These treatments are effective in reducing vision loss, especially in proliferative retinopathy, where timely treatment can reduce the risk of blindness by 95 percent,

**Supported citations**: `medquad_ans_0000055-7`
**Unsupported citations**: _none_

### Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Scores**: context_relevance=0.714, groundedness=0.528, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000201-5` (score=0.9976) - These resources address the diagnosis or management of Cole disease: - Genetic Testing Registry: Cole disease These resources from MedlinePlus offer information about the diagnosis and management of various health con...
2. `medquad_ans_0000119-9` (score=0.9946) - Proctitis that is not treated or does not respond to treatment may lead to complications, including - severe bleeding and anemiaa condition in which red blood cells are fewer or smaller than normal, which means less o...
3. `medquad_ans_0001421-3` (score=0.9645) - How might collagenous colitis be treated? Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases the condition may resolve on its own (spontaneous re...
4. `medquad_ans_0001007-5` (score=0.0028) - These resources address the diagnosis or management of ulcerative colitis: - American Society of Colon and Rectal Surgeons - Cedars-Sinai - Crohn's & Colitis Foundation of America: Colitis Diagnosis and Testing - Croh...
5. `medquad_ans_0000116-10` (score=0.0008) - No. Unlike the other inflammatory bowel diseases, such as Crohns disease and ulcerative colitis, microscopic colitis does not increase a persons risk of getting colon cancer.

**Answer**:

Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases, the condition may resolve on its own (spontaneous remission), although most people continue to experience ongoing or occasional diarrhea. Dietary changes are usually tried first to alleviate symptoms, which may include a reduced-fat diet, eliminating foods that contain caffeine and lactose, and avoiding over-the-counter pain relievers such as ibuprofen or aspirin. If these changes are not sufficient, medications can be used to control symptoms. Prescription anti-inflammatory medications such as mesalamine and sulfasalazine may help reduce swelling. Steroids including budesonide and prednisone can be used to reduce inflammation, but they are typically reserved for sudden attacks of diarrhea and are usually avoided for long-term use due to side effects. Anti-diarrheal medications such as bismuth subsalicylate, diphenoxylate with atropine,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Scores**: context_relevance=0.556, groundedness=0.872, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000241-7` (score=0.9408) - The best way to avoid hookworm infection is not to walk barefoot in areas where hookworm is common and where there may be human fecal contamination of the soil. Also, avoid other skin contact with such soil and avoid ...
2. `medquad_ans_0000265-4__dup2` (score=0.0315) - Loiasis is an infection caused by the parasitic worm Loa loa.
3. `medquad_ans_0000418-1__dup2` (score=0.0259) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...
4. `medquad_ans_0000030-1__dup4` (score=0.0163) - Ascaris is an intestinal parasite of humans. It is the most common human worm infection. The larvae and adult worms live in the small intestine and can cause intestinal disease.
5. `medquad_ans_0000418-2__dup1` (score=0.0075) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...

**Answer**:

To prevent trichinellosis (also known as trichinosis), avoid consuming raw or undercooked meat, particularly from wild game animals such as bear or cougar, or from domestic pigs and wild boar. Proper cooking of meat to safe internal temperatures kills the Trichinella larvae. Additionally, commercial and home freezing of pork can reduce the risk of infection. Public awareness of the dangers of eating raw or undercooked meat products also contributes to reduced cases. In the United States, the risk of trichinellosis from commercially raised and properly prepared pork is very low due to improved pig-raising practices and freezing methods [medquad_ans_0000418-1\_\_dup2][medquad_ans_0000418-2__dup1].

**Supported citations**: `medquad_ans_0000418-1__dup2`, `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

### Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000182-3__dup1` (score=0.9962) - A combination of autoimmunity, environmental triggers, and a genetic predisposition can lead to autoimmune hepatitis.
2. `medquad_ans_0000536-3` (score=0.9939) - What causes autoimmune hepatitis? Although the exact cause of autoimmune hepatitis is unknown, evidence suggests that liver injury in a patient with autoimmune hepatitis is the result of a cell-mediated immunologic at...
3. `medquad_ans_0000015-16` (score=0.0042) - Type 1 diabetes is an autoimmune disease. In an autoimmune reaction, antibodies, or immune cells, attach to the body's own healthy tissues by mistake, signaling the body to attack them. At present, scientists do not k...
4. `medquad_ans_0000535-3` (score=0.0029) - What causes autoimmune hemolytic anemia? In about half of cases, the cause of autoimmune hemolytic anemia cannot be determined (idiopathic or primary). This condition can also be caused by or occur with another disord...
5. `medquad_ans_0000182-10` (score=0.0016) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing autoimmune hepatitis.

**Answer**:

Autoimmune hepatitis is caused by a combination of autoimmunity, environmental triggers, and a genetic predisposition. The liver injury in autoimmune hepatitis results from a cell-mediated immunologic attack, which may be triggered by genetic factors, viral infections, or chemical agents. Autoimmune hepatitis sometimes occurs in relatives of people with autoimmune diseases, further suggesting a genetic cause [medquad_ans_0000182-3__dup1]. However, the exact cause is unknown [medquad_ans_0000536-3]. Eating, diet, and nutrition do not play a role in causing or preventing autoimmune hepatitis [medquad_ans_0000182-10].

**Supported citations**: `medquad_ans_0000182-3__dup1`, `medquad_ans_0000536-3`, `medquad_ans_0000182-10`
**Unsupported citations**: _none_

### Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Scores**: context_relevance=0.800, groundedness=0.948, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000006_3-1` (score=0.9580) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
2. `medquad_ans_0000006_5-3` (score=0.0033) - Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not having risk factors doesnt mean that you will not get cancer. Talk with ...
3. `medquad_ans_0000006_8-5` (score=0.0022) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The size of the tumor. - Where the tumor is in the brain...
4. `medquad_ans_0000006_1-3` (score=0.0018) - The cause of most adult brain and spinal cord tumors is not known.
5. `medquad_ans_0000006_3-7` (score=0.0017) - Key Points - The grade of the tumor is used to plan cancer treatment. - Low-grade astrocytomas - High-grade astrocytomas - An MRI is done after surgery. The grade of the tumor is used to plan cancer treatment. Staging...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They are tumors that start in star-shaped brain cells called astrocytes, which are a type of glial cell that holds nerve cells in place, brings food and oxygen to them, and helps protect them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

### Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Scores**: context_relevance=0.857, groundedness=0.882, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000013_3-1__dup3` (score=0.8728) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia in...
2. `medquad_ans_0000040-10` (score=0.0081) - Acute leukemia gets worse quickly. In chronic leukemia, symptoms develop gradually and are generally not as severe as in acute leukemia.
3. `medquad_ans_0000013_2-2__dup3` (score=0.0018) - See the PDQ summary about Chronic Myelogenous Leukemia Treatment for information.
4. `medquad_ans_0000013_3-3__dup3` (score=0.0008) - Certain factors affect prognosis (chance of recovery). The prognosis (chance of recovery) for atypical CML depends on the number of red blood cells and platelets in the blood.
5. `medquad_ans_0000040-11` (score=0.0006) - There are four common types of leukemia. They are chronic lymphocytic leukemia, chronic myeloid leukemia, acute myeloid leukemia, and acute lymphocytic leukemia. Chronic lymphocytic leukemia, chronic myeloid leukemia,...

**Answer**:

Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. In this condition, the body tells too many blood stem cells to become granulocytes, and some of these stem cells never mature into full white blood cells; these immature cells are called blasts. Over time, the granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. Atypical CML is similar to chronic myelogenous leukemia (CML) under a microscope, but it does not have the "Philadelphia chromosome" chromosome change that is typically present in CML [medquad_ans_0000013_3-1__dup3]. The prognosis for atypical CML depends on the number of red blood cells and platelets in the blood [medquad_ans_0000013_3-3\_\_dup3

**Supported citations**: `medquad_ans_0000013_3-1__dup3`
**Unsupported citations**: _none_

### Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Scores**: context_relevance=0.750, groundedness=0.952, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000996-3` (score=0.9960) - Most cases of trisomy 13 result from having three copies of chromosome 13 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the cha...
2. `medquad_ans_0000373-3` (score=0.0800) - Researchers have identified changes in five regions of the X chromosome that are linked to FG syndrome in affected families. Mutations in a gene called MED12, which is located in one of these regions, appear to be the...
3. `medquad_ans_0000253-3` (score=0.0071) - In most cases, the cause of CCHD is unknown. A variety of genetic and environmental factors likely contribute to this complex condition. Changes in single genes have been associated with CCHD. Studies suggest that the...
4. `medquad_ans_0000996-4` (score=0.0003) - Most cases of trisomy 13 are not inherited and result from random events during the formation of eggs and sperm in healthy parents. An error in cell division called nondisjunction results in a reproductive cell with a...
5. `medquad_ans_0000303-3` (score=0.0003) - Most cases of Down syndrome result from trisomy 21, which means each cell in the body has three copies of chromosome 21 instead of the usual two copies. Less commonly, Down syndrome occurs when part of chromosome 21 b...

**Answer**:

The genetic changes related to trisomy 13 include having three copies of chromosome 13 in each cell instead of the usual two copies, which disrupts normal development and causes characteristic features of the condition. In some cases, part of chromosome 13 becomes attached (translocated) to another chromosome during the formation of reproductive cells or very early in fetal development, resulting in individuals having two normal copies of chromosome 13 and an extra copy attached to another chromosome. In rare cases, only part of chromosome 13 is present in three copies, leading to potentially different physical signs and symptoms compared to full trisomy 13. Additionally, in mosaic trisomy 13, an extra copy of chromosome 13 is present in only some of the body's cells, and the severity of the condition depends on the type and number of affected cells [medquad_ans_0000996-3].

**Supported citations**: `medquad_ans_0000996-3`
**Unsupported citations**: _none_

### Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Scores**: context_relevance=0.857, groundedness=0.862, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000731-5` (score=0.9866) - These resources address the diagnosis or management of Niemann-Pick disease: - Baby's First Test - Gene Review: Gene Review: Acid Sphingomyelinase Deficiency - Gene Review: Gene Review: Niemann-Pick Disease Type C - G...
2. `medquad_ans_0000218-2__dup1` (score=0.9733) - There is currently no cure for Niemann-Pick disease. Treatment is supportive. Children usually die from infection or progressive neurological loss. There is currently no effective treatment for persons with type A. Bo...
3. `medquad_ans_0004401-5` (score=0.8926) - How might neutral lipid storage disease with myopathy be treated? To date, there is no treatment for the underlying metabolic problem. Current therapies include adhering to strict dietary guidelines and utilizing trea...
4. `medquad_ans_0004415-3` (score=0.8279) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respirato...
5. `medquad_ans_0000014-2__dup1` (score=0.0187) - There is no cure for Alexander disease, nor is there a standard course of treatment. Treatment of Alexander disease is symptomatic and supportive.

**Answer**:

There is currently no cure for Niemann-Pick disease. Treatment is supportive and focuses on managing symptoms. For Niemann-Pick disease type A, there is no specific treatment; supportive care from specialists such as pulmonologists, cardiologists, liver and spleen specialists, nutritionists, physical therapists, and gastroenterologists may be helpful for managing respiratory, cardiac, and other related symptoms [medquad_ans_0004415-3]. For Niemann-Pick disease type B, bone marrow transplantation has been attempted in a few individuals, and enzyme replacement and gene therapies might be helpful [medquad_ans_0000218-2__dup1]. Restricting one's diet does not prevent the buildup of lipids in cells and tissues [medquad_ans_0000218-2__dup1]. Overall, treatment is directed at symptom management rather than addressing the underlying metabolic defect.

**Supported citations**: `medquad_ans_0004415-3`, `medquad_ans_0000218-2__dup1`
**Unsupported citations**: _none_

### Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001714-1` (score=0.9994) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial poly...
2. `medquad_ans_0000290-2` (score=0.0001) - Distal arthrogryposis type 1 affects an estimated 1 in 10,000 people worldwide.
3. `medquad_ans_0000270-2` (score=0.0001) - Dandy-Walker malformation is estimated to affect 1 in 10,000 to 30,000 newborns.
4. `medquad_ans_0000736-2` (score=0.0001) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
5. `medquad_ans_0000067-3__dup1` (score=0.0001) - The prognosis is dependent upon the underlying disorder. Some of the disorders that are associated with cerebellar hypoplasia are progressive, which means the condition will worsen over time, and will most likely have...

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, autosomal recessive inheritance, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, and wide nasal bridge [

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.965, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0005096-2` (score=0.9948) - What causes progressive transformation of germinal centers? Is it genetic? The cause of progressive transformation of germinal centers (PTGC) is currently unknown. Also, there is no evidence in the medical literature ...
2. `medquad_ans_0003789-3` (score=0.9941) - What causes Mantle cell lymphoma? Most lymphomas are not inherited, but rather an acquired disease in which the DNAwithin the cells has been damaged. Damage to the DNA occurs by a combination of different factors. Man...
3. `medquad_ans_0002483-3` (score=0.4833) - What causes gamma heavy chain disease? The causes or risk factors for gamma heavy chain disease are not known.
4. `medquad_ans_0000004_7-4` (score=0.0008) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on: - The type of lymphoma. - Where the tumor is in the body when the tumor ...
5. `medquad_ans_0000698-3` (score=0.0003) - The cause of mycosis fungoides is unknown. Most affected individuals have one or more chromosomal abnormalities, such as the loss or gain of genetic material. These abnormalities occur during a person's lifetime and a...

**Answer**:

Most lymphomas are not inherited but are acquired diseases in which the DNA within the cells has been damaged. Damage to the DNA occurs by a combination of different factors. Many mantle cell lymphomas are found to be associated with a chromosome translocation. Some causes of non-Hodgkin lymphomas (NHL) have been linked to viral infections including Epstein-Barr virus, HIV, and human herpesvirus 6. It has also been found that immunodeficiencies and environmental factors like hair dyes and pesticides may lead to NHLs [medquad_ans_0003789-3].

**Supported citations**: `medquad_ans_0003789-3`
**Unsupported citations**: _none_

### Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Scores**: context_relevance=0.750, groundedness=0.009, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000842-4` (score=0.1669) - Pseudoachondroplasia is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation fr...
2. `medquad_ans_0000689-4` (score=0.0162) - Multiple epiphyseal dysplasia can have different inheritance patterns. This condition can be inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the...
3. `medquad_ans_0002705-3` (score=0.0002) - How is Hailey-Hailey disease inherited? Hailey-Hailey disease is inherited in an autosomal dominant manner. This means that having only one mutated copy of the disease-causing gene in each cell is enough to cause sign...
4. `medquad_ans_0000464-2` (score=0.0001) - X-linked hypophosphatemic rickets is the most common form of rickets that runs in families. It affects about 1 in 20,000 newborns. Each of the other forms of hereditary hypophosphatemic rickets has been identified in ...
5. `medquad_ans_0001374-4` (score=0.0001) - How is cleidocranial dysplasia inherited? Cleidocranial dysplasia is inherited in an autosomal dominant manner, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases,...

**Answer**:

The provided context does not contain any information about the inheritance pattern of Legg-Calv-Perthes disease. Therefore, it is not possible to determine based on the given text whether Legg-Calv-Perthes disease is inherited.

Answer: Insufficient context [doc_id].

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Scores**: context_relevance=0.800, groundedness=0.979, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000192-1__dup1` (score=0.9699) - Metachromatic leukodystrophy (MLD) is one of a group of genetic disorders called the leukodystrophies, which are characterized by the toxic buildup of lipids (fatty materials such as oils and waxes) and other storage ...
2. `medquad_ans_0003965-1` (score=0.7330) - Metachromatic leukodystrophy is an inherited condition characterized by the accumulation of fats called sulfatides in cells, especially cells of the nervous system. This accumulation results in progressive destruction...
3. `medquad_ans_0003566-1` (score=0.1022) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to def...
4. `medquad_ans_0000181-1__dup2` (score=0.0983) - Leukodystrophy refers to progressive degeneration of the white matter of the brain due to imperfect growth or development of the myelin sheath, the fatty covering that acts as an insulator around nerve fiber. Myelin, ...
5. `medquad_ans_0000192-4__dup1` (score=0.0028) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge of the brain and nervous system and to use that knowledge to reduce the burden of neurological diseas...

**Answer**:

Metachromatic leukodystrophy (MLD) is an inherited condition characterized by the accumulation of fats called sulfatides in cells, especially in cells of the nervous system. This accumulation results in progressive destruction of the white matter of the brain, which consists of nerve fibers covered by myelin. Affected individuals experience progressive deterioration of intellectual functions and motor skills, such as the ability to walk, as well as loss of sensation in the extremities, incontinence, seizures, paralysis, and inability to speak [medquad_ans_0003965-1]. MLD is one of a group of genetic disorders called leukodystrophies, characterized by the toxic buildup of lipids and other storage materials in cells of the central nervous system and peripheral nerves, impairing the growth or development of the myelin sheath [medquad_ans_0000192-1__dup1]. It is caused by

**Supported citations**: `medquad_ans_0003965-1`, `medquad_ans_0000192-1__dup1`
**Unsupported citations**: _none_

### Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Scores**: context_relevance=0.571, groundedness=0.000, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000224-2` (score=0.0001) - Congenital fiber-type disproportion is thought to be a rare condition, although its prevalence is unknown.
2. `medquad_ans_0000290-2` (score=0.0001) - Distal arthrogryposis type 1 affects an estimated 1 in 10,000 people worldwide.
3. `medquad_ans_0000976-2` (score=0.0001) - This condition occurs in 1 in 20,000 to 50,000 newborns. Type I thanatophoric dysplasia is more common than type II.
4. `medquad_ans_0000132-2` (score=0.0001) - Boomerang dysplasia is a rare disorder; its exact prevalence is unknown. Approximately 10 affected individuals have been identified.
5. `medquad_ans_0000144-4` (score=0.0001) - Campomelic dysplasia is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in or near the SOX9 g...

**Answer**:

The provided context does not contain any information about the symptoms of Pseudodiastrophic dysplasia. Therefore, the symptoms cannot be determined from the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.985, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001664-1` (score=0.8285) - Cri du chat syndrome, also known as 5p- (5p minus) syndrome or cat cry syndrome, is a genetic condition that is caused by the deletion of genetic material on the small arm (the p arm) of chromosome 5. Infants with thi...
2. `medquad_ans_0000251-1` (score=0.8085) - Cri-du-chat (cat's cry) syndrome, also known as 5p- (5p minus) syndrome, is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that ...
3. `medquad_ans_0000117-4__dup2` (score=0.0071) - The NINDS conducts and supports a wide range of studies that explore the complex mechanisms of normal brain development. The knowledge gained from these fundamental studies provides the foundation for understanding ho...
4. `medquad_ans_0000979-2` (score=0.0011) - TAR syndrome is a rare disorder, affecting fewer than 1 in 100,000 newborns.
5. `medquad_ans_0000995-2` (score=0.0009) - This condition occurs in about 1 in 1,000 newborn girls. Five to 10 girls with triple X syndrome are born in the United States each day.

**Answer**:

Cri-du-chat syndrome, also known as 5p- (5p minus) syndrome or cat cry syndrome, is a genetic condition caused by the deletion of genetic material on the small arm (the p arm) of chromosome 5. Infants with this condition often have a high-pitched cry that sounds like that of a cat. The disorder is characterized by intellectual disability and delayed development, small head size (microcephaly), low birth weight, weak muscle tone (hypotonia) in infancy, and distinctive facial features such as widely set eyes (hypertelorism), low-set ears, a small jaw, and a rounded face. Some children with cri-du-chat syndrome are born with a heart defect. Most cases are not inherited [medquad_ans_0001664-1][medquad_ans_0000251-1].

**Supported citations**: `medquad_ans_0001664-1`, `medquad_ans_0000251-1`
**Unsupported citations**: _none_

### Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000719-2__dup1` (score=0.9992) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmen...
2. `medquad_ans_0003881-2` (score=0.1891) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speec...
3. `medquad_ans_0000230-3__dup1` (score=0.0001) - The prognosis for individuals with PVL depends upon the severity of the brain damage. Some children exhibit fairly mild symptoms, while others have significant deficits and disabilities.
4. `medquad_ans_0000718-1__dup1` (score=0.0001) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
5. `medquad_ans_0000390-2` (score=0.0001) - Frontonasal dysplasia is likely a rare condition; at least 100 cases have been reported in the scientific literature.

**Answer**:

The symptoms of bilateral frontoparietal polymicrogyria may include moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze, ataxia, strabismus, increased muscle tone, finger dysmetria, ankle clonus, Babinski sign, broad-based gait, cerebellar hypoplasia, esotropia, exotropia, hyperreflexia, hypertonia, hypoplasia of the brainstem, nystagmus, and truncal ataxia [medquad_ans_0000719-2__dup1]. The frequency of these symptoms varies and is based on studies, with some listed as common or rare depending on the number of patients examined [medquad_ans_0000719-2__dup1].

**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

### Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.974, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003414-3` (score=0.9965) - What causes Konigsmark Knox Hussels syndrome? Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid ar...
2. `medquad_ans_0000556-2` (score=0.0017) - KBG syndrome is a rare disorder that has been reported in around 60 individuals. For unknown reasons, males are affected more often than females. Doctors think the disorder is underdiagnosed because the signs and symp...
3. `medquad_ans_0000896-2` (score=0.0001) - Senior-Lken syndrome is a rare disorder, with an estimated prevalence of about 1 in 1 million people worldwide. Only a few families with the condition have been described in the medical literature.
4. `medquad_ans_0000553-2` (score=0.0001) - Kabuki syndrome occurs in approximately 1 in 32,000 newborns.
5. `medquad_ans_0000554-2` (score=0.0001) - Kallmann syndrome is estimated to affect 1 in 10,000 to 86,000 people and occurs more often in males than in females. Kallmann syndrome 1 is the most common form of the disorder.

**Answer**:

Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene, specifically a mutation that replaces the amino acid arginine with the amino acid histidine at position 445 in the OPA1 protein, written as Arg445His or R445H [medquad_ans_0003414-3]. It is unclear why this R445H mutation causes both hearing and vision loss in affected individuals.

**Supported citations**: `medquad_ans_0003414-3`
**Unsupported citations**: _none_

### Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Scores**: context_relevance=0.571, groundedness=0.000, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000249-2` (score=0.0012) - Craniofacial-deafness-hand syndrome is an extremely rare condition. Only a few cases have been reported in the scientific literature.
2. `medquad_ans_0000248-2` (score=0.0006) - Craniofacial microsomia has been estimated to occur in between 1 in 5,600 and 1 in 26,550 newborns. However, this range may be an underestimate because not all medical professionals agree on the criteria for diagnosis...
3. `medquad_ans_0000973-2` (score=0.0002) - Tetra-amelia syndrome has been reported in only a few families worldwide.
4. `medquad_ans_0000064-2` (score=0.0001) - Apert syndrome affects an estimated 1 in 65,000 to 88,000 newborns.
5. `medquad_ans_0000746-2` (score=0.0001) - OFCD syndrome is very rare; the incidence is estimated to be less than 1 in 1 million people.

**Answer**:

The provided context does not contain information about the symptoms of Orofaciodigital syndrome 1. Therefore, the question cannot be answered based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Scores**: context_relevance=1.000, groundedness=0.903, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000863-1__dup1` (score=0.9972) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For exa...
2. `medquad_ans_0000338-1__dup1` (score=0.9020) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the s...
3. `medquad_ans_0000067-17` (score=0.0329) - It is important to keep a copy of your medical records to share with any new doctors that you see. This information should contain the type of cancer you were diagnosed with, test results, and treatment details. It is...
4. `medquad_ans_0000869-1__dup1` (score=0.0296) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can brin...
5. `medquad_ans_0000694-1__dup1` (score=0.0276) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of me...

**Answer**:

Yes, there is information about surgery. Surgery can relieve or prevent pain, reduce symptoms, improve body function, or be performed to find a problem such as through a biopsy. Some surgeries, like heart surgery, can save a life. Laparoscopic surgery, which uses small cuts and a thin tube with a camera, is now an option for some procedures that once required large incisions. After surgery, there is a risk of complications including infection, excessive bleeding, reaction to anesthesia, or accidental injury. There is also almost always some pain following surgery [medquad_ans_0000863-1__dup1].

**Supported citations**: `medquad_ans_0000863-1__dup1`
**Unsupported citations**: _none_

### Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Scores**: context_relevance=0.875, groundedness=0.985, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0002935-1` (score=0.9831) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affe...
2. `medquad_ans_0000040-10` (score=0.0193) - Acute leukemia gets worse quickly. In chronic leukemia, symptoms develop gradually and are generally not as severe as in acute leukemia.
3. `medquad_ans_0000040-18` (score=0.0036) - Treatment depends on a number of factors, including the type of leukemia, the patient's age and general health, where leukemia cells have collected in the body, and whether the leukemia has been treated before. Certai...
4. `medquad_ans_0000040-23` (score=0.0027) - Clinical trials are research studies in which new treatments -- drugs, diagnostics, procedures, vaccines, and other therapies -- are tested in people to see if they are safe, effective, and better than the current sta...
5. `medquad_ans_0001068-2` (score=0.0018) - ZAP70-related SCID is a rare disorder. Only about 20 affected individuals have been identified. The prevalence of SCID from all genetic causes combined is approximately 1 in 50,000.

**Answer**:

Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affects the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affected people may later develop neurological problems and/or chronic lung infections. HTLV-2 is spread by blood transfusions, sexual contact, and sharing needles. It can also be spread from mother to child during birth or breast-feeding. There is no cure or treatment for HTLV-2, and it is considered a lifelong condition; however, most infected people remain asymptomatic (show no symptoms) throughout life [medquad_ans_0002935-1].

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
5. `medquad_ans_0000078-6` (score=0.0030) - Idiopathic pulmonary fibrosis (IPF) causes the same kind of scarring and symptoms as some other lung diseases. This makes it hard to diagnose. Seeking medical help as soon as you have symptoms is important. If possibl...

**Answer**:

Interstitial lung diseases (ILDs) are a large group of diseases that inflame or scar the lungs. The inflammation and scarring, known as pulmonary fibrosis, make it difficult to get enough oxygen. These diseases can be caused by breathing in dust or other particles in the air, such as coal dust (black lung disease), farm dust (farmer's lung), asbestos fibers (asbestosis), iron from mines or welding fumes (siderosis), or silica dust (silicosis). Other causes include autoimmune diseases, occupational exposures to molds, gases, or fumes, and some types have no known cause. Treatment depends on the type of exposure and the stage of the disease and may involve medicines, oxygen therapy, or a lung transplant in severe cases [medquad_ans_0000522-1__dup1]. Additionally, children's interstitial and diffuse lung disease (chILD) is a group of rare lung

**Supported citations**: `medquad_ans_0000522-1__dup1`
**Unsupported citations**: _none_

### Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.036, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000167-4` (score=0.9982) - As its name suggests, this condition is inherited in an autosomal recessive pattern. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. The parents of an individual with an auto...
2. `medquad_ans_0000662-4` (score=0.9959) - This condition is inherited in a mitochondrial pattern, which is also known as maternal inheritance. This pattern of inheritance applies to genes contained in mtDNA. Because egg cells, but not sperm cells, contribute ...
3. `medquad_ans_0000443-4` (score=0.0022) - Almost all cases of Guillain-Barr syndrome are sporadic, which means they occur in people with no history of the condition in their family. A few families with more than one affected family member have been described;...
4. `medquad_ans_0001084-3__dup1` (score=0.0003) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
5. `medquad_ans_0004872-3` (score=0.0001) - How is Pierson syndrome inherited? Pierson syndrome is inherited in an autosomal recessive manner. This means that to be affected, a person must have a mutation in both copies of the responsible gene in each cell. Aff...

**Answer**:

The provided context does not contain information about the inheritance pattern of Melnick-Needles syndrome. Therefore, based on the given context, it is not possible to determine whether Melnick-Needles syndrome is inherited.

[No supporting source available]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source available`

### Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Scores**: context_relevance=0.750, groundedness=0.532, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000908-1__dup2` (score=0.9954) - Your trachea, or windpipe, is one part of your airway system. Airways are pipes that carry oxygen-rich air to your lungs. They also carry carbon dioxide, a waste gas, out of your lungs. When you inhale, air travels fr...
2. `medquad_ans_0000012-7__dup1` (score=0.9806) - Chronic obstructive pulmonary disease, or COPD, is a progressive lung disease in which the airways of the lungs become damaged, making it harder to breathe. With COPD, airways become blocked, making it harder to get a...
3. `medquad_ans_0000133-1__dup2` (score=0.9724) - The bronchi are two tubes that branch off the trachea, or windpipe. The bronchi carry air to your lungs. The most common problem with the bronchi is bronchitis, an inflammation of the tubes. Bronchitis can be acute or...
4. `medquad_ans_0006138-1` (score=0.9703) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of ...
5. `medquad_ans_0006140-1` (score=0.0106) - Tracheobronchopathia osteoplastica (TO) is a rare condition of the large airways. It is characterized by the presence of multiple growths (nodules) made of bone and cartilage tissue, in the submucosa of the tracheobro...

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse. There are two forms: a congenital form (called primary TBM) that typically develops during infancy or early childhood, and an acquired form (called secondary TBM) usually seen in adults. Some affected people may initially have no signs or symptoms, but the condition is typically progressive and most people will eventually develop characteristic features such as shortness of breath, cough, sputum retention, and wheezing or stridor with breathing. Most cases of primary TBM are caused by genetic conditions that weaken the walls of the airway, while the secondary form often occurs due to trauma, chronic inflammation, and/or prolonged compression of the airways. Treatment is generally only required in those who have signs and symptoms [med

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Scores**: context_relevance=0.600, groundedness=0.993, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000880-1__dup2` (score=0.9909) - The temporomandibular joint (TMJ) connects your jaw to the side of your head. When it works well, it enables you to talk, chew, and yawn. For people with TMJ dysfunction, problems with the joint and muscles around it ...
2. `medquad_ans_0000531-1__dup2` (score=0.8110) - Your jaw is a set of bones that holds your teeth. It consists of two main parts. The upper part is the maxilla. It doesn't move. The moveable lower part is called the mandible. You move it when you talk or chew. The t...
3. `medquad_ans_0000114-3__dup2` (score=0.7308) - ESS is not a life-threatening condition. Most often, and particularly among those with primary ESS, the disorder does not cause health problems and does not affect life expectancy.
4. `medquad_ans_0006030-1` (score=0.4411) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people m...
5. `medquad_ans_0000874-2` (score=0.0061) - This condition affects an estimated 1 in 8,500 females.

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

### Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.929, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000554-4` (score=0.9615) - Kallmann syndrome 1 (caused by ANOS1 gene mutations) has an X-linked recessive pattern of inheritance. The ANOS1 gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only on...
2. `medquad_ans_0003339-3` (score=0.9550) - How is Kallmann syndrome inherited? Kallmann syndrome (KS) may be inherited in an X-linked recessive, autosomal dominant, or autosomal recessive manner depending on the gene(s) responsible. For example: KS due to muta...
3. `medquad_ans_0001478-4` (score=0.6491) - Is congenital anosmia inherited? Most cases of isolated congenital anosmia (not associated with additional symptoms) occur sporadically in people with no family history of the condition. Rarely, more than one family m...
4. `medquad_ans_0003474-3` (score=0.0006) - How is Larson syndrome inherited? Larson syndrome is inherited in an autosomal dominant manner. A condition is autosomal dominant when having one copy of the changed (mutated) gene in each cell is enough to cause sign...
5. `medquad_ans_0003339-1` (score=0.0001) - Kallmann syndrome (KS) is a condition characterized primarily by hypogonadotropic hypogonadism (HH) and absent or diminished sense of smell (anosmia or hyposmia, respectively). HH is present from birth and is due to d...

**Answer**:

Yes, Kallmann syndrome can be inherited in different patterns depending on the genetic cause. It may be inherited in an X-linked recessive, autosomal dominant, or autosomal recessive manner [medquad_ans_0003339-3]. For example, Kallmann syndrome 1, caused by mutations in the ANOS1 (KAL1) gene, is inherited in an X-linked recessive pattern [medquad_ans_0000554-4]. Other forms, such as those caused by mutations in FGFR1, PROKR2, PROK2, CHD7, or FGF8 genes, are predominantly inherited in an autosomal dominant manner [medquad_ans_0003339-3]. In some cases, mutations in PROKR2 and PROK2 can also follow an autosomal recessive pattern [medquad_ans_000333

**Supported citations**: `medquad_ans_0003339-3`, `medquad_ans_0000554-4`
**Unsupported citations**: _none_

### Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Scores**: context_relevance=1.000, groundedness=0.895, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005967-2` (score=0.9994) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how...
2. `medquad_ans_0000290-2` (score=0.0001) - Distal arthrogryposis type 1 affects an estimated 1 in 10,000 people worldwide.
3. `medquad_ans_0000810-1__dup1` (score=0.0001) - What are the signs and symptoms of Brachydactyly type A2? The Human Phenotype Ontology provides the following list of signs and symptoms for Brachydactyly type A2. If the information is available, the table below incl...
4. `medquad_ans_0005970-1` (score=0.0001) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how...
5. `medquad_ans_0005967-1` (score=0.0001) - Syndactyly type 3 (SD3) is a limb abnormality present at birth that is characterized by complete fusion of the 4th and 5th fingers on both hands. In most cases only the soft tissue is fused, but in some cases the bone...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and autosomal dominant inheritance [medquad_ans_0005967-2].

**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

### Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Scores**: context_relevance=0.800, groundedness=0.248, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000012-8__dup1` (score=0.9866) - COPD is a disease that slowly worsens over time, especially if you continue to smoke. If you have COPD, you are more likely to have lung infections, which can be fatal. If the lungs are severely damaged, the heart may...
2. `medquad_ans_0000012-7__dup1` (score=0.9730) - Chronic obstructive pulmonary disease, or COPD, is a progressive lung disease in which the airways of the lungs become damaged, making it harder to breathe. With COPD, airways become blocked, making it harder to get a...
3. `medquad_ans_0000232-1__dup2` (score=0.8652) - COPD (chronic obstructive pulmonary disease) makes it hard for you to breathe. The two main types are chronic bronchitis and emphysema. The main cause of COPD is long-term exposure to substances that irritate and dama...
4. `medquad_ans_0000133-1__dup2` (score=0.7908) - The bronchi are two tubes that branch off the trachea, or windpipe. The bronchi carry air to your lungs. The most common problem with the bronchi is bronchitis, an inflammation of the tubes. Bronchitis can be acute or...
5. `medquad_ans_0000012-1__dup5` (score=0.4235) - Chronic obstructive pulmonary disease, or COPD, is a progressive lung disease in which the airways of the lungs become damaged, making it hard to breathe. You may also have heard COPD called other names, like emphysem...

**Answer**:

Chronic bronchitis is one of the two main types of chronic obstructive pulmonary disease (COPD). It is characterized by inflammation of the bronchi, the tubes that carry air to the lungs, leading to a persistent cough that produces a lot of mucus. It is often caused by long-term exposure to irritants such as cigarette smoke, air pollution, chemical fumes, or dust. Chronic bronchitis is part of a progressive lung disease that makes it harder to breathe due to damaged airways [medquad_ans_00000232-1__dup2].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_00000232-1__dup2`

### Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Scores**: context_relevance=0.750, groundedness=0.953, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000937-5` (score=0.9966) - These resources address the diagnosis or management of SMARD1: - Genetic Testing Registry: Spinal muscular atrophy with respiratory distress 1 These resources from MedlinePlus offer information about the diagnosis and...
2. `medquad_ans_0000173-2__dup2` (score=0.9964) - There is no cure for SMA. Treatment consists of managing the symptoms and preventing complications.
3. `medquad_ans_0000208-2__dup1` (score=0.9945) - Treatments for the myopathies depend on the disease or condition and specific causes. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders. Treatment for other disor...
4. `medquad_ans_0000936-5` (score=0.9910) - These resources address the diagnosis or management of spinal muscular atrophy with progressive myoclonic epilepsy: - Genetic Testing Registry: Jankovic Rivera syndrome - Muscular Dystrophy Association: Spinal Muscula...
5. `medquad_ans_0000935-5` (score=0.9884) - These resources address the diagnosis or management of spinal muscular atrophy: - Gene Review: Gene Review: Spinal Muscular Atrophy - Gene Review: Gene Review: Spinal Muscular Atrophy, X-Linked Infantile - Genetic Tes...

**Answer**:

There is no cure for spinal muscular atrophy (SMA). Treatment consists of managing the symptoms and preventing complications [medquad_ans_0000173-2__dup2]. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders, and this may include drug therapy, physical therapy, bracing to support weakened muscles, and surgery [medquad_ans_0000208-2__dup1]. Resources such as MedlinePlus provide information on diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care for the diagnosis and management of SMA [medquad_ans_0000935-5].

**Supported citations**: `medquad_ans_0000173-2__dup2`, `medquad_ans_0000208-2__dup1`, `medquad_ans_0000935-5`
**Unsupported citations**: _none_

### Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Scores**: context_relevance=0.889, groundedness=0.821, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000267-5` (score=0.9944) - These resources address the diagnosis or management of cytogenetically normal acute myeloid leukemia: - Fred Hutchinson Cancer Research Center - National Cancer Institute: Acute Myeloid Leukemia Treatment - St. Jude C...
2. `medquad_ans_0000040-18` (score=0.9944) - Treatment depends on a number of factors, including the type of leukemia, the patient's age and general health, where leukemia cells have collected in the body, and whether the leukemia has been treated before. Certai...
3. `medquad_ans_0000001_6-5` (score=0.9902) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000040-23` (score=0.9826) - Clinical trials are research studies in which new treatments -- drugs, diagnostics, procedures, vaccines, and other therapies -- are tested in people to see if they are safe, effective, and better than the current sta...
5. `medquad_ans_0000001_1-7` (score=0.9738) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - C...

**Answer**:

The provided context does not contain information about treatments for Adult Acute Lymphoblastic Leukemia. While the context discusses treatments for cytogenetically normal acute myeloid leukemia and mentions clinical trials involving CAR T-cell therapy for childhood ALL that has relapsed, there is no specific information regarding Adult Acute Lymphoblastic Leukemia. [medquad_ans_0000267-5] [medquad_ans_0000040-18] [medquad_ans_0000001_6-5]

**Supported citations**: `medquad_ans_0000267-5`, `medquad_ans_0000040-18`, `medquad_ans_0000001_6-5`
**Unsupported citations**: _none_

### Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Scores**: context_relevance=1.000, groundedness=0.991, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002439-3` (score=0.9899) - What causes Friedreich ataxia? Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known...
2. `medquad_ans_0000033-3__dup1` (score=0.0037) - Average lifespan has been improving for years, for unknown reasons, and varies with the severity of the underlying mutations, ATM (ataxia-telangiectasia mutated) protein levels, and residual ATM kinase activity. Some ...
3. `medquad_ans_0000032-4__dup1` (score=0.0003) - The NINDS supports and conducts a broad range of basic and clinical research on cerebellar and spinocerebellar degeneration, including work aimed at finding the cause(s) of ataxias and ways to treat, cure, and, ultima...
4. `medquad_ans_0000133-3__dup1` (score=0.0002) - Generally, within 15 to 20 years after the appearance of the first symptoms, the person is confined to a wheelchair, and in later stages of the disease, individuals may become completely incapacitated. Friedreich's at...
5. `medquad_ans_0000383-5` (score=0.0001) - These resources address the diagnosis or management of FXTAS: - Fragile X Research Foundation of Canada: FXTAS - Gene Review: Gene Review: FMR1-Related Disorders - Genetic Testing Registry: Fragile X tremor/ataxia syn...

**Answer**:

Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known as a GAA trinucleotide repeat. Normally, this segment is repeated 5 to 33 times within the FXN gene. In people with Friedreich ataxia, the GAA segment is repeated 66 to more than 1,000 times. The abnormally long GAA trinucleotide repeat disrupts the production of frataxin, which severely reduces the amount of this protein in cells. Certain nerve and muscle cells cannot function properly with a shortage of frataxin, leading to the characteristic signs and symptoms of Friedreich ataxia [medquad_ans_0002439-3].

**Supported citations**: `medquad_ans_0002439-3`
**Unsupported citations**: _none_

### Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Scores**: context_relevance=0.833, groundedness=0.647, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000253-4` (score=0.9779) - Most cases of CCHD are sporadic, which means they occur in people with no history of the disorder in their family. However, close relatives (such as siblings) of people with CCHD may have an increased risk of being bo...
2. `medquad_ans_0000013-4__dup2` (score=0.9705) - Atrial fibrillation (AF) occurs if the heart's electrical signals don't travel through the heart in a normal way. Instead, they become very rapid and disorganized. Damage to the heart's electrical system causes AF. Th...
3. `medquad_ans_0000341-1` (score=0.8466) - Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm. This condition is characterized by uncoordinated electrical activity in the heart's upper chambers (the atria), which cau...
4. `medquad_ans_0002230-1` (score=0.8236) - Familial atrial fibrillation is an inherited heart condition that disrupts the heart's rhythm. It is characterized by erratic electrical activity in the heart's upper chambers (the atria), causing an irregular respons...
5. `medquad_ans_0000341-2` (score=0.7779) - Atrial fibrillation is the most common type of sustained abnormal heart rhythm (arrhythmia), affecting more than 3 million people in the United States. The risk of developing this irregular heart rhythm increases with...

**Answer**:

Familial atrial fibrillation is an inherited heart condition that disrupts the heart's rhythm, characterized by erratic electrical activity in the heart's upper chambers (the atria), which causes an irregular response in the heart's lower chambers (the ventricles) and results in a fast and irregular heartbeat (arrhythmia) [medquad_ans_0002230-1]. It may be caused by changes (mutations) in various genes, some of which have not been identified, and is most often inherited in an autosomal dominant manner, although autosomal recessive inheritance has been reported [medquad_ans_0002230-1]. Signs and symptoms may include dizziness, chest pain, palpitations, shortness of breath, or fainting, and affected individuals have an increased risk of stroke and sudden death [medquad_ans_000341-1]. Complications can occur at

**Supported citations**: `medquad_ans_0002230-1`
**Unsupported citations**: `medquad_ans_000341-1`

### Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Scores**: context_relevance=0.857, groundedness=0.964, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.8245) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the assoc...
2. `medquad_ans_0000922-1` (score=0.0092) - SOST-related sclerosing bone dysplasia is a disorder of bone development characterized by excessive bone formation (hyperostosis). As a result of hyperostosis, bones throughout the body are denser and wider than norma...
3. `medquad_ans_0000057-2__dup1` (score=0.0017) - Canavan disease causes progressive brain atrophy. There is no cure, nor is there a standard course of treatment. Treatment is symptomatic and supportive.
4. `medquad_ans_0000166-2` (score=0.0017) - CADASIL is likely a rare condition; however, its prevalence is unknown.
5. `medquad_ans_0000935-2` (score=0.0017) - Spinal muscular atrophy affects 1 in 6,000 to 1 in 10,000 people.

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia characterized by genetic changes (mutations) in the WTX gene, inherited in an X-linked dominant manner. It affects bone development and growth, leading to features present at birth such as skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected individuals may also have developmental delay, hearing loss, heart defects, and ophthalmoplegia (paralysis of the muscles surrounding the eyes). Treatment is based on the specific signs and symptoms present in each individual [medquad_ans_0004627-1].

**Supported citations**: `medquad_ans_0004627-1`
**Unsupported citations**: _none_

### Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004097-2` (score=0.9992) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus eryt...
2. `medquad_ans_0000103-3__dup1` (score=0.0090) - Most cases of dermatomyositis respond to therapy. The disease is usually more severe and resistant to therapy in individuals with cardiac or pulmonary problems.
3. `medquad_ans_0000163-3__dup1` (score=0.0045) - Most cases of dermatomyositis respond to therapy. The prognosis for polymyositis varies. Most individuals respond fairly well to therapy, but some people have a more severe disease that does not respond adequately to ...
4. `medquad_ans_0000966-2` (score=0.0002) - For unknown reasons, in industrialized Western countries SLE has become 10 times more common over the past 50 years. While estimates of its prevalence vary, SLE is believed to affect 14.6 to 68 per 100,000 people in t...
5. `medquad_ans_0000686-2` (score=0.0001) - Multiminicore disease is thought to be a rare disorder, although its incidence is unknown.

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. Common symptoms include pain in multiple joints, inflammation of joints (arthritis), muscle weakness, fevers, and fatigue. Other signs and symptoms may include accumulation of fluid in the tissue of the hands (edema), skin findings such as lupus-like rashes, reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia). Dysfunction of the esophagus (hypomotility), abnormalities in lung function leading to breathing difficulties or pulmonary hypertension, heart involvement (such as pericarditis, myocarditis, or aortic insufficiency), neurologic abnormalities (e

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Scores**: context_relevance=0.714, groundedness=0.743, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000098-2__dup1` (score=0.9580) - Drugs to specifically treat Alzheimers disease and some other progressive dementias are now available. Although these drugs do not halt the disease or reverse existing brain damage, they can improve symptoms and slow ...
2. `medquad_ans_0000003-14` (score=0.9261) - Currently, no medication can cure Alzheimers disease, but four medicines are approved to treat the symptoms of the disease. - Aricept (donezepil)for all stages of Alzheimers - Exelon (rivastigmine)for mild to moderate...
3. `medquad_ans_0000098-3__dup1` (score=0.4812) - There are many disorders that can cause dementia. Some, such as Alzheimers disease or Huntingtons disease, lead to a progressive loss of mental functions. But other types of dementia can be halted or reversed with app...
4. `medquad_ans_0000003-16` (score=0.4790) - When you learn that someone has Alzheimers disease, you may wonder when and how to tell your family and friends. You may be worried about how others will react to or treat the person. Others often sense that something...
5. `medquad_ans_0000004-21` (score=0.1315) - An early, accurate diagnosis of Alzheimer's disease helps people and their families plan for the future. It gives them time to discuss care options, find support, and make legal and financial arrangements while the pe...

**Answer**:

The provided context does not address treatments specifically for Alzheimer's caregiving. While it discusses medications for Alzheimer's disease and cognitive exercises that may benefit individuals in the early stages, it does not outline interventions or strategies aimed at supporting caregivers. Therefore, based on the given information, the treatments for Alzheimer's caregiving cannot be determined. [medquad_ans_0000098-2\_\_dup1][medquad_ans_0000003-14][medquad_ans_0000098-3__dup1]

**Supported citations**: `medquad_ans_0000098-2__dup1`, `medquad_ans_0000003-14`, `medquad_ans_0000098-3__dup1`
**Unsupported citations**: _none_

### Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Scores**: context_relevance=0.900, groundedness=0.897, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000511-3` (score=0.9972) - Idiopathic inflammatory myopathy is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Researchers have identi...
2. `medquad_ans_0000915-3` (score=0.9842) - Sjgren syndrome is thought to result from a combination of genetic and environmental factors; however, no associations between specific genetic changes and the development of Sjgren syndrome have been confirmed. Resea...
3. `medquad_ans_0000953-4` (score=0.0025) - SJS/TEN is not an inherited condition. However, the genetic changes that increase the risk of developing SJS/TEN can be passed from one generation to the next.
4. `medquad_ans_0000547-3` (score=0.0003) - Juvenile idiopathic arthritis is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Its signs and symptoms res...
5. `medquad_ans_0000163-4__dup1` (score=0.0001) - The National Institutes of Health (NIH), through the collaborative efforts of its National Institute of Neurological Disorders and Stroke (NINDS), National Institute of Arthritis and Musculoskeletal and Skin Diseases ...

**Answer**:

Genetic changes related to idiopathic inflammatory myopathy include variations in several genes, particularly within the human leukocyte antigen (HLA) complex. These HLA gene variations are associated with an increased risk of developing the disorder. The HLA complex plays a role in helping the immune system distinguish between the body's own proteins and foreign proteins. Specific variations in these genes appear to influence susceptibility to idiopathic inflammatory myopathy [medquad_ans_0000511-3]. Researchers are also studying other genes related to immune function to better understand their contribution to the condition's development [medquad_ans_0000511-3].

**Supported citations**: `medquad_ans_0000511-3`
**Unsupported citations**: _none_

### Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0005225-5` (score=0.9947) - Is genetic testing available for pyruvate kinase deficiency? Yes. GeneTests lists laboratories offering clinical genetic testing for this condition. Clinical genetic tests are ordered to help diagnose an affected pers...
2. `medquad_ans_0005223-5` (score=0.9899) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for...
3. `medquad_ans_0000855-2` (score=0.0001) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
4. `medquad_ans_0000856-2` (score=0.0001) - Pyruvate kinase deficiency is the most common inherited cause of nonspherocytic hemolytic anemia. More than 500 affected families have been identified, and studies suggest that the disorder may be underdiagnosed becau...
5. `medquad_ans_0000959-2` (score=0.0001) - Approximately 350 people with succinic semialdehyde dehydrogenase deficiency have been reported worldwide.

**Answer**:

The diagnosis of pyruvate dehydrogenase (PDH) deficiency may be considered in any individual with early-onset neurological disease, especially if it appears to be associated with structural abnormalities in the brain and unexplained lactic acidosis. When lactic acid (also called lactate) and pyruvate in the blood do not seem to be significantly high, an important clue to the diagnosis may be high concentrations of lactate and/or pyruvate in the cerebrospinal fluid (the fluid that surrounds the brain and spinal cord). Additionally, magnetic resonance spectroscopy (MRS) of the brain may show concentrations of lactate in the central nervous system. Analysis of serum and urine amino acids usually shows hyperalaninemia (high levels of the amino acid alanine). When lactic acidosis is present, other disorders involving pyruvate abnormalities are part of the differential diagnosis. However, in all of these conditions,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.938, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006562-2` (score=0.9991) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb a...
2. `medquad_ans_0001086-2` (score=0.0004) - At least 500 cases of 22q13.3 deletion syndrome are known.
3. `medquad_ans_0001084-2` (score=0.0003) - 22q11.2 deletion syndrome affects an estimated 1 in 4,000 people. However, the condition may actually be more common than this estimate because doctors and researchers suspect it is underdiagnosed due to its variable ...
4. `medquad_ans_0000531-2` (score=0.0001) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
5. `medquad_ans_0001086-3` (score=0.0001) - 22q13.3 deletion syndrome is caused by a deletion near the end of the long (q) arm of chromosome 22. The signs and symptoms of 22q13.3 deletion syndrome are probably related to the loss of multiple genes in this regio...

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

**Scores**: context_relevance=0.778, groundedness=0.773, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000757-2` (score=0.0005) - This condition affects an estimated 6 to 7 per 100,000 people worldwide. Types I and IV are the most common forms of osteogenesis imperfecta, affecting 4 to 5 per 100,000 people.
2. `medquad_ans_0000092-2` (score=0.0003) - The prevalence of autosomal dominant hypocalcemia is unknown. The condition is likely underdiagnosed because it often causes no signs or symptoms.
3. `medquad_ans_0000760-4` (score=0.0001) - Osteoporosis-pseudoglioma syndrome is inherited in an autosomal recessive pattern, which means both copies of the LRP5 gene in each cell have mutations. The parents of an individual with an autosomal recessive conditi...
4. `medquad_ans_0004638-1` (score=0.0001) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of ...
5. `medquad_ans_0004632-1` (score=0.0001) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of ...

**Answer**:

The provided context does not specify the symptoms of Osteopetrosis autosomal dominant type 2. [medquad_ans_0004638-1] and [medquad_ans_0004632-1] describe osteopetrosis in general, noting that it makes bones abnormally dense and prone to breakage, and that it can be inherited in autosomal dominant, autosomal recessive, or X-linked patterns. However, no specific details about symptoms of the autosomal dominant type 2 are given. Therefore, the symptoms of Osteopetrosis autosomal dominant type 2 cannot be determined from the provided context.

**Supported citations**: `medquad_ans_0004638-1`, `medquad_ans_0004632-1`
**Unsupported citations**: _none_

### Example 188 (`0003459-4`)

**Question**: What are the treatments for Lambert Eaton myasthenic syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.992, answer_relevance=0.489

**Top docs**:

1. `medquad_ans_0000208-2__dup1` (score=0.9938) - Treatments for the myopathies depend on the disease or condition and specific causes. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders. Treatment for other disor...
2. `medquad_ans_0000206-3__dup1` (score=0.9911) - With treatment, most individuals with myasthenia can significantly improve their muscle weakness. Some case of myasthenia gravis may go into remission temporarily, and muscle weakness may disappear so that medications...
3. `medquad_ans_0003459-4` (score=0.9687) - How might Lambert-Eaton myasthenic syndrome be treated? Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, ...
4. `medquad_ans_0000175-4__dup1` (score=0.0374) - The NINDS supports research on neuromuscular disorders such as LEMS with the ultimate goal of finding ways to treat, prevent, and cure them.
5. `medquad_ans_0000206-4__dup1` (score=0.0330) - Scientists are evaluating new and improving current treatments for myasthenia gravis. Different drugs are being tested, either alone or in combination with existing drug therapies, to see if they are effective in trea...

**Answer**:

Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, plasmapheresis (where blood plasma is removed and replaced with fluid, protein, or donated plasma), IV immunoglobulins, steroids (e.g., prednisone), azathioprine or cyclosporine, and/or 3,4-diaminopyridine. While there has been some evidence that either 3,4-diaminopyridine or IV immunoglobulin can improve muscle strength and nerve to muscle cell communication, the degree of benefit (i.e., how much symptoms are improved) still needs to be determined [medquad_ans_0003459-4].

**Supported citations**: `medquad_ans_0003459-4`
**Unsupported citations**: _none_

### Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000445-2__dup1` (score=0.9993) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes...
2. `medquad_ans_0000067-2` (score=0.0002) - Argininosuccinic aciduria occurs in approximately 1 in 70,000 newborns.
3. `medquad_ans_0000065-3` (score=0.0001) - Mutations in the ARG1 gene cause arginase deficiency. Arginase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occurs in liver cells. This ...
4. `medquad_ans_0000065-1` (score=0.0001) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down ...
5. `medquad_ans_0000445-1__dup1` (score=0.0001) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...

**Answer**:

The symptoms of Arginase deficiency include aminoaciduria, behavioral abnormality, cognitive impairment, neurological speech impairment, EEG abnormality, hemiplegia/hemiparesis, hyperammonemia, seizures, anorexia, hyperactivity, intellectual disability, irritability, oroticaciduria, postnatal growth retardation, progressive spastic quadriplegia, and vomiting [medquad_ans_0000445-2__dup1]. These symptoms are based on the Human Phenotype Ontology and represent approximate frequencies observed in patients with the condition [medquad_ans_0000445-2__dup1].

**Supported citations**: `medquad_ans_0000445-2__dup1`
**Unsupported citations**: _none_

### Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Scores**: context_relevance=0.778, groundedness=0.957, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000465-5` (score=0.9925) - These resources address the diagnosis or management of HLRCC: - Gene Review: Gene Review: Hereditary Leiomyomatosis and Renal Cell Cancer - Genetic Testing Registry: Hereditary leiomyomatosis and renal cell cancer - M...
2. `medquad_ans_0000022-2__dup1` (score=0.9864) - Treatment for VHL varies according to the location and size of the tumor. In general, the objective of treatment is to treat the tumors before they grow to a size large enough to cause permanent problems by putting pr...
3. `medquad_ans_0006376-6` (score=0.9676) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but ar...
4. `medquad_ans_0000022-3__dup1` (score=0.0160) - The prognosis for individuals with VHL depends on then number, location, and complications of the tumors. Untreated, VHL may result in blindness and/or permanent brain damage. With early detection and treatment the pr...
5. `medquad_ans_0000022-4__dup1` (score=0.0092) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system, and to use that knowledge to reduce the burden of neurological di...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat tumors when they cause symptoms but are still small enough to avoid permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All individuals with VHL disease should be carefully followed by a physician or medical team familiar with the disorder [medquad_ans_0006376-6]. Additionally, treatment aims to remove tumors before they grow large enough to cause pressure on the brain or spinal cord, which could impair vision, cause deafness, or block cerebrospinal fluid flow [medquad_ans_0000022-2__dup1].

**Supported citations**: `medquad_ans_0006376-6`, `medquad_ans_0000022-2__dup1`
**Unsupported citations**: _none_

### Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Scores**: context_relevance=0.800, groundedness=0.701, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004456-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
2. `medquad_ans_0004451-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0004450-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004454-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004453-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

The provided context does not mention "Noonan syndrome 3" or any specific subtype of Noonan syndrome beyond general references to subtypes based on the responsible gene. Therefore, based on the available information, Noonan syndrome 3 cannot be identified or described.

Answer: Insufficient context to determine what Noonan syndrome 3 is. [medquad_ans_0004456-1]

**Supported citations**: `medquad_ans_0004456-1`
**Unsupported citations**: _none_

### Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Scores**: context_relevance=1.000, groundedness=0.956, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005865-2` (score=0.9991) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may...
2. `medquad_ans_0000562-2` (score=0.0006) - Klippel-Feil syndrome is estimated to occur in 1 in 40,000 to 42,000 newborns worldwide. Females seem to be affected slightly more often than males.
3. `medquad_ans_0000944-2` (score=0.0003) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.
4. `medquad_ans_0000170-2__dup1` (score=0.0003) - Treatment for Klippel-Feil Syndrome is symptomatic and may include surgery to relieve cervical or craniocervical instability and constriction of the spinal cord, and to correct scoliosis. Physical therapy may also be ...
5. `medquad_ans_0000565-2` (score=0.0003) - Knobloch syndrome is a rare condition. However, the exact prevalence of the condition is unknown.

**Answer**:

The symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additional features that may be present include bone, cartilage, or fibrous tissue between the shoulder blade and the spinal bones (vertebrae), cleft palate (in 7.5% of cases), scoliosis, cervical segmentation defects, hemivertebrae, rib segmentation abnormalities, neck muscle hypoplasia, shoulder muscle hypoplasia, spina bifida occulta, limb length discrepancy, abnormalities of the collarbone, abnormalities of the chest organs (such as the liver or heart being displaced), and Klippel-Feil syndrome [medquad_ans_0005865-2]. Approximately 90% of

**Supported citations**: `medquad_ans_0005865-2`
**Unsupported citations**: _none_

### Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.957, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0000882-3` (score=0.9988) - Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene. This gene provides instructions for making a protein whose function is not well understood, although it is involved in chemical signaling p...
2. `medquad_ans_0000373-3` (score=0.0161) - Researchers have identified changes in five regions of the X chromosome that are linked to FG syndrome in affected families. Mutations in a gene called MED12, which is located in one of these regions, appear to be the...
3. `medquad_ans_0000392-3` (score=0.0097) - The cause of Fryns syndrome is unknown. The disorder is thought to be genetic because it tends to run in families and has features similar to those of other genetic disorders. Duplications and deletions in several chr...
4. `medquad_ans_0000174-3` (score=0.0009) - Mutations in the CHD7 gene cause more than half of all cases of CHARGE syndrome. The CHD7 gene provides instructions for making a protein that most likely regulates gene activity (expression) by a process known as chr...
5. `medquad_ans_0000803-3` (score=0.0003) - The cause of Poland syndrome is unknown. Researchers have suggested that it may result from a disruption of blood flow during development before birth. This disruption is thought to occur at about the sixth week of em...

**Answer**:

Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene, which prevents cells from making any functional ROR2 protein and disrupts development before birth [medquad_ans_0000882-3]. Autosomal dominant Robinow syndrome can be caused by mutations in the WNT5A or DVL1 gene, with the osteosclerotic form resulting from DVL1 gene mutations [medquad_ans_0000882-3]. The proteins from these genes are part of the same chemical signaling pathways as the ROR2 protein, and mutations alter the production or function of these proteins, impairing signaling essential for early development [medquad_ans_0000882-3]. In some cases, no mutation in the ROR2, WNT5A, or DVL1 gene is identified, and the cause remains unknown [medquad_ans_00

**Supported citations**: `medquad_ans_0000882-3`
**Unsupported citations**: _none_

### Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Scores**: context_relevance=0.818, groundedness=0.054, answer_relevance=0.818

**Top docs**:

1. `medquad_ans_0000058-11__dup1` (score=0.9894) - Kidney failure, also called end-stage kidney disease or ESRD, means your kidneys no longer work well enough to do their job. You will need treatment to replace the work your kidneys have stopped doing.
2. `medquad_ans_0000199-1__dup2` (score=0.9211) - You have two kidneys, each about the size of your fist. Their main job is to filter wastes and excess water out of your blood to make urine. They also keep the body's chemical balance, help control blood pressure, and...
3. `medquad_ans_0000539-1__dup1` (score=0.8510) - Healthy kidneys clean your blood by removing excess fluid, minerals, and wastes. They also make hormones that keep your bones strong and your blood healthy. But if the kidneys are damaged, they don't work properly. Ha...
4. `medquad_ans_0000038-27` (score=0.3250) - When your kidneys fail, they are no longer able to filter blood and remove waste from your body well enough to maintain health. Kidney failure causes harmful waste and excess fluid to build up in your body. Your blood...
5. `medquad_ans_0000179-10` (score=0.1681) - If you have kidney failure, learn about the treatments and think about which one best fits you. Talk with people who are on hemodialysis or peritoneal dialysis. Ask what is good and bad about each treatment. If you ma...

**Answer**:

The provided context does not contain information about "Kidney Failure: Eat Right to Feel Right on Hemodialysis." Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Scores**: context_relevance=0.667, groundedness=0.945, answer_relevance=0.489

**Top docs**:

1. `medquad_ans_0000681-2` (score=0.9995) - The exact prevalence of MPS IV is unknown, although it is estimated to occur in 1 in 200,000 to 300,000 individuals.
2. `medquad_ans_0000677-2` (score=0.9994) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
3. `medquad_ans_0000678-2` (score=0.9945) - Severe MPS I occurs in approximately 1 in 100,000 newborns. Attenuated MPS I is less common and occurs in about 1 in 500,000 newborns.
4. `medquad_ans_0000424-2` (score=0.9915) - GSD IV is estimated to occur in 1 in 600,000 to 800,000 individuals worldwide. Type IV accounts for roughly 3 percent of all cases of glycogen storage disease.
5. `medquad_ans_0000429-2` (score=0.9865) - GM1 gangliosidosis is estimated to occur in 1 in 100,000 to 200,000 newborns. Type I is reported more frequently than the other forms of this condition. Most individuals with type III are of Japanese descent.

**Answer**:

Mucolipidosis type IV is estimated to occur in 1 in 40,000 people [medquad_ans_0000677-2].

**Supported citations**: `medquad_ans_0000677-2`
**Unsupported citations**: _none_

### Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Scores**: context_relevance=1.000, groundedness=0.994, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.9368) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chi...
2. `medquad_ans_0000033-9` (score=0.6788) - A heart attack occurs when the supply of blood and oxygen to an area of the heart muscle is blocked, usually by a blood clot in a coronary artery. If the blockage is not treated within a few hours, the heart muscle wi...
3. `medquad_ans_0000193-1__dup4` (score=0.0106) - Cirrhosis\* is scarring of the liver. Scar tissue forms because of injury or long-term disease. Scar tissue replaces healthy liver tissue and blocks the normal flow of blood through the liver. A healthy liver - makes p...
4. `medquad_ans_0000193-6` (score=0.0084) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need t...
5. `medquad_ans_0000230-1__dup2` (score=0.0069) - Connective tissue is the material inside your body that supports many of its parts. It is the "cellular glue" that gives your tissues their shape and helps keep them strong. It also helps some of your tissues do their...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams. [medquad_ans_0000795-1__dup1]

**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

### Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Scores**: context_relevance=0.667, groundedness=0.910, answer_relevance=0.489

**Top docs**:

1. `medquad_ans_0001045-3` (score=0.9992) - Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. The ABCD1 gene provides instructions for producing the adrenoleukodystrophy protein (ALDP), which is involved in transporting certain fat molecules call...
2. `medquad_ans_0000008-4__dup1` (score=0.0011) - The NINDS supports research on genetic disorders such as ALD. The aim of this research is to find ways to prevent, treat, and cure these disorders. Studies are currently underway to identify new biomarkers of disease ...
3. `medquad_ans_0000181-4__dup1` (score=0.0004) - The NINDS supports research on genetic disorders, including the leukodystrophies. The goals of this research are to increase scientific understanding of these disorders, and to find ways to prevent, treat, and, ultima...
4. `medquad_ans_0000227-4__dup1` (score=0.0003) - NINDS supports research on gene-linked disorders, including the leukodystrophies. The goals of this research are to increase scientific understanding of these disorders and to find ways to prevent, treat, and ultimate...
5. `medquad_ans_0001054-4` (score=0.0001) - This condition is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each cell. In males...

**Answer**:

Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. These mutations result in a deficiency of the adrenoleukodystrophy protein (ALDP), which is responsible for transporting very long-chain fatty acids (VLCFAs) into peroxisomes. The lack of this protein disrupts the transport and breakdown of VLCFAs, leading to their abnormal accumulation in the body [medquad_ans_0001045-3]. This condition is inherited in an X-linked pattern, meaning the mutated gene is located on the X chromosome, and in males, one altered copy of the gene is sufficient to cause the disorder [medquad_ans_0001054-4].

**Supported citations**: `medquad_ans_0001045-3`, `medquad_ans_0001054-4`
**Unsupported citations**: _none_

### Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Scores**: context_relevance=0.625, groundedness=0.972, answer_relevance=0.425

**Top docs**:

1. `medquad_ans_0000003-2` (score=0.9992) - Abetalipoproteinemia is a rare disorder with approximately 100 cases described worldwide.
2. `medquad_ans_0000826-2` (score=0.9987) - These disorders are very rare. Although the exact prevalence of prion disease is unknown, studies suggest that this group of conditions affects about one person per million worldwide each year. Approximately 350 new c...
3. `medquad_ans_0000988-2` (score=0.9981) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of Europe...
4. `medquad_ans_0000204-3__dup1` (score=0.9961) - Most individuals with multiple system atrophy with orthostatic hypotension die within 7 to 10 years after the onset of symptoms. A problem with the respiratory system is the most common cause of death.
5. `medquad_ans_0000457-2` (score=0.9945) - The prevalence of hereditary cerebral amyloid angiopathy is unknown. The Dutch type is the most common, with over 200 affected individuals reported in the scientific literature.

**Answer**:

The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence is thought to be one in 538 people. Among Americans of European descent, it is estimated to affect one in 100,000 people. The cardiac form of transthyretin amyloidosis is more common among people with African ancestry, affecting between 3 percent and 3.9 percent of African Americans and approximately 5 percent of people in some areas of West Africa [medquad_ans_0000988-2].

**Supported citations**: `medquad_ans_0000988-2`
**Unsupported citations**: _none_

### Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Scores**: context_relevance=1.000, groundedness=0.778, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.9994) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the informatio...
2. `medquad_ans_0000874-2` (score=0.0241) - This condition affects an estimated 1 in 8,500 females.
3. `medquad_ans_0000295-2` (score=0.0067) - DOLK-CDG is likely a rare condition; at least 18 cases have been reported in the scientific literature.
4. `medquad_ans_0000094-2` (score=0.0035) - This condition appears to be uncommon, although its prevalence is unknown.
5. `medquad_ans_0000277-2` (score=0.0005) - Dentinogenesis imperfecta affects an estimated 1 in 6,000 to 8,000 people.

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include information compiled by the Human Phenotype Ontology (HPO), which provides a list of features observed in individuals with this condition. The frequency of these symptoms is often estimated as a percentage or fraction based on specific studies, with some symptoms being reported in a significant proportion of patients. However, the provided context does not specify the exact symptoms or their frequencies beyond stating that the HPO has collected such data [medquad_ans_0001780-1]. Therefore, the specific symptoms are not detailed in the available information.

**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

### Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Scores**: context_relevance=0.500, groundedness=0.711, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000782-2` (score=0.9993) - Periventricular heterotopia is a rare condition. Its incidence is unknown.
2. `medquad_ans_0000736-2` (score=0.0073) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
3. `medquad_ans_0000230-3__dup1` (score=0.0005) - The prognosis for individuals with PVL depends upon the severity of the brain damage. Some children exhibit fairly mild symptoms, while others have significant deficits and disabilities.
4. `medquad_ans_0000807-2` (score=0.0002) - The prevalence of isolated polymicrogyria is unknown. Researchers believe that it may be relatively common overall, although the individual forms of the disorder (such as bilateral generalized polymicrogyria) are prob...
5. `medquad_ans_0004818-2` (score=0.0001) - What are the signs and symptoms of periventricular nodular heterotopia? The condition is first noticed when seizures appear, often during the teenage years. The nodules around the ventricles are then typically discove...

**Answer**:

The incidence of periventricular heterotopia is unknown [medquad_ans_0000782-2]. Therefore, the number of people affected by periventricular heterotopia cannot be determined from the provided context.

**Supported citations**: `medquad_ans_0000782-2`
**Unsupported citations**: _none_

### Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Scores**: context_relevance=0.545, groundedness=0.000, answer_relevance=0.691

**Top docs**:

1. `medquad_ans_0000214-4__dup1` (score=0.9987) - The NINDS conducts and supports a wide range of studies that explore the complex systems of brain development. These studies include the identification of the mechanism of action of the known causes of NMD as well as ...
2. `medquad_ans_0000113-4__dup1` (score=0.9987) - The NINDS conducts and supports an extensive research program on seizures and seizure-related disorders. Much of this research is aimed at increasing scientific understanding of these disorders and finding ways to pre...
3. `medquad_ans_0000021-4__dup1` (score=0.9983) - The NINDS supports and conducts research on neurogenetic disorders such as Angelman syndrome, to develop techniques to diagnose, treat, prevent, and ultimately cure them.
4. `medquad_ans_0000169-4` (score=0.9983) - NINDS supports a broad range of clinical and basic research on diseases causing sleep disorders in an effort to clarify the mechanisms of these conditions and to develop better treatments for them.
5. `medquad_ans_0000178-4__dup2` (score=0.9983) - The NINDS supports and encourages a broad range of basic and clinical research on neurogenetic disorders such as Leigh's disease. The goal of this research is to understand what causes these disorders and then to appl...

**Answer**:

The provided context does not mention Landau-Kleffner Syndrome or any research specifically directed at it. Therefore, based on the available information, no research or clinical trials for Landau-Kleffner Syndrome are described. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.303, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000205-1__dup2` (score=0.9314) - The muscular dystrophies (MD) are a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or ...
2. `medquad_ans_0000631-1__dup1` (score=0.8258) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...
3. `medquad_ans_0000201-4__dup1` (score=0.2482) - The NINDS supports a broad range of research on neuromuscular disorders with the goal of finding ways to prevent, treat, and, ultimately, cure them.
4. `medquad_ans_0003601-1` (score=0.0265) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
5. `medquad_ans_0001935-1` (score=0.0195) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystroph...

**Answer**:

Muscular dystrophy (MD) is a group of more than 30 inherited diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. The disorders vary in terms of the distribution and extent of muscle weakness, age of onset, rate of progression, and pattern of inheritance. Some forms of MD appear in infancy or childhood, while others may not appear until middle age or later. All forms of MD worsen over time as muscle strength decreases, and most people eventually lose the ability to walk. There is no cure for muscular dystrophy; treatments aim to manage symptoms and prevent complications, including physical and speech therapy, orthopedic devices, surgery, and medications [medquad_ans_0000205-1__dup2, medquad_ans_0000631-1__dup1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000205-1__dup2, medquad_ans_0000631-1__dup1`

### Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.342, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.8062) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that t...
2. `medquad_ans_0005020-1` (score=0.0002) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the tabl...
3. `medquad_ans_0000632-5` (score=0.0001) - These resources address the diagnosis or management of McKusick-Kaufman syndrome: - Gene Review: Gene Review: McKusick-Kaufman Syndrome - Genetic Testing Registry: McKusick Kaufman syndrome - MedlinePlus Encyclopedia:...
4. `medquad_ans_0000734-2` (score=0.0001) - Aplasia cutis congenita affects approximately 1 in 10,000 newborns. The incidence of the nonsyndromic form is unknown.
5. `medquad_ans_0003728-1` (score=0.0001) - Macrodactyly of the hand is a rare condition in which a person's fingers are abnormally large due to the overgrowth of the underlying bone and soft tissue. This condition is congenital, meaning that babies are born wi...

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding with no other physical anomalies or intellectual impairment, or it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits, with surgical management depending greatly on the complexity of the deformity [1] [2].

**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

### Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Scores**: context_relevance=0.500, groundedness=0.006, answer_relevance=0.633

**Top docs**:

1. `medquad_ans_0000248-2` (score=0.0120) - Craniofacial microsomia has been estimated to occur in between 1 in 5,600 and 1 in 26,550 newborns. However, this range may be an underestimate because not all medical professionals agree on the criteria for diagnosis...
2. `medquad_ans_0001036-2` (score=0.0095) - Williams syndrome affects an estimated 1 in 7,500 to 10,000 people.
3. `medquad_ans_0000815-2` (score=0.0051) - Prader-Willi syndrome affects an estimated 1 in 10,000 to 30,000 people worldwide.
4. `medquad_ans_0000560-2` (score=0.0016) - The prevalence of Kleefstra syndrome is unknown. Only recently has testing become available to distinguish it from other disorders with similar features.
5. `medquad_ans_0004934-5` (score=0.0010) - When is Poland syndrome typically first diagnosed? The severity of Poland syndrome differs from person to person. As a result it is not often diagnosed or reported. Sometimes, a person does not realize they have the c...

**Answer**:

The provided context does not contain any information about the symptoms of Geniospasm. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=0.800, groundedness=0.983, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000512-3` (score=0.9928) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of devel...
2. `medquad_ans_0000262-3` (score=0.0022) - Mutations in the CFTR gene cause cystic fibrosis. The CFTR gene provides instructions for making a channel that transports negatively charged particles called chloride ions into and out of cells. Chloride is a compone...
3. `medquad_ans_0000849-3` (score=0.0011) - Mutations in the BMPR2 gene are the most common genetic cause of pulmonary arterial hypertension. This gene plays a role in regulating the number of cells in certain tissues. Researchers suggest that a mutation in thi...
4. `medquad_ans_0000262-2` (score=0.0007) - Cystic fibrosis is a common genetic disease within the white population in the United States. The disease occurs in 1 in 2,500 to 3,500 white newborns. Cystic fibrosis is less common in other ethnic groups, affecting ...
5. `medquad_ans_0000037-3` (score=0.0006) - The cause of allergic asthma is complex. It is likely that a combination of multiple genetic and environmental factors contribute to development of the condition. Doctors believe genes are involved because having a fa...

**Answer**:

Genetic changes related to idiopathic pulmonary fibrosis include mutations in the TERC and TERT genes. These mutations have been found in about 15 percent of all cases of familial pulmonary fibrosis and a smaller percentage of cases of sporadic idiopathic pulmonary fibrosis. The TERC and TERT genes provide instructions for making components of an enzyme called telomerase, which maintains telomeres at the ends of chromosomes. It is not well understood how defects in telomerase are associated with the lung damage characteristic of idiopathic pulmonary fibrosis [medquad_ans_0000512-3]. Most of the genetic changes identified account for only a small proportion of cases [medquad_ans_0000512-3].

**Supported citations**: `medquad_ans_0000512-3`
**Unsupported citations**: _none_

### Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.981, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000883-4` (score=0.9980) - This condition is typically inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutatio...
2. `medquad_ans_0000903-4` (score=0.9967) - SHORT syndrome has an autosomal dominant pattern of inheritance, which means one copy of the altered PIK3R1 gene in each cell is sufficient to cause the disorder. In most cases, the condition results from a new mutati...
3. `medquad_ans_0000813-4` (score=0.9934) - Potocki-Shaffer syndrome follows an autosomal dominant inheritance pattern, which means a deletion of genetic material from one copy of chromosome 11 is sufficient to cause the disorder. In some cases, an affected per...
4. `medquad_ans_0000392-4` (score=0.3359) - Fryns syndrome appears to be inherited in an autosomal recessive pattern, which means both copies of a gene in each cell have mutations. However, no associated gene has been identified. The parents of an individual wi...
5. `medquad_ans_0000912-4` (score=0.0266) - Silver syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In these cases, the affected person inherits the mutation from ...

**Answer**:

Yes, Potocki-Shaffer syndrome is inherited in an autosomal dominant pattern. This means a deletion of genetic material from one copy of chromosome 11 is sufficient to cause the disorder. In some cases, an affected person inherits the chromosome with a deleted segment from an affected parent. More commonly, the condition results from a deletion that occurs during the formation of reproductive cells (eggs and sperm) in a parent or in early fetal development, which occurs in people with no history of the disorder in their family [medquad_ans_0000813-4].

**Supported citations**: `medquad_ans_0000813-4`
**Unsupported citations**: _none_

### Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Scores**: context_relevance=0.714, groundedness=0.084, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000190-1__dup2` (score=0.9980) - Summary : Kids, this page is for you. Learn about everything from how the body works to what happens when you go to the hospital. There are quizzes, games and lots of cool web sites for you to explore. Have fun!
2. `medquad_ans_0000076-1__dup3` (score=0.9978) - Summary : There are many new responsibilities when you have a baby. One of them is to make sure they get the checkups that they need. Well-baby exams are important in making sure that your baby is growing and developi...
3. `medquad_ans_0000897-1__dup1` (score=0.9972) - Summary : Most young children get sick. It is hard for parents to know what is serious. You can learn what the common warning signs are. In the end, trust your intuition. If you are worried about your toddler, call yo...
4. `medquad_ans_0000737-1__dup1` (score=0.9966) - Summary : So you're going to have a baby! Whether you are pregnant or are planning to get pregnant, you will want to give your baby a healthy start. You need to have regular visits with your healthcare provider. These...
5. `medquad_ans_0000506-1__dup1` (score=0.9941) - Summary : Going home with a new baby is exciting, but it can be scary, too. Newborns have many needs, like frequent feedings and diaper changes. Babies can have health issues that are different from older children and...

**Answer**:

No, there is no information about child care in the provided context. [medquad_ans_0000190-1__dup2, medquad_ans_0000076-1__dup3, medquad_ans_0000897-1__dup1, medquad_ans_0000737-1__dup1, medquad_ans_0000506-1__dup1] discuss topics such as well-child visits, immunizations, newborn needs, and prenatal care, but do not address child care.

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000190-1__dup2, medquad_ans_0000076-1__dup3, medquad_ans_0000897-1__dup1, medquad_ans_0000737-1__dup1, medquad_ans_0000506-1__dup1`

### Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Scores**: context_relevance=0.700, groundedness=0.908, answer_relevance=0.780

**Top docs**:

1. `medquad_ans_0000046-2` (score=0.9995) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
2. `medquad_ans_0000230-3__dup1` (score=0.0012) - The prognosis for individuals with PVL depends upon the severity of the brain damage. Some children exhibit fairly mild symptoms, while others have significant deficits and disabilities.
3. `medquad_ans_0000016-2__dup1` (score=0.0001) - Drug therapy including verapamil may help to reduce the severity and duration of attacks of paralysis associated with the more serious form of alternating hemiplegia
4. `medquad_ans_0000016-3__dup1` (score=0.0001) - Children with the benign form of alternating hemiplegia have a good prognosis. Those who experience the more severe form have a poor prognosis because intellectual and mental capacities do not respond to drug therapy,...
5. `medquad_ans_0000046-4` (score=0.0001) - Alternating hemiplegia of childhood is considered an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of alternating hemiplegia of chi...

**Answer**:

Approximately 1 in 1 million people are affected by alternating hemiplegia of childhood [medquad_ans_0000046-2].

**Supported citations**: `medquad_ans_0000046-2`
**Unsupported citations**: _none_

### Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Scores**: context_relevance=0.833, groundedness=0.942, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000159-3__dup2` (score=0.9977) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time...
2. `medquad_ans_0000159-6` (score=0.0062) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
3. `medquad_ans_0000182-6` (score=0.0043) - The most common symptoms of autoimmune hepatitis are - fatigue - joint pain - nausea - loss of appetite - pain or discomfort over the liver - skin rashes - dark yellow urine - light-colored stools - jaundice, or yello...
4. `medquad_ans_0000186-2__dup1` (score=0.0014) - There is no cure for lupus. Treatment is symptomatic. With a combination of medication, rest, exercise, proper nutrition, and stress management, most individuals with lupus can often achieve remission or reduce their ...
5. `medquad_ans_0000966-5` (score=0.0004) - These resources address the diagnosis or management of systemic lupus erythematosus: - MedlinePlus Encyclopedia: Antinuclear Antibody Panel These resources from MedlinePlus offer information about the diagnosis and ma...

**Answer**:

The symptoms of lupus may include high blood pressure, foamy urine, and edema (swelling), usually in the legs, feet, or ankles and less often in the hands or face. Other symptoms include joint pain or swelling, muscle pain, fever with no known cause, and red rashes, often on the face, which are also called butterfly rashes due to their shape [medquad_ans_0000159-3__dup2]. Additionally, lupus can lead to kidney problems that develop at the same time or shortly after lupus symptoms appear [medquad_ans_0000159-3__dup2]. Treatment is symptomatic and may involve medications such as aspirin, nonsteroidal anti-inflammatory drugs, antimalarials, corticosteroids, and immunosuppressive drugs, with most individuals being able to achieve remission or reduce symptom levels with proper management [medquad

**Supported citations**: `medquad_ans_0000159-3__dup2`
**Unsupported citations**: _none_

### Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.534, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000194-1` (score=0.2765) - Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features of this disorder include a failure to gain weight and grow at the expected rate (failure to thrive), a...
2. `medquad_ans_0000033-3__dup1` (score=0.2757) - Average lifespan has been improving for years, for unknown reasons, and varies with the severity of the underlying mutations, ATM (ataxia-telangiectasia mutated) protein levels, and residual ATM kinase activity. Some ...
3. `medquad_ans_0001398-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
4. `medquad_ans_0001397-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
5. `medquad_ans_0001395-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...

**Answer**:

Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging (progeria). Features include failure to gain weight and grow at the expected rate (failure to thrive), abnormally small head size (microcephaly), and impaired development of the nervous system. Affected individuals have extreme sensitivity to sunlight (photosensitivity), and even small amounts of sun exposure can cause sunburn. Other possible signs and symptoms include hearing loss, eye abnormalities, severe tooth decay, bone abnormalities, and changes in the brain visible on brain scans. The syndrome is divided into three subtypes: classical (type I) or type A, which has an onset in early childhood (usually after age 1 year) and is sometimes called "moderate" or "classic" Cockayne syndrome; type II, which is more severe and presents at birth with growth and developmental abnormalities, also known as cerebro-oculo-facio-skeletal (COFS

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

**Scores**: context_relevance=0.667, groundedness=0.939, answer_relevance=0.489

**Top docs**:

1. `medquad_ans_0000505-2` (score=0.9993) - Although its exact prevalence is unknown, hypokalemic periodic paralysis is estimated to affect 1 in 100,000 people. Men tend to experience symptoms of this condition more often than women.
2. `medquad_ans_0000494-2` (score=0.9992) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
3. `medquad_ans_0000168-3__dup1` (score=0.0022) - KSS is a slowly progressive disorder. The prognosis for individuals with KSS varies depending on the severity and the number of organs involved. Early diagnosis and periodic electrocardiogram (ECG) are important since...
4. `medquad_ans_0000126-4__dup1` (score=0.0001) - The NINDS conducts and supports research on neuromuscular disorders such as the familial periodic paralyses. These studies are aimed at increasing knowledge about these disorders and finding ways to prevent, treat, an...
5. `medquad_ans_0000126-3__dup1` (score=0.0001) - The prognosis for the familial periodic paralyses varies. Chronic attacks may result in progressive weakness that persists between attacks. Some cases respond well to treatment, which can prevent or reverse progressiv...

**Answer**:

Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people [medquad_ans_0000494-2].

**Supported citations**: `medquad_ans_0000494-2`
**Unsupported citations**: _none_

### Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Scores**: context_relevance=0.600, groundedness=0.000, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000121-12` (score=0.0115) - Eating, diet, and nutrition can play a role in causing, preventing, and treating some of the diseases and disorders of the digestive system that are affected by smoking, including heartburn and GERD, liver diseases, C...
2. `medquad_ans_0000121-3__dup3` (score=0.0059) - Smoking contributes to many common disorders of the digestive system, such as heartburn and gastroesophageal reflux disease (GERD), peptic ulcers, and some liver diseases. Smoking increases the risk of Crohns disease,...
3. `medquad_ans_0000103-9` (score=0.0048) - Researchers have not found that eating, diet, and nutrition play a major role in causing or preventing gastritis.
4. `medquad_ans_0000846-2` (score=0.0027) - PXE affects approximately 1 in 50,000 people worldwide. For reasons that are unclear, this disorder is diagnosed twice as frequently in females as in males.
5. `medquad_ans_0000163-2` (score=0.0012) - Celiac disease is a common disorder. Its prevalence has been estimated at about 1 in 100 people worldwide.

**Answer**:

The provided context does not mention "Pyle disease" or any condition with a similar name. Therefore, based on the given information, it is not possible to answer the question. [No relevant context]

**Supported citations**: _none_
**Unsupported citations**: `No relevant context`

### Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Scores**: context_relevance=0.800, groundedness=0.930, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000051-3__dup2` (score=0.9954) - Blood clots can form in your body's deep veins if: A vein's inner lining is damaged. Injuries caused by physical, chemical, or biological factors can damage the veins. Such factors include surgery, serious injuries, i...
2. `medquad_ans_0000841-4` (score=0.0028) - The risk of developing an abnormal clot in a blood vessel depends on whether a person inherits one or two copies of the F2 gene mutation that causes prothrombin thrombophilia. In the general population, the risk of de...
3. `medquad_ans_0000108-4__dup2` (score=0.0002) - Pulmonary embolism (PE) occurs equally in men and women. The risk increases with age. For every 10 years after age 60, the risk of having PE doubles. Certain inherited conditions, such as factor V Leiden, increase the...
4. `medquad_ans_0000104-1__dup4` (score=0.0001) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent b...
5. `medquad_ans_0000108-8` (score=0.0001) - Preventing pulmonary embolism (PE) begins with preventing deep vein thrombosis (DVT). Knowing whether you're at risk for DVT and taking steps to lower your risk are important. Exercise your lower leg muscles if you're...

**Answer**:

Deep Vein Thrombosis (DVT) is caused by a combination of factors including damage to a vein's inner lining, sluggish or slow blood flow, and blood that is thicker or more likely to clot than normal. Damage to the vein's inner lining can result from injuries caused by physical, chemical, or biological factors such as surgery, serious injuries, inflammation, or immune responses. Sluggish blood flow may occur due to lack of motion, such as after surgery, being ill and bedridden for a long time, or prolonged sitting during travel. Blood that is thicker or more prone to clotting can be due to inherited conditions like factor V Leiden or the use of hormone therapy or birth control pills [medquad_ans_0000051-3__dup2]. Additionally, certain conditions such as being bedridden, having surgery or a broken bone, or having diseases like chronic heart disease or high blood pressure increase the

**Supported citations**: `medquad_ans_0000051-3__dup2`
**Unsupported citations**: _none_

### Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Scores**: context_relevance=0.714, groundedness=0.664, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0004081-1` (score=0.5514) - Cytochrome C oxidase deficiency (COX deficiency) is a condition that can affect several parts of the body including the skeletal muscles, heart, brain and liver. The range and severity of signs and symptoms can vary w...
2. `medquad_ans_0004086-1` (score=0.0234) - Mitochondrial genetic disorders refer to a group of conditions that affect the mitochondria (the structures in each cell of the body that are responsible for making energy). People with these conditions can present at...
3. `medquad_ans_0004079-1` (score=0.0011) - Complex II deficiency is a mitochondrial disease. Mitochondria are specialized compartments in cells that create more than 90% of the energy needed by the body. In mitochondrial diseases, the mitochondria don't work c...
4. `medquad_ans_0000265-2` (score=0.0008) - In Eastern Europe, cytochrome c oxidase deficiency is estimated to occur in 1 in 35,000 individuals. The prevalence of this condition outside this region is unknown.
5. `medquad_ans_0000196-3__dup1` (score=0.0005) - The prognosis for patients with mitochondrial myopathies varies greatly, depending largely on the type of disease and the degree of involvement of various organs. These disorders cause progressive weakness and can lea...

**Answer**:

The provided context does not contain information about mitochondrial complex IV deficiency. While it discusses cytochrome c oxidase deficiency (which is equivalent to complex IV deficiency in mitochondrial function), no specific details about complex IV deficiency are given in the context. Therefore, based on the available information, it is not possible to answer the question.

[medquad_ans_0004081-1]

**Supported citations**: `medquad_ans_0004081-1`
**Unsupported citations**: _none_

### Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Scores**: context_relevance=0.857, groundedness=0.529, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000078-1` (score=0.4373) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...
2. `medquad_ans_0000043-1` (score=0.0030) - Alpha-methylacyl-CoA racemase (AMACR) deficiency is a disorder that causes a variety of neurological problems that begin in adulthood and slowly get worse. People with AMACR deficiency may have a gradual loss in intel...
3. `medquad_ans_0000032-4__dup1` (score=0.0011) - The NINDS supports and conducts a broad range of basic and clinical research on cerebellar and spinocerebellar degeneration, including work aimed at finding the cause(s) of ataxias and ways to treat, cure, and, ultima...
4. `medquad_ans_0000495-1__dup1` (score=0.0007) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...
5. `medquad_ans_0000097-1` (score=0.0004) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...

**Answer**:

Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement, with the hallmark being difficulty coordinating movements (ataxia), which is often the first symptom. Most affected individuals also have oculomotor apraxia, making it difficult to move their eyes side-to-side, requiring them to turn their head to see peripheral vision. There are multiple types, which are very similar but caused by mutations in different genes. The two most common types (types 1 and 2) share features including involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, ataxia begins around age 4; in type 2, it begins around age 15. Chorea and myoclonus tend to disappear gradually in type 1 but persist throughout life in type 2. Individuals with type 1 often

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Scores**: context_relevance=0.909, groundedness=0.951, answer_relevance=0.909

**Top docs**:

1. `medquad_ans_0000221-1` (score=0.9399) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that fo...
2. `medquad_ans_0000785-2` (score=0.0008) - Perrault syndrome is a rare disorder; fewer than 100 affected individuals have been described in the medical literature. It is likely that the condition is underdiagnosed, because males without an affected sister will...
3. `medquad_ans_0004109-1` (score=0.0004) - Mondini dysplasia is a type of inner ear malformation that is present at birth (congenital). Individuals with Mondini dysplasia have one and a half coils of the cochlea instead of the normal two coils. It may occur in...
4. `medquad_ans_0002637-1` (score=0.0003) - Goldenhar disease is a condition that is present at birth and mainly affects the development of the eye, ear and spine. Affected individuals commonly have a partially formed ear (microtia) or totally absent ear (anoti...
5. `medquad_ans_0000248-2` (score=0.0003) - Craniofacial microsomia has been estimated to occur in between 1 in 5,600 and 1 in 26,550 newborns. However, this range may be an underestimate because not all medical professionals agree on the criteria for diagnosis...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects the development of the ears and teeth. In people with this condition, the structures that form the inner ear are usually completely absent (labyrinthine aplasia). This leads to a form of hearing loss called sensorineural deafness that is present from birth (congenital). Due to the inner ear's role in balance and hearing, motor skill development, such as sitting and crawling, may be delayed in affected infants. Additionally, individuals often have abnormally small outer ears (microtia) with narrow ear canals and unusually small, widely spaced teeth (microdontia) [medquad_ans_0000221-1].

**Supported citations**: `medquad_ans_0000221-1`
**Unsupported citations**: _none_

### Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.977, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000571-4` (score=0.9983) - This condition is inherited in an autosomal dominant pattern, which means a mutation in one copy of the FGFR2, FGFR3, or FGF10 gene in each cell is sufficient to cause the disorder. In some cases, an affected person i...
2. `medquad_ans_0000378-4` (score=0.9969) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of Floating-Harbor syndrome result from new mutations...
3. `medquad_ans_0000889-4` (score=0.9962) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
4. `medquad_ans_0000064-4` (score=0.9898) - Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases of Apert syndrome result from new mutations in ...
5. `medquad_ans_0000540-5` (score=0.0001) - These resources address the diagnosis or management of Jackson-Weiss syndrome: - Gene Review: Gene Review: FGFR-Related Craniosynostosis Syndromes - Genetic Testing Registry: Jackson-Weiss syndrome - MedlinePlus Encyc...

**Answer**:

Yes, Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases result from new mutations in the gene and occur in people with no history of the disorder in their family. Individuals with Apert syndrome can pass along the condition to the next generation [medquad_ans_0000064-4].

**Supported citations**: `medquad_ans_0000064-4`
**Unsupported citations**: _none_

### Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Scores**: context_relevance=0.750, groundedness=0.550, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.1319) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organi...
2. `medquad_ans_0000341-3__dup2` (score=0.0073) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact ...
3. `medquad_ans_0000341-1__dup3` (score=0.0047) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella bu...
4. `medquad_ans_0000359-2` (score=0.0035) - This condition affects about 1 per million people worldwide. It is much more common in certain areas of the province of Quebec, Canada.
5. `medquad_ans_0000341-4__dup2` (score=0.0027) - In the United States, Q fever outbreaks have resulted mainly from occupational exposure involving veterinarians, meat processing plant workers, sheep and dairy workers, livestock farmers, and researchers at facilities...

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air that contains airborne barnyard dust contaminated by dried placental material, birth fluids, and excreta of infected animals. Other modes of transmission, including tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission, are rare. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, people with acute Q fever develop a chronic form of the disease. Treatment of the acute form is with antibiotics,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Scores**: context_relevance=0.333, groundedness=0.000, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000186-2` (score=0.9989) - Chronic granulomatous disease is estimated to occur in 1 in 200,000 to 250,000 people worldwide.
2. `medquad_ans_0000580-2` (score=0.9988) - Langerhans cell histiocytosis is a rare disorder. Its prevalence is estimated at 1 to 2 in 100,000 people.
3. `medquad_ans_0000483-2` (score=0.9986) - Histiocytosis-lymphadenopathy plus syndrome is a rare disorder, affecting approximately 100 individuals worldwide.
4. `medquad_ans_0000073-3__dup1` (score=0.9983) - COFS is a fatal disease. Most children do not live beyond five years.
5. `medquad_ans_0000161-3__dup2` (score=0.9979) - IRD is a fatal disease, but some children will survive into their teens and twenties, and possibly even beyond.

**Answer**:

The provided context does not specify who is at risk for Childhood Interstitial Lung Disease (ILD). Therefore, based on the given information, the risk population cannot be determined. [No supporting source available]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source available`

### Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Scores**: context_relevance=0.833, groundedness=0.967, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.9011) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in lat...
2. `medquad_ans_0000555-1` (score=0.0113) - Kawasaki disease is a sudden and time-limited (acute) illness that affects infants and young children. Affected children develop a prolonged fever lasting several days, a skin rash, and swollen lymph nodes in the neck...
3. `medquad_ans_0000663-2` (score=0.0091) - MPAN is a rare condition that is estimated to affect less than 1 in 1 million people.
4. `medquad_ans_0000136-5__dup1` (score=0.0035) - Vasculitis can affect people of all ages and races and both sexes. Some types of vasculitis seem to occur more often in people who: Have certain medical conditions, such as chronic hepatitis B or C infection Have cert...
5. `medquad_ans_0000136-9` (score=0.0026) - You can't prevent vasculitis. However, treatment can help prevent or delay the complications of vasculitis. People who have severe vasculitis are treated with prescription medicines. Rarely, surgery may be done. Peopl...

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
3. `medquad_ans_0001093-2` (score=0.0007) - 3-methylglutaconyl-CoA hydratase deficiency is a rare disorder; at least 20 cases have been reported in the scientific literature.
4. `medquad_ans_0001082-2` (score=0.0004) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyry...
5. `medquad_ans_0006585-1` (score=0.0001) - 3-Hydroxyisobutyric aciduria is a rare metabolic condition in which the body is unable to breakdown certain amino acids (the building blocks of protein). This leads to a toxic buildup of particular acids known as orga...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of the amino acid isoleucine. The inability to process isoleucine correctly leads to the buildup of the amino acid in the body, which can cause a variety of health problems that vary widely from severe and life-threatening to mild or absent. Signs and symptoms can begin a few days after birth or later in childhood and often include poor feeding, lack of energy (lethargy), vomiting, and irritability. These symptoms may progress to serious medical problems such as difficulty breathing, seizures, and coma. Additional issues can include poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills like standing and walking. Symptoms may be triggered by prolonged fasting, infections, or increased protein

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Scores**: context_relevance=0.750, groundedness=0.537, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000501-2` (score=0.0008) - Hypochondrogenesis and achondrogenesis, type 2 (a similar skeletal disorder) together affect 1 in 40,000 to 60,000 newborns.
2. `medquad_ans_0005840-3` (score=0.0001) - How is spondyloepiphyseal dysplasia congenita inherited? Spondyloepiphyseal dysplasia (SEDC) is typically inherited in an autosomal dominant manner. This means that one altered (mutated) gene in each cell is sufficien...
3. `medquad_ans_0000947-1` (score=0.0001) - Spondyloepiphyseal dysplasia congenita is an inherited bone growth disorder that results in short stature (dwarfism), skeletal abnormalities, and problems with vision and hearing. This condition affects the bones of t...
4. `medquad_ans_0005840-1` (score=0.0001) - Spondyloepiphyseal dysplasia congenita is an inherited disorder of bone growth that affects the bones of the spine and ends of the long bones in the arms and legs. Features of this condition include short stature (dwa...
5. `medquad_ans_0000268-1` (score=0.0001) - Czech dysplasia is an inherited condition that affects joint function and bone development. People with this condition have joint pain (osteoarthritis) that begins in adolescence or early adulthood. The joint pain mai...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include short stature (dwarfism) from birth, a very short trunk and neck, and shortened limbs. Hands and feet are usually average-sized. Adult height ranges from 3 feet to just over 4 feet. Abnormal curvature of the spine (kyphoscoliosis and lordosis) becomes more severe during childhood. Instability of the spinal bones (vertebrae) in the neck may increase the risk of spinal cord damage. Other skeletal features include flattened vertebrae (platyspondyly), an abnormality of the hip joint that causes the upper leg bones to turn inward (coxa vara), a foot deformity called a clubfoot, and a broad, barrel-shaped chest. Abnormal development of the chest can cause problems with breathing. Arthritis and decreased joint mobility often develop early in life [medquad_ans_000094

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Scores**: context_relevance=1.000, groundedness=0.543, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005219-3` (score=0.9912) - What causes pyridoxine-dependent epilepsy? Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each ...
2. `medquad_ans_0000118-3__dup1` (score=0.0059) - Treating the underlying cause of the disorder may improve symptoms. However, the encephalopathy may cause permanent structural changes and irreversible damage to the brain. Some encephalopathies can be fatal.
3. `medquad_ans_0000118-2__dup1` (score=0.0033) - Treatment is symptomatic and varies, according to the type and severity of the encephalopathy. Your physician can provide specific instructions for proper care and treatment. Anticonvulsants may be prescribed to reduc...
4. `medquad_ans_0000162-4__dup1` (score=0.0020) - The NINDS supports broad and varied programs of research on epilepsy and other seizure disorders. This research is aimed at discovering new ways to prevent, diagnose, and treat these disorders and, ultimately, to find...
5. `medquad_ans_0000120-2__dup1` (score=0.0012) - Once epilepsy is diagnosed, it is important to begin treatment as soon as possible. For about 70 percent of those diagnosed with epilepsy, seizures can be controlled with modern medicines and surgical techniques. Some...

**Answer**:

Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The ALDH7A1 gene provides instructions for making an enzyme called α-aminoadipic semialdehyde (α-AASA) dehydrogenase, also known as antiquitin. This enzyme is involved in the breakdown of the protein building block (amino acid) lysine in the brain. When antiquitin is deficient, a molecule that interferes with vitamin B6 function builds up in various tissues. It is unclear how a lack of pyridoxine causes the seizures that are characteristic of this condition. Some individuals with pyridoxine-dependent epilepsy do not have identified mutations in the ALDH7A1 gene. In these cases, the cause of the condition is unknown [medquad_ans_00052

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Scores**: context_relevance=0.875, groundedness=0.897, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000073-3__dup1` (score=0.9980) - COFS is a fatal disease. Most children do not live beyond five years.
2. `medquad_ans_0000161-3__dup2` (score=0.9974) - IRD is a fatal disease, but some children will survive into their teens and twenties, and possibly even beyond.
3. `medquad_ans_0000004_6-2` (score=0.9949) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
4. `medquad_ans_0000006_7-3` (score=0.9893) - The cause of most childhood CNS germ cell tumors is not known.
5. `medquad_ans_0000015_1-2` (score=0.9890) - The cause of most childhood extracranial germ cell tumors is unknown.

**Answer**:

Children who are infected with the Epstein-Barr virus, the human immunodeficiency virus (HIV), or have certain diseases of the immune system are at risk for childhood Hodgkin lymphoma. Additionally, children with a personal history of mononucleosis ("mono") or those with a parent or sibling who has a personal history of Hodgkin lymphoma are at increased risk. Being exposed to common infections in early childhood may decrease the risk of Hodgkin lymphoma due to its effect on the immune system [medquad_ans_0000004_6-2].

**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

### Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Scores**: context_relevance=0.800, groundedness=0.929, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000095-1__dup4` (score=0.8953) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation ...
2. `medquad_ans_0001795-1` (score=0.7929) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow...
3. `medquad_ans_0000084-2` (score=0.0050) - Atopic dermatitis is a common disorder that affects 10 to 20 percent of children and 5 to 10 percent of adults.
4. `medquad_ans_0000480-2` (score=0.0029) - Hidradenitis suppurativa was once thought to be a rare condition because only the most severe cases were reported. However, recent studies have shown that the condition affects at least 1 in 100 people when milder cas...
5. `medquad_ans_0000509-2` (score=0.0027) - HID is a rare disorder. Its prevalence is unknown.

**Answer**:

Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These lesions are commonly found on the knees, elbows, buttocks, and shoulder blades and may also appear on the face, groin, back, or scalp. The rash often presents with a burning sensation before lesion formation and may appear as erosions and excoriations due to scratching. Lesions are typically scratched off by the time a patient presents for a physical exam. The condition is frequently associated with an inability to digest gluten and is often linked to celiac disease, although less than 20 percent of people with DH have symptoms of celiac disease. It is typically treated with the drug dapsone [medquad_ans_0001795-1]. Additionally, patients with dermatitis herpetiformis may experience dental enamel defects in permanent teeth, another manifestation of celiac disease [

**Supported citations**: `medquad_ans_0001795-1`
**Unsupported citations**: _none_

### Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Scores**: context_relevance=0.857, groundedness=0.960, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000709-5` (score=0.9926) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Regi...
2. `medquad_ans_0000040-2__dup1` (score=0.9899) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors ...
3. `medquad_ans_0000040-3__dup1` (score=0.9090) - Most individuals with myotonia congenita lead long, productive lives. Although muscle stiffness may interfere with walking, grasping, chewing, and swallowing, it is usually relieved with exercise.
4. `medquad_ans_0000771-5` (score=0.3355) - These resources address the diagnosis or management of paramyotonia congenita: - Genetic Testing Registry: Paramyotonia congenita of von Eulenburg - Periodic Paralysis International: How is Periodic Paralysis Diagnose...
5. `medquad_ans_0000812-5` (score=0.0176) - These resources address the diagnosis or management of potassium-aggravated myotonia: - Genetic Testing Registry: Potassium aggravated myotonia These resources from MedlinePlus offer information about the diagnosis an...

**Answer**:

Most people with myotonia congenita do not require special treatments; stiff muscles usually resolve with exercise or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors have had some success with medications such as quinine or anticonvulsant drugs like phenytoin. Physical therapy and other rehabilitative therapies are also sometimes used to improve muscle function [medquad_ans_0000040-2__dup1].

**Supported citations**: `medquad_ans_0000040-2__dup1`
**Unsupported citations**: _none_

### Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Scores**: context_relevance=0.833, groundedness=0.291, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000133-12` (score=0.0097) - A person cannot prevent Wilson disease; however, people with a family history of Wilson disease, especially those with an affected sibling or parent, should talk with a health care provider about testing. A health car...
2. `medquad_ans_0000276-3__dup1` (score=0.0021) - Early onset of the disease may foretell a worse prognosis than later onset. If the disorder is detected early and treated appropriately, an individual with WD can usually enjoy normal health and a normal lifespan. If ...
3. `medquad_ans_0001037-5` (score=0.0008) - These resources address the diagnosis or management of Wilson disease: - Gene Review: Gene Review: Wilson Disease - Genetic Testing Registry: Wilson's disease - MedlinePlus Encyclopedia: Wilson's disease - National Hu...
4. `medquad_ans_0000133-8` (score=0.0006) - People who have Wilson disease that is not treated or diagnosed early can have serious complications, such as - cirrhosisscarring of the liver - kidney damageas liver function decreases, the kidneys may be damaged - p...
5. `medquad_ans_0000276-4__dup1` (score=0.0004) - The National Institute of Neurological Disorders and Stroke, the Eunice Kennedy Shriver National Institute of Child Health and Human Development, and other institutes of the National Institutes of Health (NIH) conduct...

**Answer**:

For Wilson disease, individuals with a family history, especially those with an affected sibling or parent, should talk with a health care provider about testing. Early diagnosis and treatment can reduce or even prevent organ damage. A health care provider may diagnose Wilson disease before symptoms appear and may refer individuals to a geneticist for genetic testing to identify gene mutations. If detected early and treated appropriately, individuals can usually enjoy normal health and a normal lifespan. Treatment is lifelong and may include diagnostic tests, drug therapy, and genetic counseling. Resources such as the Gene Review, Genetic Testing Registry, and MedlinePlus provide information on diagnosis and management [medquad_ans_0000133-12, medquad_ans_0001037-5, medquad_ans_0000276-3__dup1]. Untreated Wilson disease can lead to serious complications including cirrhosis, kidney damage, nervous system problems, liver

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000133-12, medquad_ans_0001037-5, medquad_ans_0000276-3__dup1`

### Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.935, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000592-3` (score=0.9980) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are re...
2. `medquad_ans_0000879-3` (score=0.0004) - Ring chromosome 20 syndrome is caused by a chromosomal abnormality known as a ring chromosome 20 or r(20). A ring chromosome is a circular structure that occurs when a chromosome breaks in two places and its broken en...
3. `medquad_ans_0000814-3` (score=0.0002) - Mutations in the MECP2 gene cause PPM-X syndrome. The MECP2 gene provides instructions for making a protein called MeCP2 that is critical for normal brain function. Researchers believe that this protein has several fu...
4. `medquad_ans_0000108-4__dup1` (score=0.0001) - The NINDS conducts and supports a broad program of basic and clinical research on all types of epilepsy, including Dravet syndrome. Study of the genetic defects responsible for Dravet syndrome and related disorders is...
5. `medquad_ans_0000162-4__dup1` (score=0.0001) - The NINDS supports broad and varied programs of research on epilepsy and other seizure disorders. This research is aimed at discovering new ways to prevent, diagnose, and treat these disorders and, ultimately, to find...

**Answer**:

Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. In about one-third of cases, the cause is unknown and described as cryptogenic. In other cases, Lennox-Gastaut syndrome can be associated with genetic disorders such as tuberous sclerosis complex or brain malformations like cortical dysplasia. However, no specific genes have been identified as directly causing the syndrome [medquad_ans_0000592-3].

**Supported citations**: `medquad_ans_0000592-3`
**Unsupported citations**: _none_

### Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Scores**: context_relevance=0.700, groundedness=0.969, answer_relevance=0.460

**Top docs**:

1. `medquad_ans_0000065-2` (score=0.9995) - Arginase deficiency is a very rare disorder; it has been estimated to occur once in every 300,000 to 1,000,000 individuals.
2. `medquad_ans_0000066-2` (score=0.9994) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
3. `medquad_ans_0000442-2` (score=0.9993) - Guanidinoacetate methyltransferase deficiency is a very rare disorder. About 80 affected individuals have been described in the medical literature. Of these, approximately one-third are of Portuguese origin.
4. `medquad_ans_0000067-2` (score=0.9993) - Argininosuccinic aciduria occurs in approximately 1 in 70,000 newborns.
5. `medquad_ans_0000416-2` (score=0.9991) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...

**Answer**:

The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families. [medquad_ans_0000066-2]

**Supported citations**: `medquad_ans_0000066-2`
**Unsupported citations**: _none_

### Example 231 (`0002399-2`)

**Question**: What are the symptoms of Focal dermal hypoplasia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0002399-2` (score=0.9990) - What are the signs and symptoms of Focal dermal hypoplasia? Focal dermal hypoplasia is usually evident from birth and primarily affects the skin, skeleton, eyes, and face. The signs and symptoms of vary widely, althou...
2. `medquad_ans_0000048-3__dup1` (score=0.9986) - Although the skin abnormalities usually regress, and sometimes disappear completely, there may be residual neurological difficulties.
3. `medquad_ans_0002401-1` (score=0.0004) - What are the signs and symptoms of Focal facial dermal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Focal facial dermal dysplasia. If the information is available, the ...
4. `medquad_ans_0002399-3` (score=0.0001) - How is this condition inherited? Focal dermal hypoplasia is caused by mutations in the PORCN gene and is inherited in an X-linked dominant manner. Many cases of focal dermal hypoplasia result from a new mutation and o...
5. `medquad_ans_0000201-1` (score=0.0001) - Cole disease is a disorder that affects the skin. People with this disorder have areas of unusually light-colored skin (hypopigmentation), typically on the arms and legs, and spots of thickened skin on the palms of th...

**Answer**:

The symptoms of Focal dermal hypoplasia include skin abnormalities such as streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), telangiectases, and streaks of slightly darker or lighter skin. These can cause pain, itching, irritation, or lead to skin infections. With age, most develop wart-like growths (papillomas) around the nostrils, lips, anus, and female genitalia, and may also be present in the throat, causing problems with swallowing, breathing, or sleeping. Other features include small, ridged fingernails and toenails, sparse or brittle scalp hair, hand and foot abnormalities such as missing fingers or toes (oligodactyly), webbed or fused fingers or toes (syndactyly), and a deep split in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Scores**: context_relevance=0.909, groundedness=0.827, answer_relevance=0.836

**Top docs**:

1. `medquad_ans_0000418-1__dup2` (score=0.9876) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...
2. `medquad_ans_0000418-2__dup1` (score=0.9752) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
3. `medquad_ans_0000415-1__dup2` (score=0.9549) - A single-celled parasite called Toxoplasma gondii causes a disease known as toxoplasmosis. While the parasite is found throughout the world, more than 60 million people in the United States may be infected with the To...
4. `medquad_ans_0000432-2__dup1` (score=0.9464) - Whipworm is a soil-transmitted helminth (STH) and is the third most common roundworm of humans. Whipworm causes an infection called trichuriasis and often occurs in areas where human feces is used as fertilizer or whe...
5. `medquad_ans_0000265-4__dup2` (score=0.9355) - Loiasis is an infection caused by the parasitic worm Loa loa.

**Answer**:

People at risk for trichinellosis (also known as trichinosis) include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating undercooked wild game, especially bear meat, puts individuals at risk. Additionally, individuals who taste very small amounts of undercooked meat during preparation or cooking are also at risk. Pregnant women and those with compromised immune systems are not specifically mentioned as being at risk for trichinellosis in the provided context, though they are noted as being cautious with toxoplasmosis [medquad_ans_0000418-2__dup1].

Thus, based on the context, the primary at-risk groups are consumers of raw or undercooked meat, especially from wild game or pork [medquad_ans_0000418-1__dup2]medquad_ans_0000

**Supported citations**: `medquad_ans_0000418-2__dup1`, `medquad_ans_0000418-1__dup2`
**Unsupported citations**: _none_

### Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.744, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000120-2__dup1` (score=0.0036) - Once epilepsy is diagnosed, it is important to begin treatment as soon as possible. For about 70 percent of those diagnosed with epilepsy, seizures can be controlled with modern medicines and surgical techniques. Some...
2. `medquad_ans_0000108-3__dup1` (score=0.0019) - As children with Dravet syndrome get older, their decline in cognitive function stabilizes, and in many, it improves slightly. However, most teenagers with Dravet syndrome are dependent on caregivers. The degree of co...
3. `medquad_ans_0000121-3__dup1` (score=0.0017) - Todd's paralysis is an indication that an individual has had an epileptic seizure. The outcome depends on the effects of the seizure and the subsequent treatment of the epilepsy.
4. `medquad_ans_0000108-2__dup2` (score=0.0016) - Seizures in Dravet syndrome are difficult to control, but can be reduced by anticonvulsant drugs. A ketogenic diet, high in fats and low in carbohydrates, also may be beneficial.
5. `medquad_ans_0000117-2` (score=0.0001) - Benign familial neonatal seizures occurs in approximately 1 in 100,000 newborns.

**Answer**:

The provided context does not describe the symptoms of Dravet syndrome. While it mentions that seizures in Dravet syndrome are difficult to control and can be reduced by anticonvulsant drugs or a ketogenic diet, and that cognitive decline may stabilize or improve slightly with age, it does not list specific symptoms such as developmental delays, prolonged seizures, or other clinical features associated with Dravet syndrome. Therefore, based on the given information, the symptoms of Dravet syndrome cannot be determined. [medquad_ans_0000108-3\_\_dup1][medquad_ans_0000108-2__dup2]

**Supported citations**: `medquad_ans_0000108-3__dup1`, `medquad_ans_0000108-2__dup2`
**Unsupported citations**: _none_

### Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Scores**: context_relevance=0.889, groundedness=0.976, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0000015-4__dup1` (score=0.0018) - The NINDS supports research on gene-linked neurodegenerative disorders such as Alpers' disease. The goals of this research are to increase scientific understanding of these disorders, and to find ways to prevent, trea...
2. `medquad_ans_0000011-13` (score=0.0010) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...
3. `medquad_ans_0000239-4__dup1` (score=0.0006) - The NINDS conducts and supports research on TSEs. This research is aimed at determining how abnormal prion proteins lead to disease, at finding better tests for diagnosing CJD and other disorders, and ultimately at fi...
4. `medquad_ans_0000004-29` (score=0.0002) - Basic research helps scientists gain new knowledge about a disease process, including how and why it starts and progresses. In Alzheimers disease, basic research seeks to identify the cellular, molecular, and genetic ...
5. `medquad_ans_0000089-4__dup1` (score=0.0001) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system, and to use that knowledge to reduce the burden of neurological di...

**Answer**:

People with inherited CJD have an abnormal gene that leads to changes in their prion protein. This gene makes the protein likely to assume the abnormal shape [medquad_ans_0000011-13].

**Supported citations**: `medquad_ans_0000011-13`
**Unsupported citations**: _none_

### Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0006430-1` (score=0.9994) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information ...
2. `medquad_ans_0000249-2` (score=0.0004) - Craniofacial-deafness-hand syndrome is an extremely rare condition. Only a few cases have been reported in the scientific literature.
3. `medquad_ans_0000073-2` (score=0.0001) - Arts syndrome appears to be extremely rare. Only a few families with this disorder have been described in the medical literature.
4. `medquad_ans_0000446-2` (score=0.0001) - Hand-foot-genital syndrome is very rare; only a few families with the condition have been reported worldwide.
5. `medquad_ans_0000201-1` (score=0.0001) - Cole disease is a disorder that affects the skin. People with this disorder have areas of unusually light-colored skin (hypopigmentation), typically on the arms and legs, and spots of thickened skin on the palms of th...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, cleft palate, cleft upper lip, high palate, hydronephrosis, hypoplasia of the radius, hypotelorism, long face, mesomelia, narrow face, oligodactyly (hands), proximal placement of thumb, and proximal radial head dislocation [medquad_ans_0006430-1].

**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

### Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Scores**: context_relevance=1.000, groundedness=0.888, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.9994) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the ta...
2. `medquad_ans_0000033-3__dup1` (score=0.9942) - Average lifespan has been improving for years, for unknown reasons, and varies with the severity of the underlying mutations, ATM (ataxia-telangiectasia mutated) protein levels, and residual ATM kinase activity. Some ...
3. `medquad_ans_0000260-3__dup1` (score=0.1384) - Striatonigral degeneration progresses slowly. Some patients have normal life expectancy.
4. `medquad_ans_0000220-3__dup1` (score=0.0332) - There is no cure for OPCA. The disorder is slowly progressive with death usually occurring approximately 20 years after onset.
5. `medquad_ans_0000055-3__dup1` (score=0.0030) - Kennedy's disease is slowly progressive. Individuals tend to remain ambulatory until late in the disease, although some may be wheelchair-bound during later stages. The life span of individuals with Kennedy's disease ...

**Answer**:

The signs and symptoms of Premature aging Okamoto type include abnormality of blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face. The frequency of these symptoms is estimated based on data from the Human Phenotype Ontology (HPO), which compiles information from sources such as Orphanet, with frequencies often listed as percentages or fractions representing the proportion of patients exhibiting each symptom [medquad_ans_0005029-1].

**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

### Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Scores**: context_relevance=0.222, groundedness=0.000, answer_relevance=0.844

**Top docs**:

1. `medquad_ans_0000437-2` (score=0.9994) - Gray platelet syndrome appears to be a rare disorder. About 60 cases have been reported worldwide.
2. `medquad_ans_0001065-2` (score=0.9994) - The estimated incidence of X-linked thrombocytopenia is between 1 and 10 per million males worldwide; this condition is rarer among females.
3. `medquad_ans_0000245-2` (score=0.9994) - This condition is very rare; it probably affects 200 to 300 people worldwide. Reported estimates of Costello syndrome prevalence range from 1 in 300,000 to 1 in 1.25 million people.
4. `medquad_ans_0000791-2` (score=0.9992) - Pfeiffer syndrome affects about 1 in 100,000 individuals.
5. `medquad_ans_0000975-2` (score=0.9991) - Tetrasomy 18p is a rare disorder. It is known to affect about 250 families worldwide.

**Answer**:

The provided context does not contain any information about thrombocytopenia-absent radius syndrome. Therefore, it is not possible to determine how many people are affected by this condition based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Scores**: context_relevance=0.545, groundedness=0.814, answer_relevance=0.691

**Top docs**:

1. `medquad_ans_0000181-4__dup1` (score=0.9984) - The NINDS supports research on genetic disorders, including the leukodystrophies. The goals of this research are to increase scientific understanding of these disorders, and to find ways to prevent, treat, and, ultima...
2. `medquad_ans_0000227-4__dup1` (score=0.9983) - NINDS supports research on gene-linked disorders, including the leukodystrophies. The goals of this research are to increase scientific understanding of these disorders and to find ways to prevent, treat, and ultimate...
3. `medquad_ans_0000065-4__dup1` (score=0.9966) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...
4. `medquad_ans_0000135-4__dup1` (score=0.9936) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...
5. `medquad_ans_0000265-4__dup1` (score=0.9934) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...

**Answer**:

Research on Tay-Sachs disease is focused on developing virus-delivered gene therapy, which has been explored in animal models and is being considered for use in humans. This approach is part of NINDS-funded research on gangliosidoses, including Tay-Sachs and Sandhoff diseases [medquad_ans_0000135-4__dup1]. Additionally, studies are using variations of magnetic resonance imaging to develop biomarkers for evaluating brain chemistry and disease progression in these disorders [medquad_ans_0000135-4__dup1]. However, specific clinical trials for Tay-Sachs disease are not detailed in the provided context.

**Supported citations**: `medquad_ans_0000135-4__dup1`
**Unsupported citations**: _none_

### Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Scores**: context_relevance=0.750, groundedness=0.544, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000220-1__dup2` (score=0.9917) - Most of us see our world in color. We enjoy looking at a lush green lawn or a red rose in full bloom. If you have a color vision defect, you may see these colors differently than most people. There are three main kind...
2. `medquad_ans_0000763-1__dup1` (score=0.9630) - Blue cone monochromatism is an inherited X-linked vision disorder. In this condition both red and green cone sensitivities are absent, however rod function and blue cone sensitivities are present. Signs and symptoms i...
3. `medquad_ans_0000008-1` (score=0.4357) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achro...
4. `medquad_ans_0000042-7` (score=0.0128) - Low vision can result from a variety of diseases, disorders, and injuries that affect the eye. Many people with low vision have age-related macular degeneration, cataracts, glaucoma, or diabetic retinopathy. Age-relat...
5. `medquad_ans_0000008-2` (score=0.0004) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on o...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. Achromatopsia also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals may also have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia is different from common forms of color vision deficiency (color blindness), in which people can perceive color but have difficulty distinguishing between certain colors, such as red and green [medquad_ans_0

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Scores**: context_relevance=0.571, groundedness=0.831, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000039-17` (score=0.9958) - A physical therapist will teach you exercises to help your recovery. You can expect some pain, discomfort, and stiffness as you begin therapy, but to get the best results from your new knee, it is important to do all ...
2. `medquad_ans_0000029-2__dup1` (score=0.9940) - Medications may ease certain symptoms, such as pain. Surgery is the only treatment available to correct functional disturbances or halt the progression of damage to the central nervous system. More than one surgery ma...
3. `medquad_ans_0000039-9` (score=0.9869) - The main reason to have knee replacement surgery is to ease pain and disability caused by arthritis or other joint problems, while preserving movement. Less commonly, it is used to correct some kinds of knee deformity.
4. `medquad_ans_0000039-11` (score=0.9823) - Treatments your doctor will likely recommend before knee replacement include - exercises to strengthen the muscles around the knee and improve flexibility - weight loss, if needed, to reduce the load the knee must bea...
5. `medquad_ans_0000039-26` (score=0.9592) - You can learn more about knee replacement from the following resources. National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) Information Clearinghouse National Institutes of Health 1 AMS Circl...

**Answer**:

The context provided does not list specific treatments for knee replacement itself, but rather outlines treatments recommended _before_ knee replacement surgery. These include:

- Exercises to strengthen the muscles around the knee and improve flexibility
- Weight loss, if needed, to reduce the load the knee must bear
- Walking aids such as canes to reduce stress on the joint
- Shoe inserts to improve knee alignment
- Medicines to relieve pain

These are pre-surgical treatments, not treatments for knee replacement surgery itself. Therefore, based on the provided context, there is no information about treatments _for_ knee replacement surgery.

Answer: The provided context does not specify treatments for knee replacement surgery. It only lists treatments recommended before knee replacement [medquad_ans_0000039-11].

**Supported citations**: `medquad_ans_0000039-11`
**Unsupported citations**: _none_

### Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.578, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000409-2` (score=0.0031) - Gitelman syndrome affects an estimated 1 in 40,000 people worldwide.
2. `medquad_ans_0001036-2` (score=0.0025) - Williams syndrome affects an estimated 1 in 7,500 to 10,000 people.
3. `medquad_ans_0000373-2` (score=0.0014) - The prevalence of FG syndrome is unknown, although several hundred cases have been reported worldwide. Researchers suspect that FG syndrome may be overdiagnosed because many of its signs and symptoms are also seen wit...
4. `medquad_ans_0000012-2` (score=0.0006) - The prevalence of activated PI3K-delta syndrome is unknown.
5. `medquad_ans_0000815-2` (score=0.0006) - Prader-Willi syndrome affects an estimated 1 in 10,000 to 30,000 people worldwide.

**Answer**:

The provided context does not contain information about the symptoms of Metabolic Syndrome. [medquad_ans_0000409-2][medquad_ans_0001036-2][medquad_ans_0000373-2][medquad_ans_0000012-2][medquad_ans_0000815-2] discuss the prevalence of various genetic syndromes but do not address the symptoms of Metabolic Syndrome. Therefore, the requested information is not available in the context.

**Supported citations**: `medquad_ans_0000409-2`, `medquad_ans_0001036-2`, `medquad_ans_0000373-2`, `medquad_ans_0000012-2`, `medquad_ans_0000815-2`
**Unsupported citations**: _none_

### Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Scores**: context_relevance=0.333, groundedness=0.000, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000154-6` (score=0.9991) - Many babies with kidney dysplasia in only one kidney have no signs of the condition. In some cases, the affected kidney may be enlarged at birth and may cause pain.
2. `medquad_ans_0000374-2` (score=0.0009) - Fibrochondrogenesis appears to be a rare disorder. About 20 affected individuals have been described in the medical literature.
3. `medquad_ans_0000976-2` (score=0.0006) - This condition occurs in 1 in 20,000 to 50,000 newborns. Type I thanatophoric dysplasia is more common than type II.
4. `medquad_ans_0000141-1__dup4` (score=0.0002) - An ectopic kidney is a birth defect in which a kidney is located below, above, or on the opposite side of its usual position. About one in 900 people has an ectopic kidney.1
5. `medquad_ans_0000830-2` (score=0.0002) - Progressive osseous heteroplasia is a rare condition. Its exact incidence is unknown.

**Answer**:

The provided context does not contain any information about the symptoms of Pyknoachondrogenesis. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Scores**: context_relevance=0.857, groundedness=0.744, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0003965-4` (score=0.0002) - Who might consider genetic carrier testing for a family history of metachromatic leukodystrophy? If someone has a family history of metachromatic leukodystrophy (MLD) or someone is known to be a carrier for MLD, indiv...
2. `medquad_ans_0000181-2__dup1` (score=0.0001) - Treatment for most of the leukodystrophies is symptomatic and supportive, and may include medications, physical, occupational, and speech therapies; and nutritional, educational, and recreational programs. Bone marrow...
3. `medquad_ans_0000139-3__dup1` (score=0.0001) - Krabbe disease in infants is generally fatal before age 2. Individuals with a later onset form of the disease generally have a milder course of the disease and live significantly longer.
4. `medquad_ans_0000181-1__dup2` (score=0.0001) - Leukodystrophy refers to progressive degeneration of the white matter of the brain due to imperfect growth or development of the myelin sheath, the fatty covering that acts as an insulator around nerve fiber. Myelin, ...
5. `medquad_ans_0001077-5` (score=0.0001) - These resources address the diagnosis or management of 18q deletion syndrome: - Gene Review: Gene Review: Leukodystrophy Overview - University of Texas Chromosome 18 Clinical Research Center These resources from Medli...

**Answer**:

The provided context does not include information about the symptoms of metachromatic leukodystrophy. [medquad_ans_0003965-4] discusses genetic carrier testing and risk for carriers in families with a history of metachromatic leukodystrophy, but does not describe symptoms. [medquad_ans_0000181-2__dup1] and [medquad_ans_0000139-3__dup1] discuss general leukodystrophy treatments and Krabbe disease, respectively, but not specific symptoms of metachromatic leukodystrophy. Therefore, based on the given context, the symptoms of metachromatic leukodystrophy cannot be determined.

**Supported citations**: `medquad_ans_0003965-4`, `medquad_ans_0000181-2__dup1`, `medquad_ans_0000139-3__dup1`
**Unsupported citations**: _none_

### Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Scores**: context_relevance=0.800, groundedness=0.224, answer_relevance=0.760

**Top docs**:

1. `medquad_ans_0000242-4` (score=0.9981) - This condition is reported to have an autosomal recessive pattern of inheritance, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition eac...
2. `medquad_ans_0001001-4` (score=0.9978) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. However, some people who inherit the altered gene never develop ...
3. `medquad_ans_0000435-4` (score=0.9977) - The inheritance pattern of GPA is unknown. Most instances are sporadic and occur in individuals with no history of the disorder in their family. Only rarely is more than one member of the same family affected by the d...
4. `medquad_ans_0000129-4` (score=0.9969) - Blau syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most affected individuals have one parent with the condition. In ...
5. `medquad_ans_0000324-4` (score=0.9969) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.

**Answer**:

The provided context does not mention chronic granulomatous disease or its inheritance pattern. Therefore, it is not possible to determine based on the given information whether chronic granulomatous disease is inherited.

Answer: Insufficient context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.926, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000062-3__dup1` (score=0.9964) - Central pain syndrome is not a fatal disorder, but the syndrome causes disabling chronic pain and suffering among the majority of individuals who have it.
2. `medquad_ans_0000061-3__dup1` (score=0.9886) - The prognosis for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict he likeliho...
3. `medquad_ans_0000266-3__dup1` (score=0.9799) - With treatment, individuals with tethered spinal cord syndrome have a normal life expectancy. However, some neurological and motor impairments may not be fully correctable. Surgery soon after symptoms emerge appears t...
4. `medquad_ans_0000061-2__dup1` (score=0.0046) - There is no cure for central cord syndrome although some people recover near-normal function. There is no standard course of treatment, although drug therapy, surgery, and rest are often part of the program. Magnetic ...
5. `medquad_ans_0000077-4__dup1` (score=0.0017) - The NINDS supports a broad program of research on disorders of the nervous system, including CIDP. Much of this research is aimed at increasing the understanding of these disorders and finding ways to prevent, treat, ...

**Answer**:

The outlook for central cord syndrome varies, but most people who have it due to trauma experience some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict the likelihood of natural neurological recovery. Those who receive medical intervention soon after injury often have good outcomes. Many people recover substantial function after their initial injury, with the ability to walk being recovered in most cases, although some impairment may remain. Improvement typically occurs first in the legs, then the bladder, and may be seen in the arms, with hand function recovering last, if at all. Recovery is generally better in younger patients compared to those over the age of 50 [medquad_ans_0000061-3__dup1]. There is no cure, but treatment options such as drug therapy, surgery, and rest are often part of the management plan, and early surgery may improve recovery chances [medquad_ans_0000061-2

**Supported citations**: `medquad_ans_0000061-3__dup1`
**Unsupported citations**: _none_

### Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.938, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000094-12` (score=0.0025) - During the prodrome and vomiting phases of cyclic vomiting syndrome, a person will generally take in little or no nutrition by mouth. During the recovery phase, the person may be quite hungry as soon as the vomiting s...
2. `medquad_ans_0000261-3` (score=0.0017) - Although the causes of cyclic vomiting syndrome have yet to be determined, researchers have proposed several factors that may contribute to the disorder. These factors include changes in brain function, hormonal abnor...
3. `medquad_ans_0000094-11` (score=0.0004) - A person should stay away from known triggers, especially during the well phase, as well as - get adequate sleep to prevent exhaustion - treat sinus problems or allergies - seek help on reducing stress and anxiety - a...
4. `medquad_ans_0000262-3__dup1` (score=0.0001) - There is no cure for these headaches. The disorder is not fatal but can cause considerable discomfort.
5. `medquad_ans_0000373-2` (score=0.0001) - The prevalence of FG syndrome is unknown, although several hundred cases have been reported worldwide. Researchers suspect that FG syndrome may be overdiagnosed because many of its signs and symptoms are also seen wit...

**Answer**:

The causes of cyclic vomiting syndrome have not been fully determined, but researchers have proposed several contributing factors. These include changes in brain function, hormonal abnormalities, and gastrointestinal problems. Many researchers believe that cyclic vomiting syndrome is a migraine-like condition, related to changes in signaling between nerve cells (neurons) in certain areas of the brain. Abnormalities in the autonomic nervous system, which controls involuntary body functions such as heart rate, blood pressure, and digestion, are also commonly observed. In some cases, particularly those beginning in childhood, changes in mitochondrial DNA have been associated with the condition. These mitochondrial DNA changes may impair the mitochondria's ability to produce energy, potentially leading to dysfunction in autonomic nervous system cells and affecting the digestive system. However, it remains unclear how these mitochondrial changes specifically cause symptoms such as nausea, vomiting, lethargy, abdominal pain, or migraines [medquad_ans_0000261-3].

**Supported citations**: `medquad_ans_0000261-3`
**Unsupported citations**: _none_

### Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Scores**: context_relevance=0.875, groundedness=0.958, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000189-1__dup2` (score=0.9981) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...
2. `medquad_ans_0000076-1__dup3` (score=0.9564) - Summary : There are many new responsibilities when you have a baby. One of them is to make sure they get the checkups that they need. Well-baby exams are important in making sure that your baby is growing and developi...
3. `medquad_ans_0000180-1__dup2` (score=0.6764) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not alwa...
4. `medquad_ans_0000869-1__dup1` (score=0.1114) - Summary : How well you and your doctor communicate with each other is one of the most important parts of getting good health care. Being prepared can help make the most of your visit. Here are some things you can brin...
5. `medquad_ans_0000520-1__dup1` (score=0.0514) - Summary : For most kids and teens, technology is an important part of their lives. They browse the Web for information, use social networking sites, text, and chat. But there can also be dangers, and it is important f...

**Answer**:

Yes, there is information about child mental health. It is important to recognize and treat mental illnesses in children early on, as once mental illness develops, it becomes a regular part of the child's behavior and is more difficult to treat. Warning signs that might indicate a more serious problem include problems in more than one setting (at school, at home, with peers), changes in appetite or sleep, social withdrawal or fear of things previously not feared, returning to behaviors common in younger children (such as bedwetting), signs of being upset (like sadness or tearfulness), signs of self-destructive behavior (such as head-banging or suddenly getting hurt often), and repeated thoughts of death. Diagnosis involves evaluating the child's signs and symptoms, medical history, and family history. Treatments may include medicines and talk therapy [medquad_ans_0000180-1__dup2].

**Supported citations**: `medquad_ans_0000180-1__dup2`
**Unsupported citations**: _none_

### Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001618-2` (score=0.9976) - What are the signs and symptoms of Cowden syndrome? Cowden syndrome is characterized primarily by multiple, noncancerous growths (called hamartomas) on various parts of the body. Approximately 99% of people affected b...
2. `medquad_ans_0001019-5` (score=0.0001) - These resources address the diagnosis or management of Vohwinkel syndrome: - Genetic Testing Registry: Mutilating keratoderma - Genetic Testing Registry: Vohwinkel syndrome, variant form These resources from MedlinePl...
3. `medquad_ans_0000246-1` (score=0.0001) - Cowden syndrome is a disorder characterized by multiple noncancerous, tumor-like growths called hamartomas and an increased risk of developing certain cancers. Almost everyone with Cowden syndrome develops hamartomas....
4. `medquad_ans_0000611-2` (score=0.0001) - In the United States, about 140,000 new cases of colorectal cancer are diagnosed each year. Approximately 3 to 5 percent of these cancers are caused by Lynch syndrome.
5. `medquad_ans_0001618-1` (score=0.0001) - Cowden syndrome is an inherited condition that is characterized primarily by multiple, noncancerous growths (called hamartomas) on various parts of the body. It is considered part of the PTEN Hamartoma Tumor Syndrome ...

**Answer**:

The symptoms of Cowden syndrome include multiple noncancerous growths (hamartomas) on the skin and/or in the mouth, with approximately 99% of affected individuals having these by the third decade of life. A majority also develop hamartomatous polyps in the gastrointestinal tract. Other common symptoms include benign diseases of the breast, thyroid, and endometrium; a rare noncancerous brain tumor called Lhermitte-Duclos disease; an enlarged head (macrocephaly); autism spectrum disorder; intellectual disability; and vascular abnormalities. Specific signs and symptoms reported in the Human Phenotype Ontology include abnormality of the pupil, abnormality of the tongue, aplasia/hypoplasia of the cerebellum, arteriovenous malformation, cognitive impairment, conjunctival hamartoma, dental malocclusion, epibulbar dermoid, exostoses, foot polydactyly

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.773, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000173-2__dup2` (score=0.9962) - There is no cure for SMA. Treatment consists of managing the symptoms and preventing complications.
2. `medquad_ans_0000208-2__dup1` (score=0.9941) - Treatments for the myopathies depend on the disease or condition and specific causes. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders. Treatment for other disor...
3. `medquad_ans_0000395-5` (score=0.9929) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
4. `medquad_ans_0000334-5` (score=0.0936) - These resources address the diagnosis or management of facioscapulohumeral muscular dystrophy: - Gene Review: Gene Review: Facioscapulohumeral Muscular Dystrophy - Genetic Testing Registry: Facioscapulohumeral muscula...
5. `medquad_ans_0000201-4__dup1` (score=0.0137) - The NINDS supports a broad range of research on neuromuscular disorders with the goal of finding ways to prevent, treat, and, ultimately, cure them.

**Answer**:

The provided context does not specify the exact treatments for Fukuyama congenital muscular dystrophy. It lists resources that address the diagnosis or management of the condition, such as gene reviews, genetic testing registry entries, and MedlinePlus encyclopedia information, but does not detail specific treatments [medquad_ans_0000395-5]. Therefore, based on the given information, the treatments for Fukuyama congenital muscular dystrophy cannot be determined.

**Supported citations**: `medquad_ans_0000395-5`
**Unsupported citations**: _none_

### Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Scores**: context_relevance=0.333, groundedness=0.000, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000872-2` (score=0.0094) - Retinoblastoma is diagnosed in 250 to 350 children per year in the United States. It accounts for about 4 percent of all cancers in children younger than 15 years.
2. `medquad_ans_0000073-3__dup1` (score=0.0088) - COFS is a fatal disease. Most children do not live beyond five years.
3. `medquad_ans_0000161-3__dup2` (score=0.0007) - IRD is a fatal disease, but some children will survive into their teens and twenties, and possibly even beyond.
4. `medquad_ans_0000722-2` (score=0.0001) - Neuroblastoma is the most common cancer in infants younger than 1 year. It occurs in 1 in 100,000 children and is diagnosed in about 650 children each year in the United States.
5. `medquad_ans_0000015_1-2` (score=0.0001) - The cause of most childhood extracranial germ cell tumors is unknown.

**Answer**:

The provided context does not contain information on how to diagnose Childhood Hodgkin Lymphoma. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0006446-4` (score=0.9465) - Is Williams syndrome inherited?
2. `medquad_ans_0000136-4` (score=0.8808) - BOR/BO syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In about 90 percent of cases, an affected person inherits the m...
3. `medquad_ans_0000803-4` (score=0.4354) - Most cases of Poland syndrome are sporadic, which means they are not inherited and occur in people with no history of the disorder in their families. Rarely, this condition is passed through generations in families. I...
4. `medquad_ans_0000392-4` (score=0.1395) - Fryns syndrome appears to be inherited in an autosomal recessive pattern, which means both copies of a gene in each cell have mutations. However, no associated gene has been identified. The parents of an individual wi...
5. `medquad_ans_0000108-4` (score=0.0885) - Bardet-Biedl syndrome is typically inherited in an autosomal recessive pattern, which means both copies of a BBS gene in each cell have mutations. The parents of an individual with an autosomal recessive condition eac...

**Answer**:

The provided context does not contain information about KBG syndrome or its inheritance pattern. Therefore, it is not possible to determine whether KBG syndrome is inherited based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Scores**: context_relevance=0.833, groundedness=0.344, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000193-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
2. `medquad_ans_0000029-1` (score=0.5035) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. Howeve...
3. `medquad_ans_0000513-2` (score=0.0012) - Imerslund-Grsbeck syndrome is a rare condition that was first described in Finland and Norway; in these regions, the condition is estimated to affect 1 in 200,000 people. The condition has also been reported in other ...
4. `medquad_ans_0000028-4` (score=0.0010) - Nearly all known cases of Aicardi syndrome are sporadic, which means that they are not passed down through generations and occur in people with no history of the disorder in their family. The disorder is believed to r...
5. `medquad_ans_0000028-2` (score=0.0007) - Aicardi syndrome is a very rare disorder. It occurs in about 1 in 105,000 to 167,000 newborns in the United States. Researchers estimate that there are approximately 4,000 affected individuals worldwide.

**Answer**:

Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and physical disability. Additional symptoms may include epilepsy, painful, itchy skin lesions (chilblains), vision problems, and joint stiffness. Symptoms usually progress over several months before the disease course stabilizes. There are six different types of Aicardi-Goutieres syndrome, which are distinguished by the gene that causes the condition: TREX1, RNASEH2A, RNASEH2B, RNASEH2C, SAMHD1, and ADAR genes. Most cases are inherited in an autosomal recessive pattern, although rare autosomal dominant cases have been reported. Treatment is symptomatic and supportive [medquad_ans_0000193-1]. Most newborns do not show

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000193-1`

### Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Scores**: context_relevance=0.750, groundedness=0.725, answer_relevance=0.817

**Top docs**:

1. `medquad_ans_0000003_6-4` (score=0.9965) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
2. `medquad_ans_0000006_6-6` (score=0.9962) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
3. `medquad_ans_0000024_7-8` (score=0.9962) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
4. `medquad_ans_0000035_1-8` (score=0.9962) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
5. `medquad_ans_0000024_5-8` (score=0.9962) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...

**Answer**:

The provided context does not contain specific information about research or clinical trials being conducted for Renal Cell Cancer. Therefore, it is not possible to answer the question based on the given text. [medquad_ans_0000003_6-4][medquad_ans_0000006_6-6]

**Supported citations**: `medquad_ans_0000003_6-4`, `medquad_ans_0000006_6-6`
**Unsupported citations**: _none_

### Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Scores**: context_relevance=0.857, groundedness=0.934, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000191-8` (score=0.9736) - - Primary sclerosing cholangitis (PSC) inflames, scars, and blocks bile ducts inside and outside the liver. - When bile ducts become blocked, bile builds up in the liver and damages liver cells. - PSC can lead to vita...
2. `medquad_ans_0000193-9` (score=0.9567) - - Cirrhosis is scarring of the liver. Scar tissue replaces healthy liver tissue. - Some common causes of cirrhosis include heavy alcohol use, hepatitis infections, and nonalcoholic fatty liver disease. - In the early ...
3. `medquad_ans_0000190-9` (score=0.9146) - - Primary biliary cirrhosis is a chronic disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. - When chronic inflammation damages the bile ducts, bile and toxi...
4. `medquad_ans_0000193-7` (score=0.5855) - To prevent cirrhosis, - see your doctor for treatment of your liver disease. Many of the causes of cirrhosis are treatable. Early treatment may prevent cirrhosis. - try to keep your weight in the normal range. Being o...
5. `medquad_ans_0000191-5` (score=0.1756) - PSC can lead to various complications, including - deficiencies of vitamins A, D, E, and K - infections of the bile ducts - cirrhosisextensive scarring of the liver - liver failure - bile duct cancer

**Answer**:

For Primary Biliary Cirrhosis, a healthcare provider may use the following tests to diagnose the condition: a medical and family history, a physical exam, blood tests, imaging tests, and a liver biopsy. Treatment includes prescribing ursodiol (Actigall, Urso), which, when used early, reduces the likelihood of needing a liver transplant and improves survival. A liver transplant may be considered when cirrhosis leads to liver failure or when treatment for complications is ineffective [medquad_ans_0000190-9].

**Supported citations**: `medquad_ans_0000190-9`
**Unsupported citations**: _none_

### Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Scores**: context_relevance=0.750, groundedness=0.964, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=0.9778) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People ...
2. `medquad_ans_0000243-1__dup1` (score=0.3722) - Summary : When someone's blood flow or breathing stops, seconds count. Permanent brain damage or death can happen quickly. If you know how to perform cardiopulmonary resuscitation (CPR), you could save a life. CPR is ...
3. `medquad_ans_0000379-1__dup1` (score=0.0149) - If you've ever gotten a splinter or had sand in your eye, you've had experience with a foreign body. A foreign body is something that is stuck inside you but isn't supposed to be there. You may inhale or swallow a for...
4. `medquad_ans_0000366-1__dup1` (score=0.0082) - Summary : Accidents happen. Someone chokes on an ice cube or gets stung by a bee. It is important to know when to call 9-1-1 -- it is for life-threatening emergencies. While waiting for help to arrive, you may be able...
5. `medquad_ans_0000033-14` (score=0.0065) - Very common. Each year, more than 1 million people in the U.S. have a heart attack and about half of them die. About one-half of those who die do so within 1 hour of the start of symptoms and before reaching the hospi...

**Answer**:

Drowning occurs when a person gets too much water in their lungs, which can happen even with as little as an inch or two of water. It can happen quickly and silently. Babies can drown in a sink or bathtub, and preschoolers are most likely to drown in a swimming pool. People with seizure disorders are also at risk in the water [medquad_ans_0000299-1__dup2].

**Supported citations**: `medquad_ans_0000299-1__dup2`
**Unsupported citations**: _none_

### Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Scores**: context_relevance=0.857, groundedness=0.981, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0002058-1` (score=0.9604) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragi...
2. `medquad_ans_0002049-1` (score=0.8987) - Ehlers-Danlos syndrome (EDS) is a group of inherited connective tissue disorders that is caused by abnormalities in the structure, production, and/or processing of collagen. There are 6 major forms of EDS: hypermobili...
3. `medquad_ans_0000313-1` (score=0.2940) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and sympt...
4. `medquad_ans_0000954-2` (score=0.0011) - Stickler syndrome affects an estimated 1 in 7,500 to 9,000 newborns. Type I is the most common form of the condition.
5. `medquad_ans_0000565-2` (score=0.0007) - Knobloch syndrome is a rare condition. However, the exact prevalence of the condition is unknown.

**Answer**:

Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder caused by defects in a protein called collagen. It is caused by changes (mutations) in the PLOD1 gene and is inherited in an autosomal recessive manner. Common signs and symptoms include hyperextensible skin that is fragile and bruises easily, joint hypermobility, severe hypotonia at birth, progressive kyphoscoliosis (kyphosis and scoliosis), and fragility of the sclera. Treatment is focused on preventing serious complications and relieving associated signs and symptoms [medquad_ans_0002058-1].

**Supported citations**: `medquad_ans_0002058-1`
**Unsupported citations**: _none_

### Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Scores**: context_relevance=0.875, groundedness=0.907, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000898-2` (score=0.0001) - Sepiapterin reductase deficiency appears to be a rare condition. At least 30 cases have been described in the scientific literature.
2. `medquad_ans_0000287-2` (score=0.0001) - Dihydropyrimidinase deficiency is thought to be a rare disorder. Only a few dozen affected individuals have been described in the medical literature.
3. `medquad_ans_0001860-1` (score=0.0001) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...
4. `medquad_ans_0000668-5` (score=0.0001) - These resources address the diagnosis or management of molybdenum cofactor deficiency: - Genetic Testing Registry: Combined molybdoflavoprotein enzyme deficiency - Genetic Testing Registry: Molybdenum cofactor deficie...
5. `medquad_ans_0001098-2` (score=0.0001) - 5-alpha reductase deficiency is a rare condition; the exact incidence is unknown. Large families with affected members have been found in several countries, including the Dominican Republic, Papua New Guinea, Turkey, ...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include neurological symptoms such as psychomotor delay, low muscle tone (hypotonia), seizures, abnormal movements, too much salivation, and swallowing difficulties. These occur due to decreased levels of neurotransmitters (dopamine, serotonin) and folate in the cerebrospinal fluid resulting from impaired renewal of tetrahydrobiopterin (BH4), which is necessary for the proper processing of phenylalanine and the production of neurotransmitters [medquad_ans_0001860-1].

**Supported citations**: `medquad_ans_0001860-1`
**Unsupported citations**: _none_

### Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Scores**: context_relevance=0.429, groundedness=0.921, answer_relevance=0.200

**Top docs**:

1. `medquad_ans_0000685-2` (score=0.9991) - MONA is rare; its prevalence is unknown. This condition has been reported in multiple populations worldwide.
2. `medquad_ans_0000669-2` (score=0.9987) - The prevalence of monilethrix is unknown.
3. `medquad_ans_0000021-2` (score=0.0290) - Adermatoglyphia appears to be a rare condition. Only a few affected families have been identified worldwide.
4. `medquad_ans_0000914-2` (score=0.0151) - Only 80 to 100 individuals with sitosterolemia have been described in the medical literature. However, researchers believe that this condition is likely underdiagnosed because mild cases often do not come to medical a...
5. `medquad_ans_0000187-3__dup1` (score=0.0040) - Most individuals with Lyme disease respond well to antibiotics and have full recovery. In a small percentage of individuals, symptoms may continue or recur, requiring additional antibiotic treatment. Varying degrees o...

**Answer**:

The prevalence of monilethrix is unknown [medquad_ans_0000669-2].

**Supported citations**: `medquad_ans_0000669-2`
**Unsupported citations**: _none_

### Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Scores**: context_relevance=0.600, groundedness=0.532, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.3190) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...
2. `medquad_ans_0000964-2` (score=0.0045) - Swyer syndrome occurs in approximately 1 in 80,000 people.
3. `medquad_ans_0000582-2` (score=0.0011) - Larsen syndrome occurs in approximately 1 in 100,000 newborns.
4. `medquad_ans_0000631-2` (score=0.0011) - McCune-Albright syndrome occurs in between 1 in 100,000 and 1 in 1,000,000 people worldwide.
5. `medquad_ans_0000201-2` (score=0.0009) - Cole disease is a rare disease; its prevalence is unknown. Only a few affected families have been described in the medical literature.

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria and may be diagnosed based on the results of newborn screening tests. Most people with the condition have no symptoms (asymptomatic), though those who do show symptoms typically experience onset between the ages of 3 and 9, occasionally in adulthood. Mental development is usually normal, though a few cases with intellectual impairment have been reported. Signs and symptoms include skin photosensitivity, neurologic findings, psychiatric symptoms, and ocular (eye) findings. The condition is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. People with Hartnup disease may benefit from a high-protein diet, protection from sunlight, vitamin supplementation, and avoidance of certain drugs/medications. In some cases, treatment with nicotinamide supplements and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Scores**: context_relevance=0.667, groundedness=0.016, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000808-3` (score=0.9987) - Mutations in the GAA gene cause Pompe disease. The GAA gene provides instructions for producing an enzyme called acid alpha-glucosidase (also known as acid maltase). This enzyme is active in lysosomes, which are struc...
2. `medquad_ans_0000424-3` (score=0.0018) - Mutations in the GBE1 gene cause GSD IV. The GBE1 gene provides instructions for making the glycogen branching enzyme. This enzyme is involved in the production of glycogen, which is a major source of stored energy in...
3. `medquad_ans_0000024-3` (score=0.0001) - Mutations in the GBE1 gene cause adult polyglucosan body disease. The GBE1 gene provides instructions for making the glycogen branching enzyme. This enzyme is involved in the production of a complex sugar called glyco...
4. `medquad_ans_0000427-1` (score=0.0001) - Glycogen storage disease type VI (also known as GSDVI or Hers disease) is an inherited disorder caused by an inability to break down a complex sugar called glycogen in liver cells. A lack of glycogen breakdown interfe...
5. `medquad_ans_0002620-1` (score=0.0001) - Glycogen storage disease type 4 (GSD 4) is part of a group of disorders which lead to abnormal accumulation of glycogen (a storage form of glucose) in various parts of the body. Symptoms of GSD 4 usually begin in infa...

**Answer**:

The provided context does not mention glycogen storage disease type VII or its associated genetic changes. Therefore, based on the given information, it is not possible to answer the question. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.731, answer_relevance=0.756

**Top docs**:

1. `medquad_ans_0000977-5` (score=0.9933) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megal...
2. `medquad_ans_0000513-5` (score=0.0084) - These resources address the diagnosis or management of Imerslund-Grsbeck syndrome: - MedlinePlus Encyclopedia: Anemia - B12 deficiency These resources from MedlinePlus offer information about the diagnosis and managem...
3. `medquad_ans_0000124-5` (score=0.0039) - These resources address the diagnosis or management of biotin-thiamine-responsive basal ganglia disease: - Gene Review: Gene Review: Biotin-Thiamine-Responsive Basal Ganglia Disease These resources from MedlinePlus of...
4. `medquad_ans_0000459-5` (score=0.0030) - These resources address the diagnosis or management of hereditary folate malabsorption: - Gene Review: Gene Review: Hereditary Folate Malabsorption - Genetic Testing Registry: Congenital defect of folate absorption - ...
5. `medquad_ans_0000002-8` (score=0.0024) - You might be able to prevent repeat episodes of some types of anemia, especially those caused by lack of iron or vitamins. Dietary changes or supplements can prevent these types of anemia from occurring again. Treatin...

**Answer**:

The provided context does not specify the treatments for thiamine-responsive megaloblastic anemia syndrome. While it mentions resources related to the diagnosis and management of the condition, including a Gene Review and entries in the Genetic Testing Registry and MedlinePlus Encyclopedia, no explicit information about treatments is given [medquad_ans_0000977-5]. Therefore, based on the available context, the treatments cannot be determined.

**Supported citations**: `medquad_ans_0000977-5`
**Unsupported citations**: _none_

### Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000167-18` (score=0.8975) - - Polycystic kidney disease (PKD) is a genetic disorder that causes numerous cysts to grow in the kidneys. - A gene mutation, or defect, causes polycystic kidney disease. - Autosomal dominant PKD is the most common fo...
2. `medquad_ans_0000135-4__dup3` (score=0.0017) - Researchers do not fully understand what causes cysts to grow in the kidneys of people with CKD. The fact that these cysts occur only in the kidneys and not in other parts of the body, as in PKD, indicates that the pr...
3. `medquad_ans_0000135-2__dup2` (score=0.0006) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems wi...
4. `medquad_ans_0000167-5__dup1` (score=0.0006) - Autosomal dominant PKD is the most common form of PKD and the most common inherited disorder of the kidneys.3 The term autosomal dominant means a child can get the disorder by inheriting the gene mutation from only on...
5. `medquad_ans_0000135-1__dup4` (score=0.0003) - Acquired cystic kidney disease happens when a person's kidneys develop fluid-filled sacs, called cysts, over time. Acquired cystic kidney disease is not the same as polycystic kidney disease (PKD), another disease tha...

**Answer**:

Treatment for polycystic kidney disease (PKD) can ease symptoms and prolong life. Although there is no cure currently available, people with PKD may slow the progression of kidney damage caused by high blood pressure through lifestyle changes, diet, and blood pressure medications. People with any kind of kidney disease, including PKD, should talk with a dietitian about foods they should add to their diet and foods that might be harmful [medquad_ans_0000167-18].

**Supported citations**: `medquad_ans_0000167-18`
**Unsupported citations**: _none_

### Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Scores**: context_relevance=0.909, groundedness=0.853, answer_relevance=0.764

**Top docs**:

1. `medquad_ans_0000253-4__dup1` (score=0.9979) - The goals of research on disorders such as Sjgren's syndrome focus on increasing knowledge and understanding of the disorder, improving diagnostic techniques, testing interventions, and finding ways to treat, prevent,...
2. `medquad_ans_0000090-4__dup1` (score=0.9973) - Much of the on-going research on RMDs is aimed at prevention and rehabilitation. The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) funds research on RMDs.
3. `medquad_ans_0000019_3-7` (score=0.9965) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, ta...
4. `medquad_ans_0000207-1__dup2` (score=0.9962) - Summary : Clinical trials are research studies that test how well new medical approaches work in people. Each study answers scientific questions and tries to find better ways to prevent, screen for, diagnose, or treat...
5. `medquad_ans_0000927-1__dup2` (score=0.9960) - Summary : It seems to happen almost every day - you hear about the results of a new medical research study. Sometimes the results of one study seem to disagree with the results of another study. It's important to be c...

**Answer**:

Research on Rheumatoid Arthritis (RMDs) is focused on prevention and rehabilitation [medquad_ans_0000090-4__dup1]. Clinical trials are being conducted to test new treatments, determine their safety and effectiveness, and compare them to existing treatments [medquad_ans_0000207-1__dup2]. These trials aim to find better ways to prevent, diagnose, treat, or manage the disease [medquad_ans_0000207-1__dup2]. Patients may participate in clinical trials at any stage of treatment, including before, during, or after starting treatment, and some trials specifically target patients who have not yet received treatment or whose disease has not improved [medquad_ans_0000019_3-7]. The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) funds such research [medquad

**Supported citations**: `medquad_ans_0000090-4__dup1`, `medquad_ans_0000207-1__dup2`, `medquad_ans_0000019_3-7`
**Unsupported citations**: _none_

### Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.988, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000275-2` (score=0.9786) - There is no cure for Williams syndrome, nor is there a standard course of treatment. Because WS is an uncommon and complex disorder, multidisciplinary clinics have been established at several centers in the United Sta...
2. `medquad_ans_0002561-4` (score=0.9770) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (an...
3. `medquad_ans_0003593-6` (score=0.9518) - How might Liddle syndrome be treated? Treatment for Liddle syndrome includes following a low sodium diet as well as taking potassium-sparing diuretics, which reduce blood pressure and correct hypokalemia and metabolic...
4. `medquad_ans_0000124-2__dup1` (score=0.8982) - There is no cure for Fahr's Syndrome, nor is there a standard course of treatment. Treatment addresses symptoms on an individual basis.
5. `medquad_ans_0000860-2__dup1` (score=0.3046) - How might Brown-Sequard syndrome be treated?

**Answer**:

Surgery is typically the treatment for Glass-Chapman-Hockley syndrome and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (anterior) part of the skull. The operation is usually performed when the person is between 9 to 12 months of age. If other sutures, other than the coronal suture, are involved, other surgeries may be performed [medquad_ans_0002561-4].

**Supported citations**: `medquad_ans_0002561-4`
**Unsupported citations**: _none_

### Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.739, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000868-5` (score=0.9961) - These resources address the diagnosis or management of Renpenning syndrome: - Genetic Testing Registry: Renpenning syndrome 1 - Greenwood Genetics Center: X-Linked Intellectual Disability - Kennedy Krieger Institute: ...
2. `medquad_ans_0006111-6` (score=0.1101) - How might Tietze syndrome be treated? In some individuals, the pain associated with Tietze syndrome resolves on its own without any treatment. Management options for others may include avoidance of strenuous activity;...
3. `medquad_ans_0000124-2__dup1` (score=0.1080) - There is no cure for Fahr's Syndrome, nor is there a standard course of treatment. Treatment addresses symptoms on an individual basis.
4. `medquad_ans_0000091-4__dup1` (score=0.0641) - NINDS supports research on Cushing's syndrome aimed at finding new ways to diagnose, treat, and cure the disorder.
5. `medquad_ans_0000098-7__dup1` (score=0.0568) - Treatment for dumping syndrome includes changes in eating, diet, and nutrition; medication; and, in some cases, surgery. Many people with dumping syndrome have mild symptoms that improve over time with simple dietary ...

**Answer**:

The provided context does not contain information about the treatments for Renpenning syndrome. [medquad_ans_0000868-5] mentions resources that address the diagnosis or management of Renpenning syndrome, such as genetic testing, genetic counseling, and centers for genetic disorders, but it does not specify any treatments. Therefore, based on the available information, the treatments for Renpenning syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000868-5`
**Unsupported citations**: _none_

### Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Scores**: context_relevance=1.000, groundedness=0.988, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001105-1` (score=0.9994) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is ava...
2. `medquad_ans_0001104-2` (score=0.0007) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2F? The subtypes of Charcot-Marie-Tooth type 2, including type 2F, have similar signs and symptoms. Affected individuals usually become symptomatic b...
3. `medquad_ans_0000172-2` (score=0.0001) - Char syndrome is rare, although its exact incidence is unknown. Only a few families with this condition have been identified worldwide.
4. `medquad_ans_0001110-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is ava...
5. `medquad_ans_0001094-1__dup1` (score=0.0001) - Charcot-Marie-Tooth disease type 1E (CMT1E) is a form of Charcot-Marie-Tooth disease, which is a group of rare conditions that affect the peripheral nerves. Signs and symptoms of CMT1E generally become apparent betwee...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 2G include onion bulb formation (7.5%), areflexia, autosomal recessive inheritance, axonal degeneration/regeneration, decreased motor nerve conduction velocity, decreased number of peripheral myelinated nerve fibers, distal amyotrophy, distal muscle weakness, distal sensory impairment, flexion contracture, neonatal onset, pes cavus, spinal deformities, split hand, and vocal cord paresis [medquad_ans_0001105-1].

**Supported citations**: `medquad_ans_0001105-1`
**Unsupported citations**: _none_

### Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000083-5__dup2` (score=0.9992) - In most cases, abdominal adhesions do not cause symptoms. When symptoms are present, chronic abdominal pain is the most common.
2. `medquad_ans_0001393-2` (score=0.9988) - What signs and symptoms are associated with coccygodynia? The classic symptom is pain when pressure is applied to the tailbone, such as when sitting on a hard chair. Symptoms usually improve with relief of pressure wh...
3. `medquad_ans_0000221-6` (score=0.1446) - You may have a kidney stone if you - have pain while urinating - see blood in your urine - feel a sharp pain in your back or lower abdomenthe area between your chest and hips The pain may last for a short or long time...
4. `medquad_ans_0000131-4__dup3` (score=0.0821) - Common symptoms of lactose intolerance include - bloating, a feeling of fullness or swelling, in your belly - pain in your belly - diarrhea - gas - nausea You may feel symptoms 30 minutes to 2 hours after you have mil...
5. `medquad_ans_0000262-3__dup1` (score=0.0199) - There is no cure for these headaches. The disorder is not fatal but can cause considerable discomfort.

**Answer**:

The symptoms of coccygodynia include pain when pressure is applied to the tailbone, such as when sitting on a hard chair, which usually improves with relief of pressure when standing or walking. Other symptoms are immediate and severe pain when moving from sitting to standing, pain during bowel movements, pain during sex, and a deep ache in the region of the tailbone [medquad_ans_0001393-2].

**Supported citations**: `medquad_ans_0001393-2`
**Unsupported citations**: _none_

### Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Scores**: context_relevance=1.000, groundedness=0.948, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0004025-1` (score=0.9994) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgen...
2. `medquad_ans_0000248-2` (score=0.0030) - Craniofacial microsomia has been estimated to occur in between 1 in 5,600 and 1 in 26,550 newborns. However, this range may be an underestimate because not all medical professionals agree on the criteria for diagnosis...
3. `medquad_ans_0000193-3__dup1` (score=0.0020) - Some children with microcephaly will have normal intelligence and a head that will grow bigger, but they may track below the normal growth curves for head circumference. Some children may have only mild disability, wh...
4. `medquad_ans_0000150-3__dup1` (score=0.0002) - The prognosis for individuals with the disorder depends on the severity of the brain and facial deformities.
5. `medquad_ans_0000654-2` (score=0.0001) - Microcephaly-capillary malformation syndrome is rare. About a dozen people have been diagnosed with the disorder.

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, microcephaly, preaxial hand polydactyly, proptosis, and seizures [medquad_ans_0004025-1].

**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

### Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.985, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003402-5` (score=0.9964) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to d...
2. `medquad_ans_0000562-2` (score=0.0001) - Klippel-Feil syndrome is estimated to occur in 1 in 40,000 to 42,000 newborns worldwide. Females seem to be affected slightly more often than males.
3. `medquad_ans_0000563-2` (score=0.0001) - Klippel-Trenaunay syndrome is estimated to affect at least 1 in 100,000 people worldwide.
4. `medquad_ans_0000170-4__dup1` (score=0.0001) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge abuot the brain and nervous system, and to use that knowledge to reduce the burden of neurological di...
5. `medquad_ans_0000170-3__dup1` (score=0.0001) - The prognosis for most individuals with Klippel-Feil Syndrome is good if the disorder is treated early and appropriately. Activities that can injure the neck should be avoided.

**Answer**:

Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal abnormalities, and additional imaging studies may be needed to assess the extent of the abnormality. Other initial exams are needed to detect additional physical abnormalities or underlying conditions, including examination of the chest to rule out involvement of the heart and lungs, examination of the chest wall to detect possible rib anomalies, MRI for spinal stenosis or neurological deficits, ultrasound of the kidneys for renal abnormalities, hearing evaluation due to the high incidence of hearing loss, and various lab tests to assess organ function [medquad_ans_0003402-5].

**Supported citations**: `medquad_ans_0003402-5`
**Unsupported citations**: _none_

### Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.833, groundedness=0.973, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001497-1` (score=0.9537) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severit...
2. `medquad_ans_0000464-1__dup1` (score=0.5711) - A hernia happens when part of an internal organ or tissue bulges through a weak area of muscle. Most hernias are in the abdomen. There are several types of hernias, including - Inguinal, in the groin. This is the the ...
3. `medquad_ans_0000253-2` (score=0.3229) - Heart defects are the most common type of birth defect, accounting for more than 30 percent of all infant deaths due to birth defects. CCHD represents some of the most serious types of heart defects. About 7,200 newbo...
4. `medquad_ans_0000222-1` (score=0.0661) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
5. `medquad_ans_0001012-2` (score=0.0100) - VACTERL association occurs in 1 in 10,000 to 40,000 newborns.

**Answer**:

Congenital diaphragmatic hernia (CDH) is a birth defect characterized by the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severity from a thinned area in the diaphragm to its complete absence. This defect results in an abnormal opening that allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs. This crowding can lead to underdevelopment of the lungs (pulmonary hypoplasia), potentially causing life-threatening breathing difficulties apparent from birth. In 5 to 10 percent of affected individuals, signs and symptoms may appear later in life, including breathing problems or abdominal pain from protrusion of tissue [medquad_ans_0001497-1][medquad_ans_0000464-1__dup1][med

**Supported citations**: `medquad_ans_0001497-1`, `medquad_ans_0000464-1__dup1`
**Unsupported citations**: _none_

### Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Scores**: context_relevance=0.800, groundedness=0.535, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000168-1__dup2` (score=0.7742) - Charcot-Marie-Tooth disease (CMT) is a group of genetic nerve disorders. It is named after the three doctors who first identified it. In the United States, CMT affects about 1 in 2,500 people. CMT affects your periphe...
2. `medquad_ans_0001088-1__dup1` (score=0.0761) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
3. `medquad_ans_0000173-1` (score=0.0431) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
4. `medquad_ans_0001094-1__dup1` (score=0.0020) - Charcot-Marie-Tooth disease type 1E (CMT1E) is a form of Charcot-Marie-Tooth disease, which is a group of rare conditions that affect the peripheral nerves. Signs and symptoms of CMT1E generally become apparent betwee...
5. `medquad_ans_0000173-2` (score=0.0014) - Charcot-Marie-Tooth disease is the most common inherited disorder that involves the peripheral nerves, affecting an estimated 150,000 people in the United States. It occurs in populations worldwide with a prevalence o...

**Answer**:

Charcot-Marie-Tooth disease is a group of genetic nerve disorders that affect the peripheral nerves, which carry movement and sensation signals between the brain and spinal cord and the rest of the body. It is named after the three doctors who first identified it. The disease affects the peripheral nerves that connect the brain and spinal cord to muscles and sensory cells that detect sensations such as touch, pain, heat, and sound. Defects in many different genes cause different forms of this disease, and there are over 40 types. Symptoms usually start in adolescence or early adulthood, though onset can occur from early childhood through late adulthood. Common symptoms include foot drop, foot deformity, loss of lower leg muscle, numbness in the foot or leg, slapping gait, and weakness in the hips, legs, or feet. Symptoms vary in severity, even among family members, and some people may never realize they have the disorder, while others experience moderate to

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Scores**: context_relevance=1.000, groundedness=0.943, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000623-4` (score=0.9982) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene and occur in pe...
2. `medquad_ans_0000167-4` (score=0.9980) - As its name suggests, this condition is inherited in an autosomal recessive pattern. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. The parents of an individual with an auto...
3. `medquad_ans_0000662-4` (score=0.9968) - This condition is inherited in a mitochondrial pattern, which is also known as maternal inheritance. This pattern of inheritance applies to genes contained in mtDNA. Because egg cells, but not sperm cells, contribute ...
4. `medquad_ans_0000669-4` (score=0.9908) - Monilethrix can have multiple patterns of inheritance. When the condition is caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene...
5. `medquad_ans_0004465-4` (score=0.2926) - How is Norum disease inherited? Norum disease is transmitted as an autosomal recessive trait, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive c...

**Answer**:

Yes, monilethrix can be inherited, but the pattern depends on the cause. When caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder. In rare cases, it results from a new mutation and is not inherited. When caused by mutations in the DSG4 gene, it is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations [medquad_ans_0000669-4].

**Supported citations**: `medquad_ans_0000669-4`
**Unsupported citations**: _none_

### Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Scores**: context_relevance=0.750, groundedness=0.903, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000215-14` (score=0.9937) - Children with a UTI should drink as much as they wish and not be forced to drink large amounts of fluid. The health care provider needs to know if a child is not interested in drinking or is unable to drink.
2. `medquad_ans_0000217-5` (score=0.9801) - When a child with VUR gets a UTI, bacteria can move into the kidney and lead to scarring. Scarring of the kidney can be associated with high blood pressure and kidney failure. However, most children with VUR who get a...
3. `medquad_ans_0000223-3__dup2` (score=0.1193) - Any child can get a UTI, though girls get UTIs more often than boys. Children with a condition called vesicoureteral reflux (VUR) are at higher risk for UTIs. VUR causes urine to reflux at the point where one or both ...
4. `medquad_ans_0000169-1__dup3` (score=0.0442) - Pyelonephritis is a type of urinary tract infection (UTI) that affects one or both kidneys.
5. `medquad_ans_0000223-8` (score=0.0410) - Bacteria-fighting medicines called antibiotics are used to treat a UTI. While the lab is doing the urine culture, the health care provider may begin treatment with an antibiotic that treats the bacteria most likely to...

**Answer**:

A urinary tract infection (UTI) in children is a type of infection that affects one or both kidneys, known as pyelonephritis, or can occur in the lower urinary tract [medquad_ans_0000169-1__dup3]. UTIs can occur in any child, though girls are more likely to get them than boys. Children with vesicoureteral reflux (VUR), a condition where urine flows backward from the bladder to the ureters, are at higher risk because bacteria can grow and spread in the urinary tract when urine remains in the system [medquad_ans_0000223-3__dup2]. Boys under 6 months who are not circumcised are also at greater risk than circumcised boys of the same age [medquad_ans_0000223-3__dup2]. UTIs are treated with antibiotics, which may be

**Supported citations**: `medquad_ans_0000169-1__dup3`, `medquad_ans_0000223-3__dup2`
**Unsupported citations**: _none_

### Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.807, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000135-3__dup3` (score=0.9268) - Acquired cystic kidney disease becomes more common the longer a person has CKD. - About 7 to 22 percent of people with CKD already have acquired cystic kidney disease before starting dialysis treatments. - Almost 60 p...
2. `medquad_ans_0000135-6__dup1` (score=0.6068) - People with acquired cystic kidney disease may develop the following complications: - an infected cyst, which can cause fever and back pain. - blood in the urine, which can signal that a cyst in the kidney is bleeding...
3. `medquad_ans_0000135-5__dup2` (score=0.0328) - A person with acquired cystic kidney disease often has no symptoms. However, the complications of acquired cystic kidney disease can have signs and symptoms.
4. `medquad_ans_0000135-4__dup3` (score=0.0066) - Researchers do not fully understand what causes cysts to grow in the kidneys of people with CKD. The fact that these cysts occur only in the kidneys and not in other parts of the body, as in PKD, indicates that the pr...
5. `medquad_ans_0000135-2__dup2` (score=0.0063) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems wi...

**Answer**:

The provided context does not contain information about the complications of Polycystic Kidney Disease (PKD). While it describes differences between PKD and acquired cystic kidney disease, such as PKD being a genetic disorder that can cause high blood pressure and problems with blood vessels in the brain and heart, these details are not explicitly stated in the context provided. Therefore, based on the given information, the complications of PKD cannot be fully answered.

Answer: Insufficient context to determine the complications of Polycystic Kidney Disease. [medquad_ans_0000135-2__dup2]

**Supported citations**: `medquad_ans_0000135-2__dup2`
**Unsupported citations**: _none_

### Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Scores**: context_relevance=0.800, groundedness=0.916, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000649-1__dup1` (score=0.8358) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contami...
2. `medquad_ans_0000100-3__dup3` (score=0.1056) - Anyone can get a foodborne illness. However, some people are more likely to develop foodborne illnesses than others, including - infants and children - pregnant women and their fetuses - older adults - people with wea...
3. `medquad_ans_0000073-3__dup1` (score=0.0054) - COFS is a fatal disease. Most children do not live beyond five years.
4. `medquad_ans_0000103-9` (score=0.0040) - Researchers have not found that eating, diet, and nutrition play a major role in causing or preventing gastritis.
5. `medquad_ans_0000123-1__dup3` (score=0.0038) - Viral gastroenteritis is inflammation of the lining of the stomach, small intestine, and large intestine. Several different viruses can cause viral gastroenteritis, which is highly contagious and extremely common. Vir...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by a group of related viruses. Infection leads to an inflammation of the stomach and intestines, with common symptoms including diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms are fever, headache, or body aches. The infection can spread from person to person, through contaminated food or water, or by touching a contaminated surface. It can be serious, especially in young children and older adults. There is no specific medicine to treat norovirus infections; treatment focuses on bed rest and staying hydrated to prevent dehydration. Prevention measures include proper hand washing and safe food preparation [medquad_ans_0000649-1__dup1].

**Supported citations**: `medquad_ans_0000649-1__dup1`
**Unsupported citations**: _none_

### Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.800, groundedness=0.055, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000253-3` (score=0.9988) - In most cases, the cause of CCHD is unknown. A variety of genetic and environmental factors likely contribute to this complex condition. Changes in single genes have been associated with CCHD. Studies suggest that the...
2. `medquad_ans_0000775-4` (score=0.9988) - This condition is acquired, rather than inherited. It results from new mutations in the PIGA gene, and generally occurs in people with no previous history of the disorder in their family. The condition is not passed d...
3. `medquad_ans_0000098-3__dup2` (score=0.9880) - If your child has patent ductus arteriosus (PDA), you may think you did something wrong during your pregnancy to cause the problem. However, the cause of patent ductus arteriosus isn't known. Genetics may play a role ...
4. `medquad_ans_0000392-3` (score=0.0234) - The cause of Fryns syndrome is unknown. The disorder is thought to be genetic because it tends to run in families and has features similar to those of other genetic disorders. Duplications and deletions in several chr...
5. `medquad_ans_0000222-3` (score=0.0110) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...

**Answer**:

The provided context does not contain any information about the genetic changes related to congenital diaphragmatic hernia. Therefore, it is not possible to answer the question based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.898, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004534-3` (score=0.9950) - What causes Omenn syndrome? Omenn syndrome is a genetically heterogeneous condition (meaning that it may be caused by a number of different genes). While most cases are attributed to mutations in the RAG genes (RAG-1 ...
2. `medquad_ans_0000083-3__dup2` (score=0.0010) - The cause of Kawasaki disease isn't known. The body's response to a virus or infection combined with genetic factors may cause the disease. However, no specific virus or infection has been found, and the role of genet...
3. `medquad_ans_0000851-2` (score=0.0001) - Purine nucleoside phosphorylase deficiency is rare; only about 70 affected individuals have been identified. This disorder accounts for approximately 4 percent of all SCID cases.
4. `medquad_ans_0000018-2` (score=0.0001) - Adenosine deaminase deficiency is very rare and is estimated to occur in approximately 1 in 200,000 to 1,000,000 newborns worldwide. This disorder is responsible for approximately 15 percent of SCID cases.
5. `medquad_ans_0000194-2` (score=0.0001) - Cockayne syndrome occurs in about 2 per million newborns in the United States and Europe.

**Answer**:

Omenn syndrome is a genetically heterogeneous condition that may be caused by mutations in several genes, including RAG1 and RAG2 genes located on chromosome band 11p13. Recent reports have also described cases caused by mutations in ARTEMIS, ADA, ILRA2, ILRA7, CHD7, and DNA ligase 4. Additionally, some cases have been associated with 22q11 microdeletion syndrome [medquad_ans_0004534-3].

**Supported citations**: `medquad_ans_0004534-3`
**Unsupported citations**: _none_

### Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Scores**: context_relevance=0.875, groundedness=0.969, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000097-1` (score=0.9864) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...
2. `medquad_ans_0000032-1__dup1` (score=0.0264) - Ataxia often occurs when parts of the nervous system that control movement are damaged. People with ataxia experience a failure of muscle control in their arms and legs, resulting in a lack of balance and coordination...
3. `medquad_ans_0000032-4__dup1` (score=0.0027) - The NINDS supports and conducts a broad range of basic and clinical research on cerebellar and spinocerebellar degeneration, including work aimed at finding the cause(s) of ataxias and ways to treat, cure, and, ultima...
4. `medquad_ans_0000032-2__dup1` (score=0.0024) - There is no cure for the hereditary ataxias. If the ataxia is caused by another condition, that underlying condition is treated first. For example, ataxia caused by a metabolic disorder may be treated with medications...
5. `medquad_ans_0000080-2` (score=0.0010) - Ataxia-telangiectasia occurs in 1 in 40,000 to 100,000 people worldwide.

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive movement problems due to a loss (atrophy) of nerve cells in the cerebellum, the part of the brain that coordinates movement. Signs and symptoms first appear in early to mid-adulthood and include impaired speech (dysarthria), problems with coordination and balance (ataxia), or both, as well as difficulty with movements that involve judging distance or scale (dysmetria). Other features may include abnormal eye movements (nystagmus) and problems following the movements of objects with the eyes. The movement problems are slowly progressive and may eventually result in the need for a cane, walker, or wheelchair [medquad_ans_0000097-1].

**Supported citations**: `medquad_ans_0000097-1`
**Unsupported citations**: _none_

### Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Scores**: context_relevance=0.250, groundedness=0.000, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000232-3__dup1` (score=0.9933) - With treatment, most people recover from pinched nerve. However, in some cases, the damage is irreversible.
2. `medquad_ans_0000232-2__dup1` (score=0.9932) - The most frequently recommended treatment for pinched nerve is rest for the affected area. Nonsteroidal anti-inflammatory drugs (NSAIDs) or corticosteroids may be recommended to help alleviate pain. Physical therapy i...
3. `medquad_ans_0000222-2__dup1` (score=0.9916) - The appropriate treatment for paresthesia depends on accurate diagnosis of the underlying cause.
4. `medquad_ans_0000257-3__dup1` (score=0.9905) - Recovery depends upon how quickly treatment is received and how severely the body is compromised. Paralysis may persist for many weeks or be permanent. Most individuals have a good chance of recovery.
5. `medquad_ans_0000676-4__dup1` (score=0.9731) - How might Bell's palsy be treated?

**Answer**:

The provided context does not contain information about the treatments for Brachial Plexus Injuries. Therefore, a specific answer cannot be given based on the available information. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Scores**: context_relevance=0.833, groundedness=0.512, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000205-2__dup2` (score=0.9976) - Many events or conditions can damage nerves and nerve pathways. Some of the most common causes are - vaginal childbirth - infections of the brain or spinal cord - diabetes - stroke - accidents that injure the brain or...
2. `medquad_ans_0000218-4__dup2` (score=0.9972) - Urine leakage has many possible causes.
3. `medquad_ans_0000216-8` (score=0.9936) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PU...
4. `medquad_ans_0000216-7` (score=0.0299) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
5. `medquad_ans_0000216-2__dup1` (score=0.0293) - Many types of defects in the urinary tract can cause urine blockage: - Vesicoureteral reflux (VUR). Most children with VUR are born with a ureter that did not grow long enough during development in the womb. The valve...

**Answer**:

Urine blockage in newborns can be caused by various defects in the urinary tract, including vesicoureteral reflux (VUR), ureteropelvic junction (UPJ) obstruction, bladder outlet obstruction (BOO) such as posterior urethral valves (PUV), and ureterocele. Some babies are born with genetic conditions that affect the urinary tract, such as prune belly syndrome (PBS), esophageal atresia (EA), and congenital heart defects. Urine blockage can also result from spina bifida and other birth defects that affect the spinal cord. Defects may be discovered before or after birth, with prenatal tests like ultrasound, amniocentesis, and chorionic villus sampling (CVS) used for detection. Imaging techniques such as ultrasound, voiding cystourethrogram (VCUG), and radionuclide scan can help determine the cause [medquad_ans_0

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Scores**: context_relevance=0.333, groundedness=0.945, answer_relevance=0.333

**Top docs**:

1. `medquad_ans_0000717-2` (score=0.9995) - Nemaline myopathy has an estimated incidence of 1 in 50,000 individuals.
2. `medquad_ans_0000165-2` (score=0.9994) - Centronuclear myopathy is a rare condition; its exact prevalence is unknown.
3. `medquad_ans_0000526-2` (score=0.9993) - Intranuclear rod myopathy is a rare disorder that has been identified in only a small number of individuals. Its exact prevalence is unknown.
4. `medquad_ans_0000011-2` (score=0.9991) - Actin-accumulation myopathy is a rare disorder that has been identified in only a small number of individuals. Its exact prevalence is unknown.
5. `medquad_ans_0000293-2` (score=0.2181) - The prevalence of distal myopathy 2 is unknown. At least two families with the condition have been described in the scientific literature.

**Answer**:

The exact prevalence of intranuclear rod myopathy is unknown. [medquad_ans_0000526-2]

**Supported citations**: `medquad_ans_0000526-2`
**Unsupported citations**: _none_

### Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Scores**: context_relevance=1.000, groundedness=0.896, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000058-3__dup4` (score=0.9849) - Blood pressure is the force of blood flow inside your blood vessels. Blood pressure is written with two numbers separated by a slash. For example, a blood pressure result of 130/80 is said as 130 over 80. The first nu...
2. `medquad_ans_0000036-12` (score=0.9806) - Abnormal blood pressure is higher than 120/80 mmHg. If either your systolic or diastolic blood pressure is higher than normal (120/80) but not high enough to be considered high blood pressure (140/90), you have pre-hy...
3. `medquad_ans_0000036-21` (score=0.9779) - In most cases, the goal is probably to keep your blood pressure below 140/90 mmHg (130/80 if you have diabetes or chronic kidney disease). Normal blood pressure is less than 120/80. Ask your doctor what your blood pre...
4. `medquad_ans_0000036-27` (score=0.9716) - Blood pressure rises as body weight increases. Losing even 10 pounds can lower blood pressure -- and it has the greatest effect for those who are overweight and already have hypertension. If you are overweight or obes...
5. `medquad_ans_0000036-10` (score=0.9690) - Normal blood pressure for adults is defined as a systolic pressure below 120 mmHg and a diastolic pressure below 80 mmHg. It is normal for blood pressures to change when you sleep, wake up, or are excited or nervous. ...

**Answer**:

High blood pressure is defined as a blood pressure reading of 140/90 mmHg or higher. It is also referred to as hypertension and is characterized by elevated pressure in the blood vessels, which forces the heart to work harder and can lead to strain on the heart, damage to blood vessels, and increased risk of heart attack, stroke, eye problems, and kidney problems. The goal for most adults is to keep blood pressure below 140/90 mmHg, and for individuals with diabetes or chronic kidney disease, the target is below 130/80 mmHg [medquad_ans_0000036-12][medquad_ans_0000036-21]. Abnormal blood pressure is higher than 120/80 mmHg, with readings between 120/80 and 140/90

**Supported citations**: `medquad_ans_0000036-12`, `medquad_ans_0000036-21`
**Unsupported citations**: _none_

### Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Scores**: context_relevance=1.000, groundedness=0.254, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000392-4` (score=0.9970) - Fryns syndrome appears to be inherited in an autosomal recessive pattern, which means both copies of a gene in each cell have mutations. However, no associated gene has been identified. The parents of an individual wi...
2. `medquad_ans_0000934-4` (score=0.9941) - This condition is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes. In males (who have on...
3. `medquad_ans_0000859-4` (score=0.9908) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered ATP1A3 gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits a mutation fro...
4. `medquad_ans_0001051-4` (score=0.9817) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
5. `medquad_ans_0000832-4` (score=0.0009) - Most cases of progressive supranuclear palsy are sporadic, which means they occur in people with no history of the disorder in their family. However, some people with this disorder have had family members with related...

**Answer**:

The provided context does not mention "X-linked dystonia-parkinsonism" or specify its inheritance pattern. While several conditions are described as being inherited in X-linked patterns, none are explicitly identified as X-linked dystonia-parkinsonism. Therefore, based on the given information, it is not possible to determine if X-linked dystonia-parkinsonism is inherited.

Answer: Insufficient context to determine if X-linked dystonia-parkinsonism is inherited.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.993, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000702-1__dup1` (score=0.9065) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ket...
2. `medquad_ans_0000120-1` (score=0.8601) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process keto...
3. `medquad_ans_0001093-2` (score=0.0004) - 3-methylglutaconyl-CoA hydratase deficiency is a rare disorder; at least 20 cases have been reported in the scientific literature.
4. `medquad_ans_0000855-2` (score=0.0004) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
5. `medquad_ans_0000959-2` (score=0.0003) - Approximately 350 people with succinic semialdehyde dehydrogenase deficiency have been reported worldwide.

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
4. `medquad_ans_0000006_5-6` (score=0.0401) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis and treatment options depend on: - The type of brain stem glioma. - Where the tumor is found in the brain and if it has spread...
5. `medquad_ans_0000006_7-2` (score=0.0136) - Key Points - Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and s...

**Answer**:

There is no standard staging system for childhood brain stem glioma [medquad_ans_0000006_5-7].

**Supported citations**: `medquad_ans_0000006_5-7`
**Unsupported citations**: _none_

### Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.528, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000266-1__dup2` (score=0.3112) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal c...
2. `medquad_ans_0000019-4__dup1` (score=0.0046) - Research supported by the NINDS includes studies to understand how the brain and nervous system normally develop. These studies contribute to a greater understanding of neural tube disorders, such as anencephaly, and ...
3. `medquad_ans_0000054-4__dup1` (score=0.0036) - The NINDS supports and conducts a wide range of research on spinal cord disorders such as BSS. The goal of this research is to find ways to prevent, treat, and, ultimately, cure these disorders.
4. `medquad_ans_0005742-1` (score=0.0033) - Spina bifida occulta (SBO) occurs when the bones of the spinal column do not completely close around the developing nerves of the spinal cord. In most cases SBO causes no symptoms, however cases associated with back a...
5. `medquad_ans_0000117-3__dup2` (score=0.0019) - The prognosis for individuals with encephaloceles varies depending on the type of brain tissue involved, the location of the sacs, and the accompanying brain malformations.

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal cord (conus medullaris) or develop near the site of a spinal cord injury. These attachments cause an abnormal stretching of the spinal cord, leading to a progressive course. In children, symptoms may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is believed to result from improper growth of the neural tube during fetal development and is closely linked to spina bifida. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. This delayed presentation is related to the degree of strain on the spinal cord over time

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.892, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000222-3__dup1` (score=0.9983) - The prognosis for those with paresthesia depends on the severity of the sensations and the associated disorders.
2. `medquad_ans_0000058-3__dup1` (score=0.9958) - In general, carpal tunnel syndrome responds well to treatment, but less than half of individuals report their hand(s) feeling completely normal following surgery. Some residual numbness or weakness is common. At work,...
3. `medquad_ans_0000001-7__dup1` (score=0.9100) - Topics
4. `medquad_ans_0000156-2` (score=0.0060) - Carpenter syndrome is thought to be a rare condition; approximately 70 cases have been described in the scientific literature.
5. `medquad_ans_0000079-3__dup1` (score=0.0050) - Many people with chronic pain can be helped if they understand all the causes of pain and the many and varied steps that can be taken to undo what chronic pain has done. Scientists believe that advances in neuroscienc...

**Answer**:

The outlook for carpal tunnel syndrome is generally positive, as it responds well to treatment. However, less than half of individuals report their hand(s) feeling completely normal following surgery, and some residual numbness or weakness is common [medquad_ans_0000058-3__dup1]. Non-surgical measures such as stretching exercises, frequent rest breaks, wearing splints to keep wrists straight, and maintaining correct posture and wrist position can help prevent or reduce symptoms [medquad_ans_0000058-3__dup1]. Wearing fingerless gloves can also help keep hands warm and flexible [medquad_ans_0000058-3__dup1].

**Supported citations**: `medquad_ans_0000058-3__dup1`
**Unsupported citations**: _none_

### Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.743, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000296-5` (score=0.9939) - These resources address the diagnosis or management of Donnai-Barrow syndrome: - Gene Review: Gene Review: Donnai-Barrow Syndrome - Genetic Testing Registry: Donnai Barrow syndrome - MedlinePlus Encyclopedia: Diaphrag...
2. `medquad_ans_0000860-2__dup1` (score=0.9415) - How might Brown-Sequard syndrome be treated?
3. `medquad_ans_0000124-2__dup1` (score=0.3999) - There is no cure for Fahr's Syndrome, nor is there a standard course of treatment. Treatment addresses symptoms on an individual basis.
4. `medquad_ans_0000130-2__dup1` (score=0.0273) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportiv...
5. `medquad_ans_0000130-4__dup1` (score=0.0180) - The NINDS supports research aimed at discovering new ways to diagnose, treat, and, ultimately, cure neuropathies such as Miller Fisher syndrome.

**Answer**:

The provided context does not contain specific information about the treatments for Donnai-Barrow syndrome. While it lists resources that address the diagnosis or management of Donnai-Barrow syndrome, such as genetic testing, hearing evaluation, and surgical interventions for conditions like diaphragmatic hernia and omphalocele, it does not detail the specific treatments for Donnai-Barrow syndrome itself [medquad_ans_0000296-5]. Therefore, based on the available information, the treatments for Donnai-Barrow syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000296-5`
**Unsupported citations**: _none_

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

**Scores**: context_relevance=0.917, groundedness=0.986, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0001900-1` (score=0.9606) - Dominant optic atrophy (DOA) is an inherited optic nerve disorder characterized by degeneration of the optic nerves. It typically starts during the first decade of life. Affected people usually develop moderate visual...
2. `medquad_ans_0000585-1__dup1` (score=0.9521) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
3. `medquad_ans_0006302-1` (score=0.1227) - Usher syndrome is a genetic condition characterized by hearing loss or deafness, and progressive vision loss due to retinitis pigmentosa. Three major types of Usher syndrome have been described - types I, II, and III....
4. `medquad_ans_0006301-1` (score=0.1227) - Usher syndrome is a genetic condition characterized by hearing loss or deafness, and progressive vision loss due to retinitis pigmentosa. Three major types of Usher syndrome have been described - types I, II, and III....
5. `medquad_ans_0006304-1` (score=0.1227) - Usher syndrome is a genetic condition characterized by hearing loss or deafness, and progressive vision loss due to retinitis pigmentosa. Three major types of Usher syndrome have been described - types I, II, and III....

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. It is suggested to be inherited in an autosomal recessive manner [medquad_ans_0000585-1__dup1].

**Supported citations**: `medquad_ans_0000585-1__dup1`
**Unsupported citations**: _none_

### Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Scores**: context_relevance=0.700, groundedness=0.882, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0000794-3` (score=0.9991) - Phosphoglycerate kinase deficiency is caused by mutations in the PGK1 gene. This gene provides instructions for making an enzyme called phosphoglycerate kinase, which is involved in a critical energy-producing process...
2. `medquad_ans_0000795-3` (score=0.9983) - Phosphoglycerate mutase deficiency is caused by mutations in the PGAM2 gene. This gene provides instructions for making an enzyme called phosphoglycerate mutase, which is involved in a critical energy-producing proces...
3. `medquad_ans_0000770-3` (score=0.0010) - Mutations in the PANK2 gene cause pantothenate kinase-associated neurodegeneration. The PANK2 gene provides instructions for making an enzyme called pantothenate kinase 2. This enzyme is active in mitochondria, the en...
4. `medquad_ans_0000652-2` (score=0.0001) - More than 200 people with mevalonate kinase deficiency have been reported worldwide; the majority of these individuals have HIDS.
5. `medquad_ans_0000795-4` (score=0.0001) - This condition is inherited in an autosomal recessive pattern, which means both copies of the PGAM2 gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one c...

**Answer**:

The genetic changes related to phosphoglycerate kinase deficiency are mutations in the PGK1 gene [medquad_ans_0000794-3].

**Supported citations**: `medquad_ans_0000794-3`
**Unsupported citations**: _none_

### Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Scores**: context_relevance=0.700, groundedness=0.959, answer_relevance=0.620

**Top docs**:

1. `medquad_ans_0000850-3` (score=0.9993) - The primary genetic cause of PVOD is mutations in the EIF2AK4 gene. Mutations in other genes may cause a small percentage of cases. Other suspected causes of PVOD include viral infection and exposure to toxic chemical...
2. `medquad_ans_0000382-2` (score=0.9985) - An estimated 1 in 200 females has the genetic change that leads to FXPOI, although only about a quarter of them develop the condition. FXPOI accounts for about 4 to 6 percent of all cases of primary ovarian insufficie...
3. `medquad_ans_0000823-4` (score=0.7750) - This condition is generally not inherited but arises from gene mutations that occur in early blood-forming cells after conception. These alterations are called somatic mutations.
4. `medquad_ans_0000850-4` (score=0.0050) - When caused by mutations in the EIF2AK4 gene, PVOD is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recess...
5. `medquad_ans_0000849-3` (score=0.0008) - Mutations in the BMPR2 gene are the most common genetic cause of pulmonary arterial hypertension. This gene plays a role in regulating the number of cells in certain tissues. Researchers suggest that a mutation in thi...

**Answer**:

The primary genetic cause of pulmonary veno-occlusive disease (PVOD) is mutations in the EIF2AK4 gene. Mutations in other genes may cause a small percentage of cases. The EIF2AK4 gene mutations involved in PVOD likely eliminate functional EIF2AK4 protein, although it is unknown how the absence of this protein's function leads to the pulmonary vessel abnormalities that underlie PVOD. When caused by mutations in the EIF2AK4 gene, PVOD is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations [medquad_ans_0000850-3][medquad_ans_0000850-4].

**Supported citations**: `medquad_ans_0000850-3`, `medquad_ans_0000850-4`
**Unsupported citations**: _none_

### Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Scores**: context_relevance=0.750, groundedness=0.952, answer_relevance=0.450

**Top docs**:

1. `medquad_ans_0001006-3` (score=0.9991) - Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. In the liver, enzymes break down tyrosine in a five step process, resulting in molecules that are either excreted by t...
2. `medquad_ans_0000033-3__dup1` (score=0.0159) - Average lifespan has been improving for years, for unknown reasons, and varies with the severity of the underlying mutations, ATM (ataxia-telangiectasia mutated) protein levels, and residual ATM kinase activity. Some ...
3. `medquad_ans_0000703-3` (score=0.0105) - Mutations in the MT-TK gene are the most common cause of MERRF, occurring in more than 80 percent of all cases. Less frequently, mutations in the MT-TL1, MT-TH, and MT-TS1 genes have been reported to cause the signs a...
4. `medquad_ans_0001005-3` (score=0.0021) - Mutations in the TH gene cause TH deficiency. The TH gene provides instructions for making the enzyme tyrosine hydroxylase, which is important for normal functioning of the nervous system. Tyrosine hydroxylase takes p...
5. `medquad_ans_0000396-1__dup2` (score=0.0002) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...

**Answer**:

Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. These mutations result in decreased activity of specific enzymes involved in the breakdown of tyrosine: the FAH gene provides instructions for the fumarylacetoacetate hydrolase enzyme (final step), the TAT gene encodes tyrosine aminotransferase enzyme (first step), and the HPD gene provides instructions for the 4-hydroxyphenylpyruvate dioxygenase enzyme (second step). As a result, tyrosine and its byproducts accumulate to toxic levels, causing cellular damage in the liver, kidneys, nervous system, and other organs [medquad_ans_0001006-3].

**Supported citations**: `medquad_ans_0001006-3`
**Unsupported citations**: _none_

### Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Scores**: context_relevance=0.857, groundedness=0.921, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000209-2` (score=0.9953) - Treatment involves restoring normal levels of thyroid hormone and may include thyroid drugs, radioactive iodine, and sometimes partial or complete surgical removal of the thyroid.
2. `medquad_ans_0000436-5` (score=0.9887) - These resources address the diagnosis or management of Graves disease: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Graves disease 2 - Genetic Testing Registry: Graves disease 3 -...
3. `medquad_ans_0000005-6__dup2` (score=0.9680) - Treatment generally depends on whether the thyroid is damaged enough to cause hypothyroidism. In the absence of hypothyroidism, some health care providers treat Hashimotos disease to reduce the size of the goiter. Oth...
4. `medquad_ans_0000137-2__dup2` (score=0.9501) - There is no cure for GSS, nor are there any known treatments to slow progression of the disease. Current therapies are aimed at alleviating symptoms and making the patient as comfortable as possible.
5. `medquad_ans_0000008-8__dup2` (score=0.9435) - Health care providers treat hyperthyroidism with medications, radioiodine therapy, or thyroid surgery. The aim of treatment is to bring thyroid hormone levels to a normal state, thus preventing long-term complications...

**Answer**:

Treatments for Graves' disease include thyroid drugs, radioactive iodine, and sometimes partial or complete surgical removal of the thyroid [medquad_ans_0000209-2]. Additionally, resources such as the Graves' Disease & Thyroid Foundation provide information on treatment options, and management may involve drug therapy, surgery, and genetic counseling [medquad_ans_0000436-5]. The goal of treatment is to restore normal levels of thyroid hormone and relieve uncomfortable symptoms [medquad_ans_0000008-8__dup2].

**Supported citations**: `medquad_ans_0000209-2`, `medquad_ans_0000436-5`, `medquad_ans_0000008-8__dup2`
**Unsupported citations**: _none_

### Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Scores**: context_relevance=0.800, groundedness=0.319, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000039-1__dup2` (score=0.2434) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, ...
2. `medquad_ans_0000038-3__dup1` (score=0.0061) - Early and accurate diagnosis is key to prolonged survival for boys born with Barth syndrome. The disorder was once considered uniformly fatal in infancy, but some individuals are now living much longer. Severe infecti...
3. `medquad_ans_0000039-3__dup1` (score=0.0037) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten dise...
4. `medquad_ans_0000969-2` (score=0.0019) - Tangier disease is a rare disorder with approximately 100 cases identified worldwide. More cases are likely undiagnosed. This condition is named after an island off the coast of Virginia where the first affected indiv...
5. `medquad_ans_0000519-2` (score=0.0015) - The incidence of infantile NCL is unknown. Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, where approximately 1 in 12,500 individuals are affected.

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. Early signs may include personality and behavior changes, slow learning, clumsiness, or stumbling. Symptoms are linked to a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. Vision loss is often an early sign, leading to suspicion during an eye exam. Diagnostic tests include blood or urine tests, skin or tissue sampling, electroencephalogram (EEG), electrical studies of the eyes, and brain scans. Over time, affected children suffer cognitive impairment, worsening seizures, progressive loss of sight and motor skills, and eventually become blind, bedridden, and demented. Batten disease is often fatal by the late teens or twenties [medquad_ans_0000039-1__dup2, medquad_ans_0000039-3__dup1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000039-1__dup2, medquad_ans_0000039-3__dup1`

### Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Scores**: context_relevance=0.857, groundedness=0.986, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000114-3__dup2` (score=0.9753) - ESS is not a life-threatening condition. Most often, and particularly among those with primary ESS, the disorder does not cause health problems and does not affect life expectancy.
2. `medquad_ans_0002061-1` (score=0.5247) - Ehlers-Danlos syndrome progeroid type is a genetic disorder of the connective tissue, which is the material between the cells of the body that gives tissues form and strength. The disorder primarily affects the skin, ...
3. `medquad_ans_0000634-2` (score=0.0003) - Meckel syndrome affects 1 in 13,250 to 1 in 140,000 people worldwide. It is more common in certain populations; for example, the condition affects about 1 in 9,000 people of Finnish ancestry and about 1 in 3,000 peopl...
4. `medquad_ans_0000491-4` (score=0.0001) - Hutchinson-Gilford progeria syndrome is considered an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. The condition results from new mutations i...
5. `medquad_ans_0000401-2` (score=0.0001) - Geleophysic dysplasia is a rare disorder whose prevalence is unknown. More than 30 affected individuals have been reported.

**Answer**:

Ehlers-Danlos syndrome, progeroid type is a genetic disorder of the connective tissue that primarily affects the skin, hair, and skeletal system. Symptoms usually appear by childhood or adolescence and include unusually flexible joints, loose elastic skin, and easy scarring. Unique features of this type include sparse scalp hair and eyebrows, and loose elastic skin on the face, which cause affected individuals to look older than their age. Additional symptoms may include bone weakness, weak muscle tone, mild intellectual disability, and delayed growth in affected children. The progeroid type is caused by mutations in the B4GALT7 gene and is inherited in an autosomal recessive pattern [medquad_ans_0002061-1].

**Supported citations**: `medquad_ans_0002061-1`
**Unsupported citations**: _none_

### Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=0.900, groundedness=0.980, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000416-3` (score=0.9991) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
2. `medquad_ans_0000419-3` (score=0.1369) - Mutations in the GSS gene cause glutathione synthetase deficiency. The GSS gene provides instructions for making an enzyme called glutathione synthetase. This enzyme is involved in a process called the gamma-glutamyl ...
3. `medquad_ans_0000420-3` (score=0.0024) - Mutations in the AMT and GLDC genes cause glycine encephalopathy. About 80 percent of cases of glycine encephalopathy result from mutations in the GLDC gene, while AMT mutations cause 10 percent to 15 percent of all c...
4. `medquad_ans_0000066-3` (score=0.0018) - Mutations in the GATM gene cause arginine:glycine amidinotransferase deficiency. The GATM gene provides instructions for making the enzyme arginine:glycine amidinotransferase. This enzyme participates in the two-step ...
5. `medquad_ans_0000396-1__dup2` (score=0.0005) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the breakdown of the amino acid histidine and in the production of folate [medquad_ans_0000416-3].

**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

### Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Scores**: context_relevance=0.778, groundedness=0.917, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0001046-2` (score=0.0003) - XLA occurs in approximately 1 in 200,000 newborns.
2. `medquad_ans_0000581-2` (score=0.0003) - Laron syndrome is a rare disorder. About 350 people have been diagnosed with the condition worldwide. The largest single group of affected individuals (about 100 people) lives in an area of southern Ecuador.
3. `medquad_ans_0000757-2` (score=0.0002) - This condition affects an estimated 6 to 7 per 100,000 people worldwide. Types I and IV are the most common forms of osteogenesis imperfecta, affecting 4 to 5 per 100,000 people.
4. `medquad_ans_0001058-2` (score=0.0002) - The incidence of XLAG is unknown; approximately 30 affected families have been described in the medical literature.
5. `medquad_ans_0001169-1` (score=0.0001) - Chondrodysplasia punctata 1, X-linked recessive (CDPX1) is a genetic disorder present from birth that affects bone and cartilage development. On x-ray, affected infants have characteristic spots at the ends of their b...

**Answer**:

Symptoms of Chondrodysplasia punctata 1, X-linked recessive (CDPX1) include characteristic spots at the ends of bones known as chondrodysplasia punctata or stippled epiphyses, which typically disappear between ages 2 and 3; shortened fingers; a flat nose; breathing abnormalities; hearing loss; abnormalities of the spinal bones in the neck; and delayed intellectual development [medquad_ans_0001169-1].

**Supported citations**: `medquad_ans_0001169-1`
**Unsupported citations**: _none_

### Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Scores**: context_relevance=0.714, groundedness=0.980, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000132-10` (score=0.9879) - A person with Whipple disease and malabsorption may need - a diet high in calories and protein - vitamins - nutritional supplements
2. `medquad_ans_0000165-2__dup1` (score=0.9735) - The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfa...
3. `medquad_ans_0000098-7__dup1` (score=0.1366) - Treatment for dumping syndrome includes changes in eating, diet, and nutrition; medication; and, in some cases, surgery. Many people with dumping syndrome have mild symptoms that improve over time with simple dietary ...
4. `medquad_ans_0000107-5__dup2` (score=0.0375) - Treatment of gastroparesis depends on the severity of the persons symptoms. In most cases, treatment does not cure gastroparesis, which is usually a chronic, or long-lasting, condition. Gastroparesis is also a relapsi...
5. `medquad_ans_0000132-6` (score=0.0375) - People with Whipple disease may have complications caused by malnutrition, which is due to damaged villi in the small intestine. As a result of delayed diagnosis or treatment, people may experience the following compl...

**Answer**:

The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfadizine or solfamethoxazole can treat neurological symptoms. Relapsing neurologic Whipple's disease, marked by bouts of worsening of symptoms, is sometimes treated with a combination of antibiotics and weekly injections of interferon gamma, a substance made by the body that activates the immune system [medquad_ans_0000165-2__dup1].

**Supported citations**: `medquad_ans_0000165-2__dup1`
**Unsupported citations**: _none_

### Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Scores**: context_relevance=0.714, groundedness=0.936, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000232-3__dup1` (score=0.9922) - With treatment, most people recover from pinched nerve. However, in some cases, the damage is irreversible.
2. `medquad_ans_0000050-2__dup1` (score=0.9908) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
3. `medquad_ans_0000049-2__dup1` (score=0.9908) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
4. `medquad_ans_0000222-2__dup1` (score=0.9880) - The appropriate treatment for paresthesia depends on accurate diagnosis of the underlying cause.
5. `medquad_ans_0000042-2__dup2` (score=0.9856) - Steroids such as prednisone -- used to reduce inflammation and swelling -- are an effective treatment for Bell's palsy. Antiviral drugs may have some benefit in shortening the course of the disease. Analgesics such as...

**Answer**:

Treatments for Bell's palsy include steroids such as prednisone, which reduce inflammation and swelling, antiviral drugs that may help shorten the course of the disease, and analgesics such as aspirin, acetaminophen, or ibuprofen to relieve pain. Individuals should consult their doctors before taking over-the-counter medicines due to possible drug interactions. Additionally, keeping the eye moist and protecting it from debris and injury, especially at night, is important, and lubricating eye drops can help. Other therapies such as physical therapy, facial massage, or acupuncture may provide a potential small improvement in facial nerve function and pain [medquad_ans_0000042-2__dup2].

**Supported citations**: `medquad_ans_0000042-2__dup2`
**Unsupported citations**: _none_
