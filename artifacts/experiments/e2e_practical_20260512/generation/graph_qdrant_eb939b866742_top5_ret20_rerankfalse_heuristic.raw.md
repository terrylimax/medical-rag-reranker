# Baseline Generation Examples (No Reranker)

- retriever: `graph_bm25`
- llm_model: `qwen/qwen3-4b-instruct-2507`
- top_k: `5`
- reranker_enabled: `False`
- num_examples: `300`

## Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Top docs**:

1. `medquad_ans_0006510-1` (score=0.7000) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the information is available, the ...
2. `medquad_ans_0006511-1` (score=0.6694) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the information is available, the ...
3. `medquad_ans_0006509-2` (score=0.6389) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is available, the tabl...
4. `medquad_ans_0002834-2` (score=0.2271) - What are the signs and symptoms of Hereditary lymphedema type II? Hereditary lymphedema type II is characterized by the abnormal transport of lymph fluid. This causes the lymph fluid to build up, causing swelling (lymphedema). Individual...
5. `medquad_ans_0002781-2` (score=0.1565) - What are the signs and symptoms of Hemophagocytic lymphohistiocytosis? The signs and symptoms of hemophagocytic lymphohistiocytosis typically develop during the first months or years of life. However, in rare cases, affected people may n...

**Answer**:

The symptoms of X-linked lymphoproliferative syndrome 1 include cellular immunodeficiency (90%), decreased antibody level in blood (50%), hepatomegaly (50%), lymphadenopathy (50%), lymphoma (50%), splenomegaly (50%), anemia (7.5%), encephalitis, fulminant hepatitis, hepatic encephalopathy, IgG deficiency, immunodeficiency, increased IgM level, meningitis, pancytopenia, recurrent pharyngitis, reduced natural killer cell activity, thrombocytopenia, and X-linked recessive inheritance [medquad_ans_0006510-1].

**Citations detected**: `medquad_ans_0006510-1`
**Supported citations**: `medquad_ans_0006510-1`
**Unsupported citations**: _none_

## Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Top docs**:

1. `medquad_ans_0000307-1__dup1` (score=0.7000) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a group of muscles or ...
2. `medquad_ans_0004750-4` (score=0.5012) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is passed down through fami...
3. `medquad_ans_0003601-2` (score=0.3392) - How is limb-girdle muscular dystrophy inherited? Limb-girdle muscular dystrophy (LGMD) is most often inherited in an autosomal recessive manner; less commonly, rare sub-types may be inherited in an autosomal dominant manner. There may be...
4. `medquad_ans_0000139-2__dup2` (score=0.2566) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the gene on to you. You...
5. `medquad_ans_0000673-4` (score=0.2032) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.

**Answer**:

The provided context does not specify whether early-onset primary dystonia is inherited. While it states that "some people inherit dystonia," it does not differentiate between early-onset primary dystonia and other forms or specify the inheritance pattern for early-onset cases [medquad_ans_0000307-1__dup1]. Therefore, based on the available information, the inheritance of early-onset primary dystonia cannot be determined.

**Citations detected**: `medquad_ans_0000307-1__dup1`
**Supported citations**: `medquad_ans_0000307-1__dup1`
**Unsupported citations**: _none_

## Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Top docs**:

1. `medquad_ans_0000397-1__dup2` (score=0.7000) - Taeniasis in humans is a parasitic infection caused by the tapeworm species Taenia saginata (beef tapeworm), Taenia solium (pork tapeworm), and Taenia asiatica (Asian tapeworm). Humans can become infected with these tapeworms by eating r...
2. `medquad_ans_0000397-2__dup1` (score=0.5975) - The tapeworms that cause taeniasis (Taenia saginata, T. solium, and T. asiatica) are found worldwide. Eating raw or undercooked beef or pork is the primary risk factor for acquiring taeniasis. Persons who don't eat raw or undercooked bee...
3. `medquad_ans_0000689-1__dup2` (score=0.3652) - Parasites are living things that use other living things - like your body - for food and a place to live. You can get them from contaminated food or water, a bug bite, or sexual contact. Some parasitic diseases are easily treated and som...
4. `medquad_ans_0000366-1__dup2` (score=0.3001) - Schistosomiasis, also known as bilharzia, is a disease caused by parasitic worms. Infection with Schistosoma mansoni, S. haematobium, and S. japonicum causes illness in humans; less commonly, S. mekongi and S. intercalatum can cause dise...
5. `medquad_ans_0000121-2__dup3` (score=0.2935) - Cysticercosis is an infection caused by the larvae of the parasite Taenia solium. This infection occurs after a person swallows tapeworm eggs. The larvae get into tissues such as muscle and brain, and form cysts there (these are called c...

**Answer**:

People at risk for taeniasis include those who eat raw or undercooked beef (particularly infected with Taenia saginata) or pork (with Taenia solium or Taenia asiatica). Individuals in under-developed communities with poor sanitation, especially in Latin America, Eastern Europe, sub-Saharan Africa, India, and Asia, are at higher risk. Additionally, people in the United States, particularly Latin American immigrants, may be at risk due to consumption of raw or undercooked pork or exposure to contaminated environments. Those with poor hygiene who shed tapeworm eggs in feces may also transmit the infection to themselves or others, increasing the risk of cysticercosis [medquad_ans_0000397-1\_\_dup2][medquad_ans_0000397-2__dup1].

**Citations detected**: `medquad_ans_0000397-1__dup2`, `medquad_ans_0000397-2__dup1`
**Supported citations**: `medquad_ans_0000397-1__dup2`, `medquad_ans_0000397-2__dup1`
**Unsupported citations**: _none_

## Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Top docs**:

1. `medquad_ans_0000194-2__dup1` (score=0.7000) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is available, the table below ...
2. `medquad_ans_0000196-1__dup1` (score=0.5513) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and phy...
3. `medquad_ans_0000197-1__dup1` (score=0.5513) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and phy...
4. `medquad_ans_0000029-2` (score=0.5094) - Aicardi-Goutieres syndrome is a rare disorder. Its exact prevalence is unknown.
5. `medquad_ans_0000193-2__dup1` (score=0.4552) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 1. If the information is available, the table below ...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy [medquad_ans_0000194-2__dup1]. These symptoms are based on the Human Phenotype Ontology, which compiles data on the frequency of signs and symptoms in this condition [medquad_ans_0000194-2__dup1].

**Citations detected**: `medquad_ans_0000194-2__dup1`
**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

## Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Top docs**:

1. `medquad_ans_0003421-1` (score=0.7000) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how often the symptom i...
2. `medquad_ans_0005502-2` (score=0.2976) - What are the signs and symptoms of Schnitzler syndrome? The signs and symptoms of Schnitzler syndrome vary but may include: Red raised patches of skin (urticaria) that may become itchy Recurrent fevers Join pain and inflammation Organome...
3. `medquad_ans_0003339-2` (score=0.2143) - What are the signs and symptoms of Kallmann syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kallmann syndrome. If the information is available, the table below includes how often the symptom i...
4. `medquad_ans_0003423-1` (score=0.1931) - What are the signs and symptoms of Kozlowski Celermajer Tink syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kozlowski Celermajer Tink syndrome. If the information is available, the table belo...
5. `medquad_ans_0003418-1` (score=0.1689) - What are the signs and symptoms of Kosztolanyi syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kosztolanyi syndrome. If the information is available, the table below includes how often the sym...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Citations detected**: `medquad_ans_0003421-1`
**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

## Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Top docs**:

1. `medquad_ans_0000069-1__dup3` (score=0.7000) - Espaol Hemophilia (heem-o-FILL-ee-ah) is a rare bleeding disorder in which the blood doesn't clot normally. If you have hemophilia, you may bleed for a longer time than others after an injury. You also may bleed inside your body (interna...
2. `medquad_ans_0000069-3__dup1` (score=0.5575) - A defect in one of the genes that determines how the body makes blood clotting factor VIII or IX causes hemophilia. These genes are located on the X chromosomes (KRO-muh-somz). Chromosomes come in pairs. Females have two X chromosomes, w...
3. `medquad_ans_0000455-1__dup1` (score=0.5496) - Hemophilia is a rare disorder in which the blood does not clot normally. It is usually inherited. Hemophilia usually occurs in males. If you have hemophilia, you have little or no clotting factor. Clotting factor is a protein needed for ...
4. `medquad_ans_0000067-8` (score=0.2069) - You can't prevent primary, or inherited, hemochromatosis. However, not everyone who inherits hemochromatosis genes develops symptoms or complications of the disease. In those who do, treatments can keep the disease from getting worse. Tr...
5. `medquad_ans_0000300-4` (score=0.0000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...

**Answer**:

Yes, hemophilia is inherited. It is usually inherited through genes located on the X chromosomes, and individuals are born with the disorder due to a defect in the genes that determine how the body makes blood clotting factor VIII or IX [medquad_ans_0000069-1__dup3]. Males are more commonly affected because they have only one X chromosome, so if that X chromosome carries the hemophilia gene, they will have the disorder. Females with the gene on one X chromosome are typically carriers and can pass the gene to their children [medquad_ans_0000069-1__dup3].

**Citations detected**: `medquad_ans_0000069-1__dup3`
**Supported citations**: `medquad_ans_0000069-1__dup3`
**Unsupported citations**: _none_

## Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.7000) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often the symptom is seen i...
2. `medquad_ans_0003103-2` (score=0.4501) - What are the signs and symptoms of IBIDS syndrome? The most common symptoms of Tay syndrome are brittle hair (trichothiodystrophy); dry, thickened, scaling skin (ichthyosis); photosensitivity (abnormal light sensitivity); abnormal nails;...
3. `medquad_ans_0004163-2` (score=0.3973) - What are the signs and symptoms of Mucopolysaccharidosis type IVA? The Human Phenotype Ontology provides the following list of signs and symptoms for Mucopolysaccharidosis type IVA. If the information is available, the table below includ...
4. `medquad_ans_0004162-1` (score=0.2699) - What are the signs and symptoms of Mucopolysaccharidosis type IV? The Human Phenotype Ontology provides the following list of signs and symptoms for Mucopolysaccharidosis type IV. If the information is available, the table below includes...
5. `medquad_ans_0004163-1` (score=0.2354) - Mucopolysaccharidosis type IVA (MPS IVA, also called Morquio syndrome, type A) is a metabolic condition that primarily affects the skeleton. The severity, age of onset, and associated symptoms vary significantly from person to person and...

**Answer**:

The symptoms of IVIC syndrome include hearing impairment (90%), limitation of joint mobility (90%), short stature (90%), strabismus (90%), abnormal dermatoglyphics (50%), aplasia/hypoplasia of the thumb (50%), radioulnar synostosis (50%), scoliosis (50%), synostosis of carpal bones (50%), triphalangeal thumb (50%), abnormality of the clavicle (7.5%), arrhythmia (7.5%), leukocytosis (7.5%), preaxial hand polydactyly (7.5%), thrombocytopenia (7.5%), and urogenital fistula (7.5%) [medquad_ans_0003284-1].

**Citations detected**: `medquad_ans_0003284-1`
**Supported citations**: `medquad_ans_0003284-1`
**Unsupported citations**: _none_

## Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Top docs**:

1. `medquad_ans_0000775-2` (score=0.7000) - Paroxysmal nocturnal hemoglobinuria is a rare disorder, estimated to affect between 1 and 5 per million people.
2. `medquad_ans_0000775-1` (score=0.5524) - Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. The disorder affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes)...
3. `medquad_ans_0000775-5` (score=0.4482) - These resources address the diagnosis or management of paroxysmal nocturnal hemoglobinuria: - Duke University School of Medicine: Hemostasis & Thrombosis Center - Genetic Testing Registry: Paroxysmal nocturnal hemoglobinuria - MedlinePlu...
4. `medquad_ans_0004747-1` (score=0.2726) - Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It can occur at any age, but is usually diagnosed in young adulthood. People with PNH have recurr...
5. `medquad_ans_0004744-1` (score=0.2162) - What are the signs and symptoms of Paroxysmal cold hemoglobinuria? The Human Phenotype Ontology provides the following list of signs and symptoms for Paroxysmal cold hemoglobinuria. If the information is available, the table below includ...

**Answer**:

Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. It affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes), which protect the body from infection; and platelets (thrombocytes), which are involved in blood clotting. The disorder affects both sexes equally and can occur at any age, although it is most often diagnosed in young adulthood. People with paroxysmal nocturnal hemoglobinuria experience sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses on the body, such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to hemoglobinuria—dark-colored urine due to the presence of hemoglobin. In many, but not all cases, hemoglobinuria is most noticeable in the morning. The premature destruction

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Top docs**:

1. `medquad_ans_0000010-10` (score=0.7000) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the lining of the large in...
2. `medquad_ans_0000010-1__dup5` (score=0.5611) - How Tumors Form The body is made up of many types of cells. Normally, cells grow, divide, and produce more cells as needed to keep the body healthy and functioning properly. Sometimes, however, the process goes wrong -- cells become abno...
3. `medquad_ans_0000037_3-1` (score=0.4993) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. Colorectal cancer is...
4. `medquad_ans_0000221-1__dup2` (score=0.4784) - The colon and rectum are part of the large intestine. Colorectal cancer occurs when tumors form in the lining of the large intestine. It is common in both men and women. The risk of developing colorectal cancer rises after age 50. You're...
5. `medquad_ans_0000037_4-1` (score=0.4761) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. - Different factors ...

**Answer**:

Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the lining of the large intestine, also called the large bowel [medquad_ans_0000010-10]. It is the second leading cause of death from cancer in the United States [medquad_ans_0000037_3-1].

**Citations detected**: `medquad_ans_0000010-10`, `medquad_ans_0000037_3-1`
**Supported citations**: `medquad_ans_0000010-10`, `medquad_ans_0000037_3-1`
**Unsupported citations**: _none_

## Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Top docs**:

1. `medquad_ans_0000544-2` (score=0.7000) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.
2. `medquad_ans_0000544-5` (score=0.5199) - These resources address the diagnosis or management of junctional epidermolysis bullosa: - Epidermolysis Bullosa Center, Cincinnati Children's Hospital Medical Center - Gene Review: Gene Review: Junctional Epidermolysis Bullosa - Genetic...
3. `medquad_ans_0002121-1` (score=0.4598) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main types of epidermolysis...
4. `medquad_ans_0002122-5` (score=0.3417) - How might epidermolysis bullosa acquisita be treated?
5. `medquad_ans_0003317-1` (score=0.3191) - Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz type and the Non-Her...

**Answer**:

Junctional epidermolysis bullosa (JEB) is a type of epidermolysis bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz type and the Non-Herlitz type. The Herlitz type is very severe, and individuals with this condition often do not survive infancy. The Non-Herlitz type includes several subtypes that cause mild to severe blistering of the skin present at birth or shortly thereafter. JEB is inherited in an autosomal recessive pattern and is caused by mutations in the LAMB3, COL17A1, or LAMC2, and LAMA3 genes. There is no cure for JEB; treatment is focused on management of blistering and prevention of secondary infections [medquad_ans_0003317-1]. Both types

**Citations detected**: `medquad_ans_0003317-1`
**Supported citations**: `medquad_ans_0003317-1`
**Unsupported citations**: _none_

## Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Top docs**:

1. `medquad_ans_0000668-2` (score=0.7000) - Molybdenum cofactor deficiency is a rare condition that is estimated to occur in 1 in 100,000 to 200,000 newborns worldwide. More than 100 cases have been reported in the medical literature, although it is thought that the condition is u...
2. `medquad_ans_0000668-3` (score=0.6396) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same signs and symptoms b...
3. `medquad_ans_0000668-5` (score=0.4935) - These resources address the diagnosis or management of molybdenum cofactor deficiency: - Genetic Testing Registry: Combined molybdoflavoprotein enzyme deficiency - Genetic Testing Registry: Molybdenum cofactor deficiency, complementation...
4. `medquad_ans_0000668-1` (score=0.4890) - Molybdenum cofactor deficiency is a rare condition characterized by brain dysfunction (encephalopathy) that worsens over time. Babies with this condition appear normal at birth, but within a week they have difficulty feeding and develop ...
5. `medquad_ans_0004143-3` (score=0.2450) - What causes Mowat-Wilson syndrome? Mowat-Wilson syndrome is caused by mutations in the ZEB2 (also known as ZFHX1B or SIP-1) gene. This gene provides instructions for making a protein that plays a critical role in the formation of many or...

**Answer**:

The genetic changes related to molybdenum cofactor deficiency are mutations in the MOCS1, MOCS2, or GPHN gene. These mutations cause type A, B, or C of the disorder, respectively, and result in reduced or eliminated function of the associated proteins involved in the biosynthesis of the molybdenum cofactor [medquad_ans_0000668-3].

**Citations detected**: `medquad_ans_0000668-3`
**Supported citations**: `medquad_ans_0000668-3`
**Unsupported citations**: _none_

## Example 12 (`0000756-1`)

**Question**: What is (are) ornithine translocase deficiency ?

**Top docs**:

1. `medquad_ans_0000756-2` (score=0.7000) - Ornithine translocase deficiency is a very rare disorder. Fewer than 100 affected individuals have been reported worldwide.
2. `medquad_ans_0000756-5` (score=0.3775) - These resources address the diagnosis or management of ornithine translocase deficiency: - Baby's First Test - Gene Review: Gene Review: Hyperornithinemia-Hyperammonemia-Homocitrullinuria Syndrome - Gene Review: Gene Review: Urea Cycle D...
3. `medquad_ans_0000755-4` (score=0.3354) - Ornithine transcarbamylase deficiency is an X-linked disorder. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes. A characteristic of X-linked i...
4. `medquad_ans_0004581-1` (score=0.3245) - Ornithine transcarbamylase (OTC) deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The nervous ...
5. `medquad_ans_0000755-2` (score=0.3203) - Ornithine transcarbamylase deficiency is believed to occur in approximately 1 in every 80,000 people.

**Answer**:

Ornithine translocase deficiency is a very rare disorder, with fewer than 100 affected individuals reported worldwide [medquad_ans_0000756-2].

**Citations detected**: `medquad_ans_0000756-2`
**Supported citations**: `medquad_ans_0000756-2`
**Unsupported citations**: _none_

## Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Top docs**:

1. `medquad_ans_0000362-1__dup2` (score=0.7000) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance might be available. Cer...
2. `medquad_ans_0000036-8` (score=0.6411) - Medicaid is a state health insurance program for those with low incomes and few assets. Each state runs its own program. The Federal Government requires that Medicaid programs cover a specific set of services; however, states can choose ...
3. `medquad_ans_0000045-28` (score=0.6261) - The general number for Medicare is 1-800-Medicare (1-800-633-4227). TTY users should call 1-877-486-2048. You can also visit http://www.medicare.gov. The "Medicare & You" handbook is mailed out to all Medicare enrollees in the fall. It i...
4. `medquad_ans_0000003-6__dup1` (score=0.4853) - Most people with Alzheimers disease are cared for at home by family members. Within families, caregiving is provided most often by wives and husbands, followed by daughters. As Alzheimers disease gets worse, the person will need more and...
5. `medquad_ans_0000036-16` (score=0.4747) - - Diabetes management and treatment is expensive. Many people who have diabetes need help paying for their care. For those who qualify, a variety of government and nongovernment programs can help cover health care expenses. - Health insu...

**Answer**:

Yes, there is information about financial assistance. If you don't have health insurance or need help with costs that aren't covered, financial assistance might be available through certain government programs and nonprofit organizations. You can also discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2]. Additionally, Medicaid and CHIP provide financial assistance to individuals with low incomes. Medicaid is a state health insurance program for those with low incomes and few assets, and CHIP offers free or low-cost coverage to children whose parents earn too much for Medicaid but not enough to afford a health plan [medquad_ans_0000036-8].

**Citations detected**: `medquad_ans_0000362-1__dup2`, `medquad_ans_0000036-8`
**Supported citations**: `medquad_ans_0000362-1__dup2`, `medquad_ans_0000036-8`
**Unsupported citations**: _none_

## Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Top docs**:

1. `medquad_ans_0004105-1` (score=0.7000) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About half of affected indivi...
2. `medquad_ans_0004105-2` (score=0.3894) - What are the symptoms of Mollaret meningitis? The symptoms of Mollaret meningitis are the same as those found in other types of meningitis. In Mollaret meningitis, however, the symptoms are recurring and are often accompanied by long-ter...
3. `medquad_ans_0000598-1__dup2` (score=0.2244) - Meningitis is inflammation of the thin tissue that surrounds the brain and spinal cord, called the meninges. There are several types of meningitis. The most common is viral meningitis, which you get when a virus enters the body through t...
4. `medquad_ans_0000115-1__dup1` (score=0.1865) - Meningitis is an infection of the meninges, the membranes that surround the brain and spinal cord. Encephalitis is inflammation of the brain itself. Causes of encephalitis and meningitis include viruses, bacteria, fungus, and parasites. ...
5. `medquad_ans_0004749-4` (score=0.1470) - How is pars planitis diagnosed? Pars planitis is typically diagnosed based on a specialized eye examination. During the exam, the ophthalmologist will typically see clusters of white blood cells trapped within the eyeball that are called...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches, separated by weeks or months of no symptoms. About half of affected individuals may also experience long-term abnormalities of the nervous system that come and go, such as seizures, double vision, abnormal reflexes, paralysis of a cranial nerve (paresis), hallucinations, or coma. The exact cause remains unknown, but recent data suggests that herpes simplex virus (HSV-2 and, less frequently, HSV-1) may cause some, if not most cases. Other potential causes include trauma and viral infections other than herpes simplex [medquad_ans_0004105-1]. The symptoms of Mollaret meningitis are similar to those of other types of meningitis and include high fever, severe headache, nausea, vomiting, stiff neck, photophobia (s

**Citations detected**: `medquad_ans_0004105-1`
**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

## Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Top docs**:

1. `medquad_ans_0003331-1` (score=0.7000) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells in the brain and spi...
2. `medquad_ans_0003331-3` (score=0.6787) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but it...
3. `medquad_ans_0003319-1` (score=0.6682) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. Unlike other types of...
4. `medquad_ans_0003319-2` (score=0.5225) - What are the signs and symptoms of juvenile amyotrophic lateral sclerosis? Signs and symptoms of juvenile ALS vary but include slowly to very slowly progressive muscle weakness, increased muscle tone, Babinski reflex, muscle spasm (clonu...
5. `medquad_ans_0000551-2` (score=0.5154) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.

**Answer**:

Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be inherited in an autosomal dominant or autosomal recessive fashion [medquad_ans_0003319-1]. However, the specific genetic mutations or causes are not detailed in the provided context. Therefore, based on the available information, the exact cause of juvenile ALS is not fully specified.

**Citations detected**: `medquad_ans_0003319-1`
**Supported citations**: `medquad_ans_0003319-1`
**Unsupported citations**: _none_

## Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Top docs**:

1. `medquad_ans_0000219-11` (score=0.7000) - - Erectile dysfunction (ED) is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last long enough for sex...
2. `medquad_ans_0000334-1__dup2` (score=0.6885) - Erectile dysfunction (ED) is a common type of male sexual dysfunction. It is when a man has trouble getting or keeping an erection. ED becomes more common as you get older. But it's not a natural part of aging. Some people have trouble s...
3. `medquad_ans_0000219-8` (score=0.5918) - Your doctor can offer you a number of treatments for ED. For many men, the answer is as simple as taking a pill. Other men have to try two or three options before they find a treatment that works for them. Dont give up if the first treat...
4. `medquad_ans_0000219-1__dup4` (score=0.4939) - Erectile dysfunction\* is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last long enough for sex - are...
5. `medquad_ans_0000219-5__dup2` (score=0.4629) - Having ED can cause you to feel depressed or anxious. ED may also cause low self-esteem. When you have ED, you may not have a satisfying sex life. You may not feel as close with your sexual partner, which may strain your relationship. Se...

**Answer**:

To understand erectile dysfunction (ED), know that it is when you cannot get or keep an erection firm enough for sex. This can occur if you sometimes get an erection, but it does not last long enough, or if you are unable to get an erection at all [medquad_ans_0000219-11]. An erection happens when blood flow into the penis increases, involving hormones, blood vessels, nerves, and muscles working together [medquad_ans_0000219-11]. ED often has multiple causes, such as diseases that damage nerves, arteries, or muscles [medquad_ans_0000219-11].

To determine the cause, your doctor may take your medical and sexual history, ask about your mental health, perform a physical exam, test your blood, conduct a nighttime erection test, or perform an injection or Doppler penile ultrasound [medquad*ans*

**Citations detected**: `medquad_ans_0000219-11`
**Supported citations**: `medquad_ans_0000219-11`
**Unsupported citations**: _none_

## Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Top docs**:

1. `medquad_ans_0006267-3` (score=0.7000) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation often becomes necessa...
2. `medquad_ans_0001537-1` (score=0.2484) - Congenital muscular dystrophy (CMD) refers to a group of inherited conditions that affect the muscles and are present at birth or in early infancy. The severity of the condition, the associated signs and symptoms and the disease progress...
3. `medquad_ans_0004514-4` (score=0.2283) - How is oculopharyngeal muscular dystrophy inherited?
4. `medquad_ans_0001026-5` (score=0.2027) - These resources address the diagnosis or management of Walker-Warburg syndrome: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Genetic Testing Registry: Walker-Warburg congenital muscular dystrophy These resources f...
5. `medquad_ans_0000631-1__dup1` (score=0.1895) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. The different types ...

**Answer**:

Physical therapy, including early mobilization, regular stretching, and splinting, is the main focus of supportive care for Ullrich congenital muscular dystrophy. Respiratory support and night-time ventilation may become necessary in the first or second decade of life. Prevention of chest infections can be achieved with the use of antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that cyclosporin A may benefit individuals with Ullrich congenital muscular dystrophy, although more studies are needed to confirm its effectiveness [medquad_ans_0006267-3].

**Citations detected**: `medquad_ans_0006267-3`
**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

## Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Top docs**:

1. `medquad_ans_0004005-1` (score=0.7000) - Microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) is a genetic condition that is mainly characterized by intrauterine and post-natal growth retardation; an abnormally small head size (microcephaly); abnormal bone growth (s...
2. `medquad_ans_0004006-1` (score=0.5566) - Microcephalic osteodysplastic primordial dwarfism type 2 (MOPD2) is a condition characterized by short stature (dwarfism), skeletal abnormalities and an unusually small head size (microcephaly). Other signs and symptoms of MOPD2 may incl...
3. `medquad_ans_0004005-5` (score=0.4235) - How might microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) be treated? At this time there are no specific treatments for MOPD1. Treatment is generally supportive. The prognosis is poor for affected individuals, with most ...
4. `medquad_ans_0000653-1` (score=0.3876) - Microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is a condition characterized by short stature (dwarfism) with other skeletal abnormalities (osteodysplasia) and an unusually small head size (microcephaly). The growth pr...
5. `medquad_ans_0004006-2` (score=0.3347) - What are the signs and symptoms of Microcephalic osteodysplastic primordial dwarfism type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephalic osteodysplastic primordial dwarfism type 2. If ...

**Answer**:

Yes, microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is inherited in an autosomal recessive manner [medquad_ans_0004006-1].

**Citations detected**: `medquad_ans_0004006-1`
**Supported citations**: `medquad_ans_0004006-1`
**Unsupported citations**: _none_

## Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000742-2` (score=0.7000) - Ochoa syndrome is a rare disorder. About 150 cases have been reported in the medical literature.
2. `medquad_ans_0006597-4` (score=0.3450) - Can 48,XXYY syndrome be inherited?
3. `medquad_ans_0000742-5` (score=0.2698) - These resources address the diagnosis or management of Ochoa syndrome: - Gene Review: Gene Review: Urofacial Syndrome - Genetic Testing Registry: Ochoa syndrome - National Institute of Diabetes and Digestive and Kidney Diseases: Urodynam...
4. `medquad_ans_0000324-4` (score=0.1852) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
5. `medquad_ans_0000883-4` (score=0.1685) - This condition is typically inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one affected ...

**Answer**:

The provided context does not explicitly state whether Ochoa syndrome is inherited. While it mentions that about 150 cases have been reported and lists resources for diagnosis and management, there is no direct information on the inheritance pattern of Ochoa syndrome. Therefore, based on the given context, it is not possible to determine if Ochoa syndrome is inherited.

[medquad_ans_0000742-2]

**Citations detected**: `medquad_ans_0000742-2`
**Supported citations**: `medquad_ans_0000742-2`
**Unsupported citations**: _none_

## Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Top docs**:

1. `medquad_ans_0000531-2` (score=0.7000) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
2. `medquad_ans_0000531-3` (score=0.5194) - Isodicentric chromosome 15 syndrome results from the presence of an abnormal extra chromosome, called an isodicentric chromosome 15, in each cell. An isodicentric chromosome contains mirror-image segments of genetic material and has two ...
3. `medquad_ans_0000531-5` (score=0.4602) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management of various health c...
4. `medquad_ans_0000531-1` (score=0.3179) - Isodicentric chromosome 15 syndrome is a developmental disorder with a broad spectrum of features. The signs and symptoms vary among affected individuals. Poor muscle tone is commonly seen in individuals with isodicentric chromosome 15 s...
5. `medquad_ans_0004132-4` (score=0.3169) - How might mosaic trisomy 14 be treated? Treatment for signs and symptoms of mosaic trisomy 14 focuses on the specific features present in each individual. Infants with congenital heart defects may need surgery or other therapies to allev...

**Answer**:

The provided context does not specify treatments for isodicentric chromosome 15 syndrome. It mentions resources that address the diagnosis and management of the condition, including diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care [medquad_ans_0000531-5], but does not detail specific treatments. Therefore, based on the given information, a definitive answer about treatments cannot be provided.

**Citations detected**: `medquad_ans_0000531-5`
**Supported citations**: `medquad_ans_0000531-5`
**Unsupported citations**: _none_

## Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Top docs**:

1. `medquad_ans_0004414-3` (score=0.7000) - What causes Nicolaides-Baraitser syndrome? Nicolaides-Baraitser syndrome (NCBRS) is caused by mutations in the SMARCA2 gene, which is located on the small arm of chromosome 9. All mutations that have been identified in affected people ha...
2. `medquad_ans_0002439-1` (score=0.4702) - Friedreich ataxia is an inherited condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features include the gradual...
3. `medquad_ans_0001143-1` (score=0.4544) - CHILD syndrome, also known as congenital hemidysplasia with ichthyosiform erythroderma and limb defects, is a genetic condition that is typically characterized by large patches of skin that are red and inflamed (erythroderma) and covered...
4. `medquad_ans_0000115-1__dup4` (score=0.3828) - Mntriers disease causes the ridges along the inside of the stomach wallcalled rugaeto enlarge, forming giant folds in the stomach lining. The rugae enlarge because of an overgrowth of mucous cells in the stomach wall. In a normal stomach...
5. `medquad_ans_0002049-3` (score=0.3479) - What causes Ehlers-Danlos syndrome? Ehlers-Danlos syndrome can be caused by changes (mutations) in several different genes (COL5A1, COL5A2, COL1A1, COL3A1, TNXB, PLOD1, COL1A2, and ADAMTS2). However, the underlying genetic cause is unkno...

**Answer**:

The provided context does not contain any information about Troyer syndrome. Therefore, it is not possible to determine the cause of Troyer syndrome based on the given text. [no_context]

**Citations detected**: `no_context`
**Supported citations**: _none_
**Unsupported citations**: `no_context`

## Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Top docs**:

1. `medquad_ans_0000983-4` (score=0.7000) - This condition is considered to have an autosomal dominant pattern of inheritance, which means one copy of the altered CACNA1C gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene, and o...
2. `medquad_ans_0003672-1` (score=0.5422) - Timothy syndrome is a type of long QT syndrome. It affects many parts of the body including the heart, fingers, toes, face, and the nervous system. It is characterized by long QT syndrome, although some people with Timothy syndrome also ...
3. `medquad_ans_0000983-5` (score=0.4571) - These resources address the diagnosis or management of Timothy syndrome: - Gene Review: Gene Review: Timothy Syndrome - Genetic Testing Registry: Timothy syndrome - MedlinePlus Encyclopedia: Arrhythmias - MedlinePlus Encyclopedia: Congen...
4. `medquad_ans_0000983-1` (score=0.4270) - Timothy syndrome is a rare disorder that affects many parts of the body including the heart, digits (fingers and toes), and the nervous system. Timothy syndrome is characterized by a heart condition called long QT syndrome, which causes ...
5. `medquad_ans_0000983-2` (score=0.3348) - Timothy syndrome is a rare condition; fewer than 20 people with this disorder have been reported worldwide. The classic type of Timothy syndrome appears to be more common than the atypical type, which has been identified in only two indi...

**Answer**:

The genetic changes related to Timothy syndrome are changes in the CACNA1C gene. This condition is caused by alterations in the CACNA1C gene, and all cases of Timothy syndrome appear to be due to changes in this gene [medquad_ans_0003672-1]. The disorder follows an autosomal dominant pattern of inheritance, meaning one copy of the altered CACNA1C gene in each cell is sufficient to cause the disorder [medquad_ans_0000983-4]. Most cases result from new mutations in the gene and occur in people with no family history of the disorder, although less commonly, individuals inherit the altered gene from an unaffected parent who is mosaic for a CACNA1C mutation [medquad_ans_0000983-4].

**Citations detected**: `medquad_ans_0003672-1`, `medquad_ans_0000983-4`
**Supported citations**: `medquad_ans_0003672-1`, `medquad_ans_0000983-4`
**Unsupported citations**: _none_

## Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Top docs**:

1. `medquad_ans_0000061-11` (score=0.7000) - Rheumatoid arthritis can be difficult to diagnose in its early stages because the full range of symptoms develops over time, and only a few symptoms may be present in the early stages. As part of the diagnosis, your doctor will look for ...
2. `medquad_ans_0000875-1` (score=0.6328) - Rheumatoid arthritis is a disease that causes chronic abnormal inflammation, primarily affecting the joints. The most common signs and symptoms are pain, swelling, and stiffness of the joints. Small joints in the hands and feet are invol...
3. `medquad_ans_0000061-7__dup1` (score=0.5511) - Rheumatoid arthritis is an inflammatory disease that causes pain, swelling, stiffness, and loss of function in the joints. It can cause mild to severe symptoms. People with rheumatoid arthritis may feel sick, tired, and sometimes feveris...
4. `medquad_ans_0005261-2` (score=0.4615) - What are the signs and symptoms of Reactive arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Reactive arthritis. If the information is available, the table below includes how often the symptom...
5. `medquad_ans_0000061-3__dup3` (score=0.4259) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by inflammation of the j...

**Answer**:

The symptoms of rheumatoid arthritis include pain, swelling, and stiffness of the joints, with small joints in the hands and feet being most commonly affected. The disease often affects joints in a symmetrical pattern, such as both hands or both knees. Joint pain and stiffness are typically worse in the morning or after prolonged rest. Additional signs and symptoms may include fatigue, low fever, weight loss, anemia, and inflammation of other tissues and organs such as the eyes, lungs, and blood vessels. Some individuals may develop rheumatoid nodules, which are firm, noncancerous lumps under the skin. The symptoms usually appear in mid- to late adulthood and may come in episodes of flares followed by periods of remission [medquad_ans_0000875-1]. The disease can cause mild to severe symptoms and may lead to significant joint damage and disability over time [medquad_ans_00000

**Citations detected**: `medquad_ans_0000875-1`
**Supported citations**: `medquad_ans_0000875-1`
**Unsupported citations**: _none_

## Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Top docs**:

1. `medquad_ans_0002796-1` (score=0.7000) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is not fully understood, ...
2. `medquad_ans_0000149-1__dup3` (score=0.6615) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, digestive tract, and ...
3. `medquad_ans_0002796-2` (score=0.6107) - What are the signs and symptoms of Henoch-Schonlein purpura? The Human Phenotype Ontology provides the following list of signs and symptoms for Henoch-Schonlein purpura. If the information is available, the table below includes how often...
4. `medquad_ans_0002796-4` (score=0.5885) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti-inflammatory drugs ...
5. `medquad_ans_0000149-2__dup2` (score=0.5165) - Henoch-Schnlein purpura is caused by an abnormal immune system response in which the bodys immune system attacks the bodys own cells and organs. Usually, the immune system makes antibodies, or proteins, to protect the body from foreign s...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). It is caused by inflammation of small blood vessels in the body, leading to leakage. The primary symptom is a rash that appears as many small raised bruises. HSP can affect the kidneys, digestive tract, and joints. It is most common in children between 2 and 6 years of age but can occur at any age. Most people recover completely, though kidney damage is the most likely long-term complication. In adults, HSP can lead to chronic kidney disease (CKD) and kidney failure, including end-stage renal disease requiring dialysis or kidney transplant. Common signs and symptoms include abdominal pain (90%), arthralgia (90%), bruising susceptibility (90%), hematuria (9

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Top docs**:

1. `medquad_ans_0003166-1` (score=0.7000) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information is available, the ta...
2. `medquad_ans_0003162-2` (score=0.4619) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 1? Symptoms and physical findings associated with hyper IgM syndrome usually become apparent in the first or second year of life. This condition may be characterized...
3. `medquad_ans_0003163-1` (score=0.4075) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information is available, the ta...
4. `medquad_ans_0003162-1` (score=0.3600) - Hyper IgM syndrome is a type of primary immunodeficiency syndrome. Primary immunodeficiency occurs when part of a persons immune system is missing or does not work correctly. The bodies of people with primary immunodeficiency cant get ri...
5. `medquad_ans_0001446-2` (score=0.3487) - What are the signs and symptoms of Common variable immunodeficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Common variable immunodeficiency. If the information is available, the table below in...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [medquad_ans_0003166-1].

**Citations detected**: `medquad_ans_0003166-1`
**Supported citations**: `medquad_ans_0003166-1`
**Unsupported citations**: _none_

## Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Top docs**:

1. `medquad_ans_0002533-4` (score=0.7000) - How is Gerstmann-Straussler-Scheinker disease inherited? Gerstmann-Straussler-Scheinker disease (GSS) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of th...
2. `medquad_ans_0000883-3__dup1` (score=0.3852) - How is Buschke Ollendorff syndrome inherited? Buschke Ollendorff syndrome (BOS) is caused by mutations in the LEMD3 gene and is inherited in an autosomal dominant manner. This means that only one changed (mutated) copy of the gene in eac...
3. `medquad_ans_0000700-4` (score=0.3360) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one affected parent. Ap...
4. `medquad_ans_0000516-4` (score=0.3056) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one affected parent. Ot...
5. `medquad_ans_0000149-4` (score=0.3020) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one affected parent. Ot...

**Answer**:

The provided context does not contain any information about Senior-Loken syndrome or its associated genetic changes. Therefore, it is not possible to answer the question based on the given context. [no relevant context]

**Citations detected**: `no relevant context`
**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

## Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Top docs**:

1. `medquad_ans_0000610-2` (score=0.7000) - The prevalence of lymphedema-distichiasis syndrome is unknown. Because the extra eyelashes can be overlooked during a medical examination, researchers believe that some people with this condition may be misdiagnosed as having lymphedema ...
2. `medquad_ans_0003700-2` (score=0.6216) - What are the signs and symptoms of Lymphedema-distichiasis syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lymphedema-distichiasis syndrome. If the information is available, the table below in...
3. `medquad_ans_0000195-2` (score=0.4444) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0000113-2` (score=0.2499) - Beckwith-Wiedemann syndrome affects an estimated 1 in 13,700 newborns worldwide. The condition may actually be more common than this estimate because some people with mild symptoms are never diagnosed.
5. `medquad_ans_0003700-1` (score=0.2205) - Lymphedema distichiasis syndrome is a condition that affects the normal function of the lymphatic system (part of the immune system that produces and transports fluids and immune cells throughout the body). People with this condition are...

**Answer**:

The incidence of lymphedema-distichiasis syndrome is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people [medquad_ans_0000195-2].

**Citations detected**: `medquad_ans_0000195-2`
**Supported citations**: `medquad_ans_0000195-2`
**Unsupported citations**: _none_

## Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Top docs**:

1. `medquad_ans_0000013-22` (score=0.7000) - In a genetic test, a small sample of blood, saliva, or tissue is taken to examine a person's genes. Sometimes, genetic testing can detect diseases that may be preventable or treatable. This type of testing is available for thousands of c...
2. `medquad_ans_0000399-1__dup1` (score=0.6323) - Summary : Genetic tests are tests on blood and other tissue to find genetic disorders. Over 2000 tests are available. Doctors use genetic tests for several reasons. These include - Finding genetic diseases in unborn babies - Finding out ...
3. `medquad_ans_0000013-25` (score=0.5160) - Genetic testing may be helpful whether the test identifies a mutation or not. Test results can - serve as a relief, eliminating some of the uncertainty about a health condition - help doctors make recommendations for treatment or monitor...
4. `medquad_ans_0000013-26` (score=0.4411) - Finding out your test results can affect you emotionally. Learning that you are someone in your family has or is at risk for a disease can be scary. Some people can also feel guilty, angry, anxious, or depressed when they find out their ...
5. `medquad_ans_0000397-1__dup1` (score=0.3146) - Summary : Genetic counseling provides information and support to people who have, or may be at risk for, genetic disorders. A genetic counselor meets with you to discuss genetic risks. The counseling may be for yourself or a family membe...

**Answer**:

Yes, I have information about genetic testing. Genetic testing involves taking a small sample of blood, saliva, or tissue to examine a person's genes. It can detect genetic disorders and is available for thousands of conditions. Genetic tests are used for various purposes, including identifying genetic diseases in unborn babies, determining if individuals carry a gene that could be passed to their children, screening embryos for disease, testing adults before symptoms appear, making a diagnosis in someone with symptoms, and determining the best type or dose of medicine for a person. Test results can provide relief by reducing uncertainty, help doctors make treatment or monitoring recommendations, support decisions about personal and family health, encourage preventive actions such as earlier screenings or lifestyle changes, and assist in making future life decisions like family planning. However, test results can also have emotional impacts, such as fear, guilt, anger, anxiety, or depression, and the cost of testing can vary from less than $100 to higher amounts

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Top docs**:

1. `medquad_ans_0000149-1` (score=0.7000) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxygen-rich blood from ...
2. `medquad_ans_0000654-2` (score=0.4250) - Microcephaly-capillary malformation syndrome is rare. About a dozen people have been diagnosed with the disorder.
3. `medquad_ans_0000654-5` (score=0.4068) - These resources address the diagnosis or management of microcephaly-capillary malformation syndrome: - Gene Review: Gene Review: Microcephaly-Capillary Malformation Syndrome - Genetic Testing Registry: Microcephaly-capillary malformation...
4. `medquad_ans_0006335-1` (score=0.3696) - Vein of Galen aneurysm is a rare form of arteriovenous malformation in which a particular vein at the base of the brain, the vein of Galen, dilates causing too much blood to rush to the heart and leading to congestive heart failure. Some...
5. `medquad_ans_0000056-1__dup3` (score=0.3537) - Arteriovenous malformations (AVMs) are defects in your vascular system. The vascular system includes arteries, veins, and capillaries. Arteries carry blood away from the heart to other organs; veins carry blood back to the heart. Capilla...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin. These malformations appear as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs, and may be visible at birth or during childhood. While capillary malformations alone usually do not cause health problems, some individuals with CM-AVM also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs). These abnormalities involve abnormal connections between arteries, veins, and capillaries, which can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure—some of which can be life-threatening. Complications from AVMs and AV

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Top docs**:

1. `medquad_ans_0001992-1` (score=0.7000) - What are the signs and symptoms of Dystonia 16? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 16. If the information is available, the table below includes how often the symptom is seen in pe...
2. `medquad_ans_0002000-1` (score=0.5659) - What are the signs and symptoms of Dystonia 7, torsion? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 7, torsion. If the information is available, the table below includes how often the sympt...
3. `medquad_ans_0001986-1` (score=0.4875) - What are the signs and symptoms of Dystonia 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 1. If the information is available, the table below includes how often the symptom is seen in peop...
4. `medquad_ans_0001988-1` (score=0.4502) - What are the signs and symptoms of Dystonia 11? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 11. If the information is available, the table below includes how often the symptom is seen in pe...
5. `medquad_ans_0001994-1` (score=0.4499) - What are the signs and symptoms of Dystonia 18? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 18. If the information is available, the table below includes how often the symptom is seen in pe...

**Answer**:

The symptoms of Dystonia 16 include cognitive impairment (5%), abnormal pyramidal signs, bradykinesia, delayed speech and language development, dysarthria, dysphagia, gait disturbance, hyperreflexia, involuntary movements, laryngeal dystonia, limb dystonia, lower limb pain, morphological abnormality of the pyramidal tract, motor delay, parkinsonism, postural tremor, progressive features, retrocollis [medquad_ans_0001992-1].

**Citations detected**: `medquad_ans_0001992-1`
**Supported citations**: `medquad_ans_0001992-1`
**Unsupported citations**: _none_

## Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Top docs**:

1. `medquad_ans_0005548-5` (score=0.7000) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not treatable. Vision, phy...
2. `medquad_ans_0000899-2` (score=0.5128) - Septo-optic dysplasia has a reported incidence of 1 in 10,000 newborns.
3. `medquad_ans_0000899-5` (score=0.3803) - These resources address the diagnosis or management of septo-optic dysplasia: - Genetic Testing Registry: Septo-optic dysplasia sequence - MedlinePlus Encyclopedia: Growth Hormone Deficiency - MedlinePlus Encyclopedia: Hypopituitarism Th...
4. `medquad_ans_0005548-2` (score=0.3583) - What are the signs and symptoms of Septo-optic dysplasia? Symptoms may include blindness in one or both eyes, pupil dilation in response to light, nystagmus (a rapid, involuntary to-and-fro movement of the eyes), inward and outward devia...
5. `medquad_ans_0005548-1` (score=0.2878) - Septo-optic dysplasia is a disorder of early brain development. The signs and symptoms vary from person to person; however, underdevelopment (hypoplasia) of the optic nerve, abnormal formation of structures along the midline of the brain...

**Answer**:

Treatment for septo-optic dysplasia is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not treatable, and vision, physical, and occupational therapies may be required [medquad_ans_0005548-5].

**Citations detected**: `medquad_ans_0005548-5`
**Supported citations**: `medquad_ans_0005548-5`
**Unsupported citations**: _none_

## Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Top docs**:

1. `medquad_ans_0000208-1__dup4` (score=0.7000) - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. Benign prostatic hyperplasia is also called benign prostatic hypertrophy or benign prostatic obstruction. The pro...
2. `medquad_ans_0000036_3-1` (score=0.4586) - Key Points - Prostate cancer is a disease in which malignant (cancer) cells form in the tissues of the prostate. - Prostate cancer is the most common nonskin cancer among men in the United States. - Different factors increase or decrease...
3. `medquad_ans_0000208-14` (score=0.4205) - - Benign prostatic hyperplasiaalso called BPHis a condition in men in which the prostate gland is enlarged and not cancerous. - The prostate is a walnut-shaped gland that is part of the male reproductive system. - The cause of benign pro...
4. `medquad_ans_0000752-1__dup1` (score=0.3889) - The prostate is a gland in men. It helps make semen, the fluid that contains sperm. The prostate surrounds the tube that carries urine away from the bladder and out of the body. A young man's prostate is about the size of a walnut. It sl...
5. `medquad_ans_0000058-19` (score=0.3792) - Yes. Any of the symptoms caused by prostate cancer may also be due to enlargement of the prostate, which is not cancer. If you have any of the symptoms mentioned in question #10, see your doctor or a urologist to find out if you need tre...

**Answer**:

The complications of prostate enlargement due to benign prostatic hyperplasia (BPH) include narrowing of the urethra and urinary retention, which is the inability to empty the bladder completely. This can lead to the bladder wall becoming thicker and eventually weakening, resulting in incomplete bladder emptying and leaving some urine in the bladder [medquad_ans_0000208-1__dup4]. These issues contribute to many of the symptoms and problems associated with BPH [medquad_ans_0000208-1__dup4].

**Citations detected**: `medquad_ans_0000208-1__dup4`
**Supported citations**: `medquad_ans_0000208-1__dup4`
**Unsupported citations**: _none_

## Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Top docs**:

1. `medquad_ans_0004784-5` (score=0.7000) - How might Pelizaeus-Merzbacher disease be treated?
2. `medquad_ans_0000029-18` (score=0.6300) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
3. `medquad_ans_0004770-2` (score=0.5124) - How might patulous eustacian tube be treated? While no standard treatment has been found to work for every patient, there are several options that have been used to successfully manage the symptoms in a number of cases. Patients are ofte...
4. `medquad_ans_0006236-5` (score=0.4791) - How might tubular aggregate myopathy be treated?
5. `medquad_ans_0000860-2__dup1` (score=0.4674) - How might Brown-Sequard syndrome be treated?

**Answer**:

The provided context does not contain information about the treatments for Absence of the Septum Pellucidum. Therefore, no treatments can be identified from the given sources. [no relevant context]

**Citations detected**: `no relevant context`
**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

## Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.7000) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (i.e., PLEVA). In FUM...
2. `medquad_ans_0002322-1` (score=0.3642) - Febrile ulceronecrotic Mucha-Habermann disease (FUMHD) is a rare and severe form of pityriasis lichenoides et varioliformis acuta (PLEVA). PLEVA is characterized by skin lesions that ulcerate, breakdown, form open sores, then form a red-...
3. `medquad_ans_0002322-4` (score=0.3640) - How is febrile ulceronecrotic Mucha-Habermann disease definitively diagnosed? FUMHD is diagnosed based upon the clinical symptoms in the patient, with confirmation by skin biopsy. Skin biopsy findings suggestive of FUMHD are outlined bel...
4. `medquad_ans_0002322-5` (score=0.1326) - How is febrile ulceronecrotic Mucha-Habermann disease (FUMHD) treated? It is important that FUMHD is diagnosed and treated as soon as possible. While a number of treatments have been tried, it is hard to asses the benefit of the therapie...
5. `medquad_ans_0000114-2__dup1` (score=0.1287) - What are the signs and symptoms of Acute febrile neutrophilic dermatosis? The most obvious signs of acute febrile neutrophilic dermatosis are distinctive skin lesions that usually develop according to a specific pattern. Typically, a ser...

**Answer**:

The symptoms of Febrile Ulceronecrotic Mucha-Habermann disease (FUMHD) include initial red scaly skin lesions (papules) that ulcerate, break down, form open sores, and then develop a red-brown crust (i.e., PLEVA). These lesions can rapidly progress to large, destructive ulcers with extensive, painful loss of skin tissue. Skin lesions may become infected, leading to pus and a putrid odor. FUMHD is often associated with high fever (up to 104°F), which may be persistent or intermittent. Other symptoms may include feeling ill, sore throat, congestion, muscle soreness or pain, joint pain, diarrhea, central nervous system symptoms, abdominal pain, enlarged spleen, arthritis, megaloblastic anemia, interstitial pneumonitis, lymphocytic (viral) myocarditis, and sepsis. FUMHD can

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Top docs**:

1. `medquad_ans_0003951-3` (score=0.7000) - Is Merkel cell carcinoma inherited? To our knowledge, there currently is no evidence that Merkel cell carcinoma (MCC) is inherited. While DNA changes (mutations) found in the cells of MCC tumors can lead to MCC, these types of mutations ...
2. `medquad_ans_0000167-5__dup1` (score=0.2671) - Autosomal dominant PKD is the most common form of PKD and the most common inherited disorder of the kidneys.3 The term autosomal dominant means a child can get the disorder by inheriting the gene mutation from only one parent. Each child...
3. `medquad_ans_0000615-4` (score=0.1614) - Maffucci syndrome is not inherited. The mutations that cause this disorder are somatic, which means they occur during a person's lifetime. A somatic mutation occurs in a single cell. As that cell continues to grow and divide, the cells d...
4. `medquad_ans_0000167-2__dup3` (score=0.0176) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up a gene. In most cas...
5. `medquad_ans_0000676-4` (score=0.0000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...

**Answer**:

The provided context does not contain information about hereditary leiomyomatosis and renal cell cancer (HLRCC). Therefore, based on the given context, it is not possible to determine whether HLRCC is inherited.

Answer: Insufficient context.

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Top docs**:

1. `medquad_ans_0002445-1` (score=0.7000) - What are the signs and symptoms of Frontometaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontometaphyseal dysplasia. If the information is available, the table below includes how...
2. `medquad_ans_0000389-2` (score=0.4527) - Frontometaphyseal dysplasia is a rare disorder; only a few dozen cases have been reported worldwide.
3. `medquad_ans_0000389-1` (score=0.4150) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which also includes otopalat...
4. `medquad_ans_0002444-1` (score=0.3065) - What are the signs and symptoms of Frontofacionasal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontofacionasal dysplasia. If the information is available, the table below includes how o...
5. `medquad_ans_0000389-5` (score=0.2710) - These resources address the diagnosis or management of frontometaphyseal dysplasia: - Gene Review: Gene Review: Otopalatodigital Spectrum Disorders - Genetic Testing Registry: Frontometaphyseal dysplasia These resources from MedlinePlus ...

**Answer**:

The symptoms of Frontometaphyseal dysplasia include abnormality of dental morphology, abnormality of frontal sinus, abnormality of the metaphyses, bowing of the long bones, camptodactyly of finger, craniofacial hyperostosis, hypertelorism, limitation of joint mobility, prominent supraorbital ridges, abnormal form of the vertebral bodies, abnormality of the palate, accelerated skeletal maturation, aplasia/hypoplasia of the thumb, arachnodactyly, conductive hearing impairment, elbow dislocation, scoliosis, sensorineural hearing impairment, skeletal muscle atrophy, synostosis of carpal bones, ulnar deviation of finger, abnormality of the larynx, abnormality of the urethra, complete atrioventricular canal defect, craniosynostosis, tracheal stenosis, ureteral stenosis, ankle

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Top docs**:

1. `medquad_ans_0000128-7` (score=0.7000) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding, you need to go to t...
2. `medquad_ans_0000128-11` (score=0.6466) - - Diverticular disease is more common in people as they grow older. - A low-fiber diet is the most likely cause of the disease. - Most people are treated with a high-fiber diet and pain medication. - Add whole grain foods, high-fiber fru...
3. `medquad_ans_0000128-10` (score=0.5998) - Eat a high-fiber diet to help prevent problems. Talk to your doctor about using fiber products like Benefiber, Citrucel, or Metamucil. Daily use can help you get the fiber you need if you do not get it through your diet. Ask your doctor ...
4. `medquad_ans_0000097-8` (score=0.4777) - A health care provider may treat the symptoms of diverticulosis with a high-fiber diet or fiber supplements, medications, and possibly probiotics. Treatment for diverticular disease varies, depending on whether a person has diverticuliti...
5. `medquad_ans_0000128-5__dup1` (score=0.4341) - Many people get diverticular disease. Starting at age 40, the chance of getting it increases about every 10 years. About half of people between the ages of 60 and 80 have diverticular disease. Almost everyone over 80 has it.

**Answer**:

Treatments for diverticular disease depend on the severity of the condition and whether the individual has diverticulosis or diverticulitis. Most people improve with dietary changes, particularly a high-fiber diet, which includes whole grain foods, high-fiber fruits, and vegetables such as raw apples, peaches, pears, tangerines, broccoli, squash, carrots, brussels sprouts, potatoes, baked beans, kidney beans, lima beans, whole-wheat bread, brown rice, bran flake cereal, and oatmeal [medquad_ans_0000128-10].

For diverticulosis, a high-fiber diet or fiber supplements like methylcellulose (Citrucel) or psyllium (Metamucil) may be recommended, with fiber supplements taken one to three times a day and with at least 8 ounces of water [medquad_ans_00

**Citations detected**: `medquad_ans_0000128-10`
**Supported citations**: `medquad_ans_0000128-10`
**Unsupported citations**: _none_

## Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Top docs**:

1. `medquad_ans_0000619-1__dup1` (score=0.7000) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is available, the table belo...
2. `medquad_ans_0000676-2__dup1` (score=0.3075) - What are the symptoms of Bell's palsy?
3. `medquad_ans_0006066-1` (score=0.1854) - What are the signs and symptoms of Thompson Baraitser syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thompson Baraitser syndrome. If the information is available, the table below includes how...
4. `medquad_ans_0004414-2` (score=0.1123) - What are the signs and symptoms of Nicolaides-Baraitser syndrome? Nicolaides-Baraitser syndrome (NCBRS) is typically characterized by intellectual disability, seizures, short stature, sparse hair, distinctive facial features, short finge...
5. `medquad_ans_0004414-1` (score=0.1099) - Nicolaides-Baraitser syndrome (NCBRS) is a very rare condition characterized by severe intellectual disability and various physical features. Signs and symptoms may include seizures, short stature, sparse hair, distinctive facial charact...

**Answer**:

The symptoms of Baraitser Brett Piesowicz syndrome include cerebral calcification, hyperreflexia, hypertonia, microcephaly, seizures, abnormality of movement, cerebral cortical atrophy, cataract, opacification of the corneal stroma, renal insufficiency, anteverted nares, cerebellar hypoplasia, decreased liver function, elevated hepatic transaminases, failure to thrive, hepatomegaly, high palate, increased CSF protein, intellectual disability, profound, jaundice, lissencephaly, long philtrum, low-set ears, microretrognathia, muscular hypotonia of the trunk, nystagmus, pachygyria, petechiae, phenotypic variability, polymicrogyria, sloping forehead, spasticity, splenomegaly, thrombocytopenia, ventriculomeg

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Top docs**:

1. `medquad_ans_0000052-2__dup1` (score=0.7000) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
2. `medquad_ans_0000029-18` (score=0.4698) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
3. `medquad_ans_0000273-2__dup2` (score=0.3460) - There is no cure for TSC, although treatment is available for a number of the symptoms. Rapamycin and related drugs are not yet approved by the U.S. Food and Drug Administration (FDA) for any purpose in individuals with TSC. The FDA has ...
4. `medquad_ans_0004053-4` (score=0.2931) - What is the treatment for microscopic polyangiitis (MPA)? MPA is treated with medications that suppress the immune system, which can lower an individual's resistance to infections. There are a variety of immune suppressing medications th...
5. `medquad_ans_0000205-2__dup1` (score=0.2835) - There is no specific treatment to stop or reverse any form of MD. Treatment may include physical therapy, respiratory therapy, speech therapy, orthopedic appliances used for support, and corrective orthopedic surgery. Drug therapy includ...

**Answer**:

The provided context does not contain information about the treatments for meningioma. Therefore, a specific answer cannot be given based on the available context. [no relevant context]

**Citations detected**: `no relevant context`
**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

## Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Top docs**:

1. `medquad_ans_0000006_7-2` (score=0.7000) - Key Points - Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Childhood central nervous system (CNS) germ cell tumors rarely spread outside of the brain and spinal cord. Staging ...
2. `medquad_ans_0000006_7-8` (score=0.6023) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a team of health care pr...
3. `medquad_ans_0000006_7-3` (score=0.5144) - The cause of most childhood CNS germ cell tumors is not known.
4. `medquad_ans_0000015_1-2` (score=0.3984) - The cause of most childhood extracranial germ cell tumors is unknown.
5. `medquad_ans_0000006_7-1` (score=0.3970) - Key Points - Childhood central nervous system (CNS) germ cell tumors form from germ cells. - There are different types of childhood CNS germ cell tumors. - Germinomas - Nongerminomas - The cause of most childhood CNS germ cell tumors is ...

**Answer**:

The treatments for Childhood Central Nervous System (CNS) Germ Cell Tumors include radiation therapy, chemotherapy, surgery, high-dose chemotherapy with stem cell rescue, and new treatments being tested in clinical trials. Treatment is planned by a team of healthcare providers who are experts in treating cancer in children. Patients may consider participating in a clinical trial, which can occur before, during, or after starting treatment [medquad_ans_0000006_7-8].

**Citations detected**: `medquad_ans_0000006_7-8`
**Supported citations**: `medquad_ans_0000006_7-8`
**Unsupported citations**: _none_

## Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Top docs**:

1. `medquad_ans_0000729-2` (score=0.7000) - Neutral lipid storage disease with myopathy is a rare condition; its incidence is unknown.
2. `medquad_ans_0000729-1` (score=0.6756) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the accumulation of fats...
3. `medquad_ans_0004401-1` (score=0.6453) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (myopathy). This condit...
4. `medquad_ans_0004401-5` (score=0.4803) - How might neutral lipid storage disease with myopathy be treated? To date, there is no treatment for the underlying metabolic problem. Current therapies include adhering to strict dietary guidelines and utilizing treatments focused on th...
5. `medquad_ans_0004401-3` (score=0.3985) - What causes neutral lipid storage disease with myopathy? Neutral lipid storage disease with myopathy is caused by mutations in the PNPLA2 gene. This gene provides instructions for making an enzyme called adipose triglyceride lipase (ATGL...

**Answer**:

Neutral lipid storage disease with myopathy is a rare condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. This abnormal accumulation leads to muscle weakness (myopathy) due to fat buildup in muscle tissue. Other possible features include a fatty liver, cardiomyopathy (weakened and enlarged heart), pancreatitis, hypothyroidism, and type 2 diabetes mellitus. The condition is caused by mutations in the PNPLA2 gene, which results in impaired function of the adipose triglyceride lipase (ATGL) enzyme, reducing the breakdown of triglycerides and leading to their accumulation. It is inherited in an autosomal recessive pattern and currently has no treatment to correct the underlying metabolic problem; management focuses on dietary guidelines and symptom-related treatments. A recent study suggests potential benefit from bezafibrate in reducing lipid accumulation and improving fat oxidative capacity, though further research is needed [medquad_ans

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Top docs**:

1. `medquad_ans_0006546-1` (score=0.7000) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. Cases of Zika virus ...
2. `medquad_ans_0000981-1__dup1` (score=0.6338) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been reports that the virus h...
3. `medquad_ans_0000193-1__dup2` (score=0.6299) - Microcephaly is a medical condition in which the circumference of the head is smaller than normal because the brain has not developed properly or has stopped growing. Microcephaly can be present at birth or it may develop in the first fe...
4. `medquad_ans_0006546-2` (score=0.4397) - How might a Zika virus infection be treated? There is no vaccine to prevent Zika virus infections, nor is there a specific medicine to treat Zika. Individuals infected with the Zika virus should get plenty of rest, drink fluids, and take...
5. `medquad_ans_0000174-1__dup3` (score=0.2626) - Chikungunya is a virus that spread by the same kinds of mosquitoes that spread dengue and Zika virus. Rarely, it can spread from mother to newborn around the time of birth. It may also possibly spread through infected blood. There have b...

**Answer**:

Zika virus is a virus spread primarily through the bite of an infected mosquito. It can also be transmitted from a pregnant mother to her child during pregnancy or around the time of birth, through sexual contact with an affected male partner, and through blood transfusions. The illness associated with Zika virus infection is usually mild, with symptoms such as fever, rash, joint pain, and conjunctivitis, which typically last for several days to a week. In rare cases, Zika virus infection has been linked to serious birth defects, including microcephaly and other brain anomalies, as well as Guillain-Barre syndrome. There is currently no vaccine or medicine to treat Zika virus infection, and prevention measures include avoiding mosquito bites, postponing travel to affected areas, and taking precautions for sexual transmission [medquad_ans_0006546-1][medquad_ans_0000981-1__dup1].

**Citations detected**: `medquad_ans_0006546-1`, `medquad_ans_0000981-1__dup1`
**Supported citations**: `medquad_ans_0006546-1`, `medquad_ans_0000981-1__dup1`
**Unsupported citations**: _none_

## Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Top docs**:

1. `medquad_ans_0000019-1__dup2` (score=0.7000) - Broken heart syndrome is a condition in which extreme stress can lead to heart muscle failure. The failure is severe, but often short-term. Most people who experience broken heart syndrome think they may be having a heart attack, a more ...
2. `medquad_ans_0000019-3__dup2` (score=0.6533) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack, what is happening ...
3. `medquad_ans_0000019-8` (score=0.3266) - Researchers are still learning about broken heart syndrome, and no treatments have been shown to prevent it. For people who have experienced the condition, the risk of recurrence is low. An emotionally upsetting or serious physical event...
4. `medquad_ans_0000029-1__dup3` (score=0.2904) - Cardiomyopathy refers to diseases of the heart muscle. These diseases have many causes, signs and symptoms, and treatments. In cardiomyopathy, the heart muscle becomes enlarged, thick, or rigid. In rare cases, the muscle tissue in the he...
5. `medquad_ans_0000019-4__dup2` (score=0.2443) - Broken heart syndrome affects women more often than men. Often, people who experience broken heart syndrome have previously been healthy. Research shows that the traditional risk factors for heart disease may not apply to broken heart sy...

**Answer**:

The cause of broken heart syndrome is not fully known. In most cases, symptoms are triggered by extreme emotional or physical stress, such as intense grief, anger, or surprise. Researchers think that the stress releases hormones that stun the heart and affect its ability to pump blood to the body. [medquad_ans_0000019-1__dup2]

**Citations detected**: `medquad_ans_0000019-1__dup2`
**Supported citations**: `medquad_ans_0000019-1__dup2`
**Unsupported citations**: _none_

## Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Top docs**:

1. `medquad_ans_0000005-1__dup5` (score=0.7000) - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. An autoimmune disease is a disorder in which the bodys immune system attacks the bodys own cells and organs. Normally, t...
2. `medquad_ans_0000005-9__dup1` (score=0.6679) - - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. - Hashimotos disease often leads to reduced thyroid function, or hypothyroidism. Hypothyroidism is a disorder that occ...
3. `medquad_ans_0000009-3__dup4` (score=0.6495) - Hypothyroidism has several causes, including - Hashimotos disease - thyroiditis, or inflammation of the thyroid - congenital hypothyroidism, or hypothyroidism that is present at birth - surgical removal of part or all of the thyroid - ra...
4. `medquad_ans_0002735-1` (score=0.6139) - Hashimoto's encephalitis (HE) is a condition characterized by onset of confusion with altered level of consciousness; seizures; and jerking of muscles (myoclonus). Psychosis, including visual hallucinations and paranoid delusions, has al...
5. `medquad_ans_0000005-3__dup4` (score=0.4580) - Many people with Hashimotos disease have no symptoms at first. As the disease slowly progresses, the thyroid usually enlarges and may cause the front of the neck to look swollen. The enlarged thyroid, called a goiter, may create a feelin...

**Answer**:

Hashimoto thyroiditis is an autoimmune disease in which the body's immune system attacks the thyroid gland, causing inflammation and interfering with its ability to produce thyroid hormones. This condition is also known as chronic lymphocytic thyroiditis or autoimmune thyroiditis. Large numbers of white blood cells called lymphocytes accumulate in the thyroid and produce antibodies that initiate the autoimmune process. Hashimoto thyroiditis often leads to reduced thyroid function, or hypothyroidism, which occurs when the thyroid does not make enough thyroid hormone for the body's needs. Thyroid hormones regulate metabolism and affect nearly every organ in the body. Hashimoto thyroiditis is the most common cause of hypothyroidism in the United States and is much more common in women than men, typically appearing between 30 and 50 years of age [medquad_ans_0000005-1__dup5]medquad_ans_0000005-9\_\_

**Citations detected**: `medquad_ans_0000005-1__dup5`
**Supported citations**: `medquad_ans_0000005-1__dup5`
**Unsupported citations**: _none_

## Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Top docs**:

1. `medquad_ans_0001043-1` (score=0.7000) - Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). In affected individuals, harmful amounts of lipids accumulate in the spleen, liver, bone marrow, small i...
2. `medquad_ans_0006463-1` (score=0.6972) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include enlarged liver and ...
3. `medquad_ans_0000002-3__dup1` (score=0.4793) - Wolmans disease is usually fatal by age 1. The onset and course of cholesteryl ester storage disease varies, and individuals may live into adulthood.
4. `medquad_ans_0001043-5` (score=0.2898) - These resources address the diagnosis or management of Wolman disease: - Genetic Testing Registry: Lysosomal acid lipase deficiency These resources from MedlinePlus offer information about the diagnosis and management of various health c...
5. `medquad_ans_0006463-2` (score=0.2323) - What are the signs and symptoms of Wolman disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Wolman disease. If the information is available, the table below includes how often the symptom is seen...

**Answer**:

Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). It is a type of lysosomal storage disorder caused by mutations in the LIPA gene and is inherited in an autosomal recessive manner. The condition leads to a buildup of lipids in body organs such as the spleen, liver, bone marrow, small intestine, adrenal glands, and lymph nodes, along with calcium deposits in the adrenal glands. Infants are typically healthy at birth but soon develop symptoms including enlarged liver and spleen (hepatosplenomegaly), poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor nutrient absorption. The disease is severe and life-threatening, with most affected individuals not surviving past early childhood; mortality typically occurs by age 1 [medquad_ans_0001043-1]medquad_ans

**Citations detected**: `medquad_ans_0001043-1`
**Supported citations**: `medquad_ans_0001043-1`
**Unsupported citations**: _none_

## Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Top docs**:

1. `medquad_ans_0002753-3` (score=0.7000) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma and occupational viny...
2. `medquad_ans_0002753-1` (score=0.5925) - Hemangiopericytoma is a term used to described a group of tumors that are derived from pericytes, the cells normally arranged along specific types of blood vessels called capillaries and venules. These types of tumors are typically slow-...
3. `medquad_ans_0002750-1` (score=0.3155) - The term hemangioendothelioma describes several types of vascular neosplasms and includes both non-cancerous (benign) and cancerous (malignant) growths. The term has also been applied to those that show "borderline" behavior, intermediat...
4. `medquad_ans_0002753-2` (score=0.2125) - What are the signs and symptoms of Hemangiopericytoma? The Human Phenotype Ontology provides the following list of signs and symptoms for Hemangiopericytoma. If the information is available, the table below includes how often the symptom...
5. `medquad_ans_0002826-1` (score=0.2025) - Hereditary hemorrhagic telangiectasia (HHT) is an inherited disorder of the blood vessels that can cause excessive bleeding. People with this condition can develop abnormal blood vessels called arteriovenous malformations (AVMs) in sever...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a possible relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Citations detected**: `medquad_ans_0002753-3`
**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

## Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Top docs**:

1. `medquad_ans_0000396-1__dup2` (score=0.7000) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of the brain. Some geneti...
2. `medquad_ans_0000588-3` (score=0.5058) - Mutations in the MT-ND1, MT-ND4, MT-ND4L, or MT-ND6 gene can cause LHON. These genes are found in the DNA of cellular structures called mitochondria, which convert the energy from food into a form that cells can use. Although most DNA is...
3. `medquad_ans_0000181-3__dup2` (score=0.4912) - Alagille syndrome is caused by a gene mutation, or defect. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the DNA sequence that makes up a gene. DNA, or deoxyribonucleic acid, is the ...
4. `medquad_ans_0004086-1` (score=0.4745) - Mitochondrial genetic disorders refer to a group of conditions that affect the mitochondria (the structures in each cell of the body that are responsible for making energy). People with these conditions can present at any age with almost...
5. `medquad_ans_0000398-1__dup1` (score=0.4432) - Genes are the building blocks of heredity. They are passed from parent to child. They hold DNA, the instructions for making proteins. Proteins do most of the work in cells. They move molecules from one place to another, build structures,...

**Answer**:

The provided context does not contain information about genetic changes related to microphthalmia. Therefore, it is not possible to answer the question based on the given text. [no context]

**Citations detected**: `no context`
**Supported citations**: _none_
**Unsupported citations**: `no context`

## Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Top docs**:

1. `medquad_ans_0000028-13` (score=0.7000) - Gout is a form of arthritis that frequently affects joints in the lower part of the body such as the knees, ankles, or toes. The affected joint may become swollen, red, or warm. Attacks usually occur at night. Sometime during the course ...
2. `medquad_ans_0000028-1__dup4` (score=0.5677) - Sudden, Intense Joint Pain Gout is a form of arthritis that causes the sudden onset of intense pain and swelling in the joints, which also may be warm and red. Attacks frequently occur at night and can be triggered by stressful events, a...
3. `medquad_ans_0000409-1__dup1` (score=0.5562) - Gout is a common, painful form of arthritis. It causes swollen, red, hot and stiff joints. Gout happens when uric acid builds up in your body. Uric acid comes from the breakdown of substances called purines. Purines are in your body's ti...
4. `medquad_ans_0000028-8` (score=0.5272) - Gout is a form of arthritis that causes sudden onset of intense pain and swelling in the joints, which also may be warm and red. Attacks frequently occur at night and can be triggered by stressful events, alcohol or drugs, or the presenc...
5. `medquad_ans_0000028-18` (score=0.4542) - The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH has more information about gout. Check out the information here. Also, see "Key Words" related to gout.

**Answer**:

The symptoms of gout include swollen, red, or warm joints, sudden and intense joint pain, attacks that develop in a day, and arthritis affecting only one joint, often the toe, ankle, or knee. Hyperuricemia—high levels of uric acid in the body—is associated with gout, as is the presence of uric acid crystals in joint fluid. More than one attack of acute arthritis may occur, and attacks frequently happen at night [medquad_ans_0000028-13]. [medquad_ans_0000028-1__dup4]

**Citations detected**: `medquad_ans_0000028-13`, `medquad_ans_0000028-1__dup4`
**Supported citations**: `medquad_ans_0000028-13`, `medquad_ans_0000028-1__dup4`
**Unsupported citations**: _none_

## Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Top docs**:

1. `medquad_ans_0001903-1` (score=0.7000) - Dopa-responsive dystonia (DRD) is an inherited type of dystonia that typically begins during childhood but may begin in adolescence or adulthood. Depending on the specific type of DRD, specific symptoms can vary. Features can range from ...
2. `medquad_ans_0000307-1__dup1` (score=0.5420) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a group of muscles or ...
3. `medquad_ans_0003335-1` (score=0.5174) - Juvenile-onset dystonia is a form of dystonia, which is a movement disorder characterized by involuntary muscle contractions that cause repetitive movements and/or abnormal postures. The severity and frequency of the movements vary signi...
4. `medquad_ans_0000112-1__dup1` (score=0.4622) - The dystonias are movement disorders in which sustained muscle contractions cause twisting and repetitive movements or abnormal postures. The movements, which are involuntary and sometimes painful, may affect a single muscle; a group of ...
5. `medquad_ans_0002400-1` (score=0.4043) - Focal dystonia is a movement disorder that is localized to a specific part of the body. The dystonias are a group of movement problems characterized by involuntary, sustained muscle contractions, tremors, and other uncontrolled movements...

**Answer**:

The provided context does not mention "Dystonia 8" or its specific cause. Therefore, based on the available information, the cause of Dystonia 8 cannot be determined. [doc_id]

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Top docs**:

1. `medquad_ans_0000612-2` (score=0.7000) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unknown.
2. `medquad_ans_0003719-1` (score=0.5353) - Lysinuric protein intolerance is a metabolic disorder caused by the body's inability to digest and use the amino acids lysine, arginine, and ornithine. Because the body cannot effectively break down these amino acids, which are found in ...
3. `medquad_ans_0000195-2` (score=0.1255) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0001101-2` (score=0.0786) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
5. `medquad_ans_0000136-2` (score=0.0721) - Researchers estimate that BOR/BO syndrome affects about 1 in 40,000 people.

**Answer**:

Lysinuric protein intolerance affects an estimated 1 in 40,000 to 50,000 people [medquad_ans_0000195-2]. In specific populations, the incidence is 1 in 60,000 in Finland and 1 in 57,000 in Japan [medquad_ans_0000612-2]. The prevalence is estimated to be 1 in 7,500 to 20,000 people [medquad_ans_0001101-2]. However, the exact global incidence is unknown [medquad_ans_0000612-2].

**Citations detected**: `medquad_ans_0000195-2`, `medquad_ans_0000612-2`, `medquad_ans_0001101-2`
**Supported citations**: `medquad_ans_0000195-2`, `medquad_ans_0000612-2`, `medquad_ans_0001101-2`
**Unsupported citations**: _none_

## Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Top docs**:

1. `medquad_ans_0000170-10` (score=0.7000) - - Renal artery stenosis (RAS) is the narrowing of one or both renal arteries. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood vessel that carries blood from the heart to arteries througho...
2. `medquad_ans_0000150-4__dup2` (score=0.6333) - Most people with high blood pressure do not have symptoms. In rare cases, high blood pressure can cause headaches. Kidney disease also does not have symptoms in the early stages. A person may have swelling called edema, which happens whe...
3. `medquad_ans_0000170-1__dup3` (score=0.6301) - Renal artery stenosis is the narrowing of one or both renal arteries. Renal means kidney and stenosis means narrowing. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood vessel that carries ...
4. `medquad_ans_0000145-3__dup3` (score=0.6019) - The symptoms of Goodpasture syndrome may initially include fatigue, nausea, vomiting, and weakness. The lungs are usually affected before or at the same time as the kidneys, and symptoms can include shortness of breath and coughing, some...
5. `medquad_ans_0000071-5__dup1` (score=0.5568) - Because diagnosis is based on blood pressure readings, this condition can go undetected for years, as symptoms do not usually appear until the body is damaged from chronic high blood pressure. Complications of High Blood Pressure When bl...

**Answer**:

Renal artery stenosis (RAS) often has no symptoms until it becomes severe. The first symptoms are usually either high blood pressure or decreased kidney function, or both [medquad_ans_0000170-10]. High blood pressure in RAS is typically detected without symptoms in most individuals, though in rare cases, it may cause headaches [medquad_ans_0000150-4__dup2]. Decreased kidney function may present with symptoms such as edema (swelling in the legs, feet, ankles, hands, or face), appetite loss, nausea, vomiting, drowsiness or fatigue, trouble concentrating, sleep problems, increased or decreased urination, generalized itching or numbness, dry skin, headaches, weight loss, darkened skin, muscle cramps, shortness of breath, and chest pain [medquad_ans_0000150-4\_\_dup

**Citations detected**: `medquad_ans_0000170-10`, `medquad_ans_0000150-4__dup2`
**Supported citations**: `medquad_ans_0000170-10`, `medquad_ans_0000150-4__dup2`
**Unsupported citations**: _none_

## Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Top docs**:

1. `medquad_ans_0005502-5` (score=0.7000) - How might Schnitzler syndrome be treated? The treatment of Schnitzler syndrome is aimed at alleviating the signs and symptoms associated with the condition. The following medications have been used with variable success: Nonsteroidal ant...
2. `medquad_ans_0000712-5` (score=0.4692) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis - Genetic Testing Reg...
3. `medquad_ans_0004303-4` (score=0.4353) - Is there a treatment for Naegeli syndrome? Treatment for Naegeli syndrome is based on an individual's symptoms. Dry skin can be moisturized with creams. To avoid overheating, affected individuals should wear appropriate clothing and use ...
4. `medquad_ans_0000712-1` (score=0.4043) - Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis (NFJS/DPR) represents a rare type of ectodermal dysplasia, a group of about 150 conditions characterized by abnormal development of ectodermal tissues includin...
5. `medquad_ans_0002049-6` (score=0.4032) - How might Ehlers-Danlos syndrome be treated? There is no specific cure for Ehlers-Danlos syndrome (EDS). The treatment and management is focused on preventing serious complications and relieving associated signs and symptoms. Because the...

**Answer**:

Treatment for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis (NFJS/DPR) is based on an individual's symptoms. Dry skin can be moisturized with creams. To avoid overheating, affected individuals should wear appropriate clothing and use wet dressings. Dental care is needed to treat cavities and tooth loss [medquad_ans_0004303-4].

**Citations detected**: `medquad_ans_0004303-4`
**Supported citations**: `medquad_ans_0004303-4`
**Unsupported citations**: _none_

## Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.7000) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is available, the tabl...
2. `medquad_ans_0000353-5` (score=0.4125) - These resources address the diagnosis or management of familial hyperaldosteronism: - Genetic Testing Registry: Familial hyperaldosteronism type 1 - Genetic Testing Registry: Familial hyperaldosteronism type 3 - Hormone Health Network: A...
3. `medquad_ans_0005138-2` (score=0.3917) - What are the signs and symptoms of Pseudohypoaldosteronism type 2? The most common symptom of pseudohypoaldosteronism type 2 is high blood pressure in adolescents or young adults. In its most severe form, it is associated with muscle wea...
4. `medquad_ans_0000353-1` (score=0.3239) - Familial hyperaldosteronism is a group of inherited conditions in which the adrenal glands, which are small glands located on top of each kidney, produce too much of the hormone aldosterone. Aldosterone helps control the amount of salt r...
5. `medquad_ans_0002593-2` (score=0.3133) - What are the signs and symptoms of Glucocorticoid-remediable aldosteronism? The Human Phenotype Ontology provides the following list of signs and symptoms for Glucocorticoid-remediable aldosteronism. If the information is available, the ...

**Answer**:

The symptoms of Familial hyperaldosteronism type III include hypercalciuria (5%), metabolic acidosis (5%), polydipsia (5%), polyuria (5%), adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

**Citations detected**: `medquad_ans_0002259-1`
**Supported citations**: `medquad_ans_0002259-1`
**Unsupported citations**: _none_

## Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Top docs**:

1. `medquad_ans_0004530-2` (score=0.7000) - Is olivopontocerebellar atrophy inherited? Olivopontocerebellar atrophy (OPCA) may be associated with conditions that are inherited (genetic), or it may occur sporadically. Genetic forms of OPCA may be inherited in an autosomal dominant,...
2. `medquad_ans_0004750-4` (score=0.2780) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is passed down through fami...
3. `medquad_ans_0000673-4` (score=0.2271) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
4. `medquad_ans_0000700-4` (score=0.1500) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one affected parent. Ap...
5. `medquad_ans_0000786-4` (score=0.1455) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one affected parent. Ho...

**Answer**:

The provided context does not contain information about whether Spinocerebellar ataxia 11 is inherited. Therefore, based on the given context, it is not possible to determine the inheritance pattern of Spinocerebellar ataxia 11. [no relevant context]

**Citations detected**: `no relevant context`
**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

## Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Top docs**:

1. `medquad_ans_0000503-2` (score=0.7000) - Hypochromic microcytic anemia with iron overload is likely a rare disorder; at least five affected families have been reported in the scientific literature.
2. `medquad_ans_0003040-1` (score=0.6451) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload. If the information ...
3. `medquad_ans_0000503-1` (score=0.5063) - Hypochromic microcytic anemia with iron overload is a condition that impairs the normal transport of iron in cells. Iron is an essential component of hemoglobin, which is the substance that red blood cells use to carry oxygen to cells an...
4. `medquad_ans_0000503-5` (score=0.4948) - These resources address the diagnosis or management of hypochromic microcytic anemia with iron overload: - Genetic Testing Registry: Hypochromic microcytic anemia with iron overload These resources from MedlinePlus offer information abou...
5. `medquad_ans_0000185-10` (score=0.3784) - - Hemochromatosis is the most common form of iron overload disease. Too much iron in the body causes hemochromatosis. - Inherited genetic defects cause primary hemochromatosis. - Primary hemochromatosis mainly affects Caucasians of North...

**Answer**:

The symptoms of Hypochromic microcytic anemia with iron overload include pale skin (pallor), tiredness (fatigue), and slow growth. Additionally, the condition leads to anemia characterized by abnormally small (microcytic) and pale (hypochromic) red blood cells. Iron accumulation in the liver can impair its function over time, with liver problems typically becoming apparent in adolescence or early adulthood [medquad_ans_0000503-1].

**Citations detected**: `medquad_ans_0000503-1`
**Supported citations**: `medquad_ans_0000503-1`
**Unsupported citations**: _none_

## Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Top docs**:

1. `medquad_ans_0000698-1__dup2` (score=0.7000) - Pemphigus is an autoimmune disorder. If you have it, your immune system attacks healthy cells in your skin and mouth, causing blisters and sores. No one knows the cause. Pemphigus does not spread from person to person. It does not appear...
2. `medquad_ans_0000300-4` (score=0.0000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000676-4` (score=0.0000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
4. `medquad_ans_0000990-4` (score=0.0000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000685-4` (score=0.0000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...

**Answer**:

The provided context does not mention "benign chronic pemphigus" or specify whether it is inherited. While the context states that pemphigus is an autoimmune disorder and does not appear to be inherited, it also notes that "some people's genes put them more at risk for pemphigus" [medquad_ans_0000698-1__dup2]. However, this does not confirm inheritance of a specific form like "benign chronic pemphigus." Therefore, based on the available information, it cannot be determined if benign chronic pemphigus is inherited.

Answer: Insufficient context to determine if benign chronic pemphigus is inherited.

**Citations detected**: `medquad_ans_0000698-1__dup2`
**Supported citations**: `medquad_ans_0000698-1__dup2`
**Unsupported citations**: _none_

## Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Top docs**:

1. `medquad_ans_0000916-4__dup1` (score=0.7000) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is enough to cause fea...
2. `medquad_ans_0000915-1__dup1` (score=0.4015) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are...
3. `medquad_ans_0000916-1__dup1` (score=0.4015) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are...
4. `medquad_ans_0004784-4` (score=0.1435) - How is Pelizaeus-Merzbacher disease inherited?
5. `medquad_ans_0002823-3` (score=0.1312) - How is hereditary endotheliopathy with retinopathy, nephropathy, and stroke (HERNS) inherited? Hereditary endotheliopathy with retinopathy, nephropathy, and stroke (HERNS) is inherited in an autosomal dominant manner. This means that hav...

**Answer**:

Yes, Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a mutation in only one copy of the responsible gene in each cell is sufficient to cause the condition. [medquad_ans_0000916-4__dup1]

**Citations detected**: `medquad_ans_0000916-4__dup1`
**Supported citations**: `medquad_ans_0000916-4__dup1`
**Unsupported citations**: _none_

## Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Top docs**:

1. `medquad_ans_0002895-1` (score=0.7000) - Histiocytosis-lymphadenopathy plus syndrome is a group of conditions with overlapping signs and symptoms that affect many parts of the body. This group of disorders includes H syndrome, pigmented hypertrichosis with insulin-dependent dia...
2. `medquad_ans_0006446-4` (score=0.2407) - Is Williams syndrome inherited?
3. `medquad_ans_0000673-4` (score=0.0758) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
4. `medquad_ans_0002533-4` (score=0.0434) - How is Gerstmann-Straussler-Scheinker disease inherited? Gerstmann-Straussler-Scheinker disease (GSS) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of th...
5. `medquad_ans_0001030-4` (score=0.0182) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene and occur in people with no history...

**Answer**:

The genetic changes related to histiocytosis-lymphadenopathy plus syndrome are mutations in the SLC29A3 gene [medquad_ans_0002895-1].

**Citations detected**: `medquad_ans_0002895-1`
**Supported citations**: `medquad_ans_0002895-1`
**Unsupported citations**: _none_

## Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000261-4` (score=0.7000) - In most cases of cyclic vomiting syndrome, affected people have no known history of the disorder in their family. However, many affected individuals have a family history of related conditions, such as migraines, irritable bowel syndrome...
2. `medquad_ans_0000646-4` (score=0.4133) - Menkes syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy o...
3. `medquad_ans_0000139-2__dup2` (score=0.3614) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the gene on to you. You...
4. `medquad_ans_0000557-4` (score=0.2809) - This condition is generally not inherited but arises from mutations in the body's cells that occur after conception. This alteration is called a somatic mutation and is present only in certain cells. Rarely, this condition is inherited i...
5. `medquad_ans_0003743-4` (score=0.0405) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition which suggests that i...

**Answer**:

The provided context does not contain information about whether Noonan syndrome is inherited. Therefore, based on the given documents, it is not possible to determine if Noonan syndrome is inherited. [no_context]

**Citations detected**: `no_context`
**Supported citations**: _none_
**Unsupported citations**: `no_context`

## Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Top docs**:

1. `medquad_ans_0001093-2` (score=0.7000) - 3-methylglutaconyl-CoA hydratase deficiency is a rare disorder; at least 20 cases have been reported in the scientific literature.
2. `medquad_ans_0001093-5` (score=0.3967) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Acidosis These resource...
3. `medquad_ans_0001093-1` (score=0.2997) - 3-methylglutaconyl-CoA hydratase deficiency is an inherited condition that causes neurological problems. Beginning in infancy to early childhood, children with this condition often have delayed development of mental and motor skills (psy...
4. `medquad_ans_0001093-3` (score=0.2412) - Mutations in the AUH gene cause 3-methylglutaconyl-CoA hydratase deficiency. This gene provides instructions for producing 3-methylglutaconyl-CoA hydratase, an enzyme that is involved in breaking down a protein building block (amino acid...
5. `medquad_ans_0006372-6` (score=0.1608) - How might VLCAD deficiency be treated? Management of VLCAD deficiency depends on many factors, including the form of the condition and the specific signs and symptoms present. For example, people affected by the severe forms of the condi...

**Answer**:

The provided context does not mention any specific treatments for 3-methylglutaconyl-CoA hydratase deficiency. While it discusses diagnostic resources and symptoms, including elevated levels of 3-methylglutaconic acid and metabolic acidosis, there is no information on therapeutic interventions or management strategies. [medquad_ans_0001093-5] lists resources related to diagnosis and management, but does not specify treatments. Therefore, based on the given context, the treatments for this condition cannot be determined.

**Citations detected**: `medquad_ans_0001093-5`
**Supported citations**: `medquad_ans_0001093-5`
**Unsupported citations**: _none_

## Example 61 (`0003000-1`)

**Question**: What is (are) Hyperkalemic periodic paralysis ?

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.7000) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0000494-5` (score=0.5834) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis - Genetic Testing Re...
3. `medquad_ans_0000126-1__dup1` (score=0.5245) - Familial periodic paralyses are a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. They are characterized by episodes in which the affected muscles become sl...
4. `medquad_ans_0000494-1` (score=0.4672) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the...
5. `medquad_ans_0003000-1` (score=0.4546) - Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms...

**Answer**:

Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms and legs. Episodes tend to increase in frequency until mid-adulthood, after which they occur less frequently. Factors that can trigger attacks include rest after exercise, potassium-rich foods such as bananas and potatoes, stress, fatigue, alcohol, pregnancy, exposure to cold temperatures, certain medications, and periods without food (fasting). Muscle strength usually returns to normal between attacks, although many affected people continue to experience mild stiffness (myotonia), particularly in [medquad_ans_0000126-1__dup1]. It is estimated to affect 1 in 200,000 people [medquad_ans_0000494-2].

**Citations detected**: `medquad_ans_0000126-1__dup1`, `medquad_ans_0000494-2`
**Supported citations**: `medquad_ans_0000126-1__dup1`, `medquad_ans_0000494-2`
**Unsupported citations**: _none_

## Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Top docs**:

1. `medquad_ans_0000225-5` (score=0.7000) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of extraocular muscles...
2. `medquad_ans_0000225-1` (score=0.4995) - Congenital fibrosis of the extraocular muscles is a disorder that affects the muscles that surround the eyes. These muscles control eye movement and the position of the eyes (for example, looking straight ahead). Congenital fibrosis of t...
3. `medquad_ans_0000225-2` (score=0.4962) - CFEOM1 is the most common form of congenital fibrosis of the extraocular muscles, affecting at least 1 in 230,000 people. CFEOM1 and CFEOM3 have been reported worldwide, whereas CFEOM2 has been seen in only a few families of Turkish, Sau...
4. `medquad_ans_0002381-3` (score=0.2794) - How might Filippi syndrome be treated? The treatment of Filippi syndrome is directed toward the specific symptoms that are apparent in each individual. Treatment may require the coordinated efforts of a team of medical professionals who ...
5. `medquad_ans_0002374-1` (score=0.1983) - Fibrous dysplasia is a skeletal disorder that is characterized by the replacement of normal bone with fibrous bone tissue. It may involve one bone (monostotic) or multiple bones (polyostotic). Fibrous dysplasia can affect any bone in the...

**Answer**:

The provided context does not specify treatments for congenital fibrosis of the extraocular muscles. While it mentions resources related to diagnosis and management, such as MedlinePlus Encyclopedia entries on extraocular muscle function testing, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care, no explicit details on treatments for congenital fibrosis of the extraocular muscles are given [medquad_ans_0000225-5]. Therefore, based on the available information, the specific treatments are not outlined.

**Citations detected**: `medquad_ans_0000225-5`
**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

## Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Top docs**:

1. `medquad_ans_0000152-1__dup1` (score=0.7000) - Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. Symptoms of the disease, which gets progressively worse, include u...
2. `medquad_ans_0002944-1` (score=0.6451) - Huntington disease (HD) is an inherited condition that causes progressive degeneration of neurons in the brain. Signs and symptoms usually develop between ages 35 to 44 years and may include uncontrolled movements, loss of intellectual a...
3. `medquad_ans_0000491-1__dup2` (score=0.6308) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms of HD may include u...
4. `medquad_ans_0000152-3__dup1` (score=0.5663) - Huntingtons disease causes disability that gets worse over time. People with this disease usually die within 15 to 20 years following diagnosis. At this time, no treatment is available to slow, stop or reverse the course of HD.
5. `medquad_ans_0000489-1` (score=0.5118) - Huntington disease is a progressive brain disorder that causes uncontrolled movements, emotional problems, and loss of thinking ability (cognition). Adult-onset Huntington disease, the most common form of this disorder, usually appears i...

**Answer**:

Huntington's disease (HD) is an inherited disorder that causes progressive degeneration of brain cells, called neurons, in motor control regions of the brain and other areas. It is caused by a mutation in the gene for a protein called huntingtin, leading to an abnormal repetition of cytosine, adenine, and guanine (CAG) building blocks in DNA. The disease typically begins between ages 30 and 50, with an earlier onset form called juvenile HD occurring under age 20. Symptoms include uncontrolled movements (chorea), abnormal body postures, impaired coordination, slurred speech, difficulty feeding and swallowing, changes in behavior, emotion, judgment, and cognition. HD is inherited in an autosomal dominant manner, meaning each child of a parent with HD has a 50-50 chance of inheriting the mutated gene. A person who inherits the HD gene will eventually develop the disease. Diagnosis is aided

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.7000) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the severe form has been o...
2. `medquad_ans_0002601-2` (score=0.3928) - What are the signs and symptoms of Glutamate formiminotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Glutamate formiminotransferase deficiency. If the information is available, ...
3. `medquad_ans_0002601-1` (score=0.3275) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form have minor delays i...
4. `medquad_ans_0000416-1` (score=0.3103) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with the mild form of glu...
5. `medquad_ans_0000416-5` (score=0.3002) - These resources address the diagnosis or management of glutamate formiminotransferase deficiency: - Baby's First Test - Genetic Testing Registry: Glutamate formiminotransferase deficiency These resources from MedlinePlus offer informatio...

**Answer**:

Approximately 20 affected individuals have been identified with glutamate formiminotransferase deficiency [medquad_ans_0000416-2].

**Citations detected**: `medquad_ans_0000416-2`
**Supported citations**: `medquad_ans_0000416-2`
**Unsupported citations**: _none_

## Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Top docs**:

1. `medquad_ans_0000738-2` (score=0.7000) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
2. `medquad_ans_0000738-1` (score=0.1737) - Noonan syndrome is a condition that affects many areas of the body. It is characterized by mildly unusual facial characteristics, short stature, heart defects, bleeding problems, skeletal malformations, and many other signs and symptoms....
3. `medquad_ans_0004452-2` (score=0.1621) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) are generally standa...
4. `medquad_ans_0004456-3` (score=0.1621) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) are generally standa...
5. `medquad_ans_0004453-3` (score=0.1621) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) are generally standa...

**Answer**:

Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people [medquad_ans_0000738-2].

**Citations detected**: `medquad_ans_0000738-2`
**Supported citations**: `medquad_ans_0000738-2`
**Unsupported citations**: _none_

## Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Top docs**:

1. `medquad_ans_0000339-5__dup1` (score=0.7000) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, e...
2. `medquad_ans_0000339-1__dup3` (score=0.6798) - Also called crab lice or "crabs," pubic lice are parasitic insects found primarily in the pubic or genital area of humans. Pubic lice infestation is found worldwide and occurs in all races, ethnic groups, and levels of society.
3. `medquad_ans_0000339-2__dup1` (score=0.5220) - Pubic ("crab") lice infestation is found worldwide and occurs in all races and ethnic groups and in all levels of society. Pubic lice usually are spread through sexual contact and are most common in adults. Occasionally pubic lice may be...
4. `medquad_ans_0000557-1__dup2` (score=0.5176) - Summary : Lice are parasitic insects that can be found on people's heads and bodies. They survive by feeding on human blood. Lice found on each area of the body are different from each other. The three types of lice that live on humans a...
5. `medquad_ans_0000339-7` (score=0.4609) - Pubic ("crab") lice most commonly are spread directly from person to person by sexual contact. Pubic lice very rarely may be spread by clothing, bedding, or a toilet seat. The following are steps that can be taken to help prevent and con...

**Answer**:

Pubic lice infestation is diagnosed by finding a “crab” louse or eggs (nits) on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp) [medquad_ans_0000339-5__dup1]. Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Citations detected**: `medquad_ans_0000339-5__dup1`
**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

## Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Top docs**:

1. `medquad_ans_0005571-3` (score=0.7000) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of the disease, and other...
2. `medquad_ans_0005571-1` (score=0.4042) - Shapiro syndrome is a rare disease affecting about 50 people worldwide that is typically characterized by recurrent episodes of excessive sweating and hypothermia and the agenesis of the corpus callosum. The duration and frequency of the...
3. `medquad_ans_0000860-2__dup1` (score=0.1696) - How might Brown-Sequard syndrome be treated?
4. `medquad_ans_0005571-2` (score=0.0979) - What are the signs and symptoms of Shapiro syndrome? Shapiro syndrome generally consists of three findings: spontaneous periodic hypothermia, excessive sweating, and agenesis of the corpus callosum. However, there has been a documented c...
5. `medquad_ans_0002533-6` (score=0.0903) - How might Gerstmann-Straussler-Scheinker disease be treated? The treatment of Gerstmann-Straussler-Scheinker disease (GSS) is based on the signs and symptoms present in each person. There is currently no cure for the condition and no kno...

**Answer**:

Treatment options for Shapiro syndrome that have been attempted include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, and sympathectomy. However, these treatments have resulted in varying responses, and some have been unsuccessful or of doubtful efficacy due to the small number of diagnosed cases [medquad_ans_0005571-3]. The cause of Shapiro syndrome is unknown, and therefore, effective treatment remains challenging. It is recommended that treatment options be discussed with a healthcare provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

**Citations detected**: `medquad_ans_0005571-3`
**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

## Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Top docs**:

1. `medquad_ans_0002359-1` (score=0.7000) - FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. People with FG syndrome frequ...
2. `medquad_ans_0000373-1` (score=0.4886) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects intelligence and beh...
3. `medquad_ans_0002359-3` (score=0.4658) - How might FG syndrome be treated? Treatment is aimed at addressing the individual symptoms present in each case. This often involves care by a team of providers which may include pediatricians, neurologists, cardiologists, surgeons, gast...
4. `medquad_ans_0000373-5` (score=0.2888) - These resources address the diagnosis or management of FG syndrome: - Gene Review: Gene Review: MED12-Related Disorders - Genetic Testing Registry: FG syndrome - Genetic Testing Registry: FG syndrome 2 - Genetic Testing Registry: FG synd...
5. `medquad_ans_0000373-2` (score=0.2530) - The prevalence of FG syndrome is unknown, although several hundred cases have been reported worldwide. Researchers suspect that FG syndrome may be overdiagnosed because many of its signs and symptoms are also seen with other disorders.

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. It is characterized by intellectual disability ranging from mild to severe, hypotonia, constipation and/or anal anomalies, a distinctive facial appearance, broad thumbs and great toes, relative macrocephaly, and abnormalities of the corpus callosum. Medical problems such as heart defects, seizures, undescended testicles, and inguinal hernia have also been reported. The condition is inherited in an X-linked recessive pattern, and mutations in genes such as MED12 (FGS1), FLNA (FGS2), CASK (FGS4), UPF3B (FGS6), and BRWD3 (FGS7) are associated with the disorder [medquad_ans_0002359-1]. Affected individuals often

**Citations detected**: `medquad_ans_0002359-1`
**Supported citations**: `medquad_ans_0002359-1`
**Unsupported citations**: _none_

## Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Top docs**:

1. `medquad_ans_0006242-1` (score=0.7000) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often the symptom is seen...
2. `medquad_ans_0001335-1` (score=0.3411) - What are the signs and symptoms of Chudley Rozdilsky syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Chudley Rozdilsky syndrome. If the information is available, the table below includes how o...
3. `medquad_ans_0000676-2__dup1` (score=0.3332) - What are the symptoms of Bell's palsy?
4. `medquad_ans_0003339-2` (score=0.2878) - What are the signs and symptoms of Kallmann syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kallmann syndrome. If the information is available, the table below includes how often the symptom i...
5. `medquad_ans_0003459-2` (score=0.2378) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficulty lifting objects ...

**Answer**:

The signs and symptoms of Tukel syndrome include carpal bone aplasia, carpal synostosis, compensatory chin elevation, congenital fibrosis of extraocular muscles, nonprogressive restrictive external ophthalmoplegia, postaxial oligodactyly, ptosis, restrictive external ophthalmoplegia, syndactyly [medquad_ans_0006242-1].

**Citations detected**: `medquad_ans_0006242-1`
**Supported citations**: `medquad_ans_0006242-1`
**Unsupported citations**: _none_

## Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Top docs**:

1. `medquad_ans_0000199-1__dup1` (score=0.7000) - The mucolipidoses (ML) are a group of inherited metabolic diseases that affect the bodys ability to carry out the normal turnover of various materials within cells. In ML, abnormal amounts of carbohydrates and fatty materials (lipids) ac...
2. `medquad_ans_0000674-5` (score=0.6003) - These resources address the diagnosis or management of mucolipidosis II alpha/beta: - Gene Review: Gene Review: Mucolipidosis II - Genetic Testing Registry: I cell disease - MedlinePlus Encyclopedia: Clubfoot - MedlinePlus Encyclopedia: ...
3. `medquad_ans_0004153-1` (score=0.5796) - Mucolipidosis type 4 is a metabolic condition that affects the body's ability to process certain carbohydrates and fats. As a result, these materials accumulate in cells leading to the various signs and symptoms of the condition. Most pe...
4. `medquad_ans_0000677-5` (score=0.4722) - These resources address the diagnosis or management of mucolipidosis type IV: - Gene Review: Gene Review: Mucolipidosis IV - Genetic Testing Registry: Ganglioside sialidase deficiency - MedlinePlus Encyclopedia: Gastrin These resources f...
5. `medquad_ans_0000676-5` (score=0.3757) - These resources address the diagnosis or management of mucolipidosis III gamma: - Gene Review: Gene Review: Mucolipidosis III Gamma - Genetic Testing Registry: Mucolipidosis III Gamma - MedlinePlus Encyclopedia: Cloudy Cornea - MedlinePl...

**Answer**:

The provided context does not mention any specific treatments for mucolipidoses. While it describes the disease mechanisms, symptoms, and genetic basis, it does not include information on therapeutic interventions or management strategies. Therefore, based on the given context, treatments for mucolipidoses cannot be determined.

[medquad_ans_0000199-1__dup1]

**Citations detected**: `medquad_ans_0000199-1__dup1`
**Supported citations**: `medquad_ans_0000199-1__dup1`
**Unsupported citations**: _none_

## Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Top docs**:

1. `medquad_ans_0001157-1` (score=0.7000) - What are the signs and symptoms of Cholestasis, progressive familial intrahepatic 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Cholestasis, progressive familial intrahepatic 4. If the information ...
2. `medquad_ans_0005084-2` (score=0.6781) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis type 2. If the info...
3. `medquad_ans_0000118-1` (score=0.5566) - Benign recurrent intrahepatic cholestasis (BRIC) is characterized by episodes of liver dysfunction called cholestasis. During these episodes, the liver cells have a reduced ability to release a digestive fluid called bile. Because the pr...
4. `medquad_ans_0005084-1` (score=0.5530) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching, jaundice, failure ...
5. `medquad_ans_0000118-5` (score=0.5087) - These resources address the diagnosis or management of benign recurrent intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Benign recurrent intrahepatic cholestasis 1 - Genetic Testing Reg...

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 include autosomal recessive inheritance, cirrhosis, conjugated hyperbilirubinemia, death in childhood, diarrhea, elevated alkaline phosphatase, failure to thrive, fat malabsorption, hepatocellular carcinoma, hepatomegaly, infantile onset, intermittent jaundice, and intrahepatic cholestasis [medquad_ans_0005084-2].

**Citations detected**: `medquad_ans_0005084-2`
**Supported citations**: `medquad_ans_0005084-2`
**Unsupported citations**: _none_

## Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Top docs**:

1. `medquad_ans_0000099-2__dup2` (score=0.7000) - Nearly 18 million U.S. adultsabout one in 12have fecal incontinence.1 People of any age can have a bowel control problem, though fecal incontinence is more common in older adults. Fecal incontinence is slightly more common among women. H...
2. `medquad_ans_0000211-6` (score=0.6908) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he is more likely to dev...
3. `medquad_ans_0000517-2` (score=0.5076) - Incontinentia pigmenti is an uncommon disorder. Between 900 and 1,200 affected individuals have been reported in the scientific literature. Most of these individuals are female, but several dozen males with incontinentia pigmenti have al...
4. `medquad_ans_0000211-1__dup3` (score=0.4456) - Urinary incontinence is the loss of bladder control, resulting in the accidental leakage of urine from the body. For example, a man may feel a strong, sudden need, or urgency, to urinate just before losing a large amount of urine, called...
5. `medquad_ans_0000211-4__dup2` (score=0.3722) - Urinary incontinence in men results when the brain does not properly signal the bladder, the sphincters do not squeeze strongly enough, or both. The bladder muscle may contract too much or not enough because of a problem with the muscle ...

**Answer**:

Urinary incontinence occurs in 11 to 34 percent of older men [medquad_ans_0000211-6]. Two to 11 percent of older men report daily urinary incontinence [medquad_ans_0000211-6]. However, the exact number of individuals affected is not specified in the provided context. Therefore, a precise total number of people affected by urinary incontinence in men cannot be determined from the given information.

**Citations detected**: `medquad_ans_0000211-6`
**Supported citations**: `medquad_ans_0000211-6`
**Unsupported citations**: _none_

## Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Top docs**:

1. `medquad_ans_0000882-4` (score=0.7000) - As discussed above, Robinow syndrome can have either an autosomal recessive or an autosomal dominant pattern of inheritance. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. The parents of an ind...
2. `medquad_ans_0000882-1` (score=0.5901) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the severity of their sig...
3. `medquad_ans_0000882-2` (score=0.5738) - Both the autosomal recessive and autosomal dominant forms of Robinow syndrome are rare. Fewer than 200 people with autosomal recessive Robinow syndrome have been described in the medical literature. This form of the condition has been id...
4. `medquad_ans_0005415-1` (score=0.4560) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant Robinow syndrome. T...
5. `medquad_ans_0000882-5` (score=0.4387) - These resources address the diagnosis or management of Robinow syndrome: - Gene Review: Gene Review: Autosomal Dominant Robinow Syndrome - Gene Review: Gene Review: ROR2-Related Robinow Syndrome - Genetic Testing Registry: Robinow syndro...

**Answer**:

Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome, distinguished by the severity of signs and symptoms and by their pattern of inheritance—autosomal recessive or autosomal dominant. Autosomal recessive Robinow syndrome is characterized by skeletal abnormalities such as shortening of the long bones, brachydactyly, hemivertebrae leading to kyphoscoliosis, fused or missing ribs, and short stature. Affected individuals also have distinctive facial features, including a broad forehead, widely spaced eyes, a short nose with an upturned tip, a wide nasal bridge, and a broad, triangle-shaped mouth, often described as "fetal facies." Other features include underdeveloped genitalia, dental problems like crowded teeth and gum overgrowth, and possible kidney and heart defects. Delayed development occurs in 10 to

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 74 (`0000551-1`)

**Question**: What is (are) juvenile primary lateral sclerosis ?

**Top docs**:

1. `medquad_ans_0003331-1` (score=0.7000) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells in the brain and spi...
2. `medquad_ans_0000551-2` (score=0.6627) - Juvenile primary lateral sclerosis is a rare disorder, with few reported cases.
3. `medquad_ans_0003319-1` (score=0.5039) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. Unlike other types of...
4. `medquad_ans_0000551-1` (score=0.4983) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. The features of this disorder are caused by damage to motor neurons, which are...
5. `medquad_ans_0003331-2` (score=0.3401) - What are the signs and symptoms of Juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. Symptoms of juvenile...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. It is caused by mutations in the ALS2 gene and is inherited in an autosomal recessive pattern. The disorder damages motor neurons, which are specialized nerve cells in the brain and spinal cord that control muscle movement. Symptoms begin in early childhood and progress over a period of 15 to 20 years. Early symptoms include clumsiness, muscle weakness and spasticity in the legs, and difficulty with balance. As symptoms progress, they include weakness and stiffness in the arms and hands, slurred speech, drooling, difficulty swallowing, and an inability to walk [medquad_ans_0003331-1, medquad_ans_0000551-1, medquad_ans_0003331-2]. Few cases of this disorder have been reported [

**Citations detected**: `medquad_ans_0003331-1, medquad_ans_0000551-1, medquad_ans_0003331-2`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0003331-1, medquad_ans_0000551-1, medquad_ans_0003331-2`

## Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Top docs**:

1. `medquad_ans_0000040-9` (score=0.7000) - Leukemia is a cancer of the blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white blood cells builds u...
2. `medquad_ans_0000010-1__dup4` (score=0.4956) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. These cells crowd out th...
3. `medquad_ans_0000188-1__dup2` (score=0.4352) - Leukemia is cancer of the white blood cells. It is the most common type of childhood cancer. Your blood cells form in your bone marrow. White blood cells help your body fight infection. In leukemia, the bone marrow produces abnormal whit...
4. `medquad_ans_0000040-1__dup3` (score=0.4211) - Leukemia is a cancer of the blood cells. It is the most common type of blood cancer and affects 10 times as many adults as children. Most people diagnosed with leukemia are over 50 years old. Leukemia Starts in Bone Marrow Leukemia usual...
5. `medquad_ans_0000009-1__dup3` (score=0.3389) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. These cells crowd out th...

**Answer**:

Leukemia is a cancer of the white blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white blood cells builds up in the blood, they crowd out healthy blood cells, making it difficult for the blood to carry out its normal functions [medquad_ans_0000040-9]. Leukemia is the most common type of childhood cancer and can develop quickly or slowly, with acute leukemia being a fast-growing type and chronic leukemia growing slowly [medquad_ans_0000188-1__dup2]. In acute myeloid leukemia (AML), there are too many myeloblasts, a specific type of white blood cell, and AML is the most common type of acute leukemia in adults [medquad_ans_0000010-1\_\_

**Citations detected**: `medquad_ans_0000040-9`, `medquad_ans_0000188-1__dup2`
**Supported citations**: `medquad_ans_0000040-9`, `medquad_ans_0000188-1__dup2`
**Unsupported citations**: _none_

## Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Top docs**:

1. `medquad_ans_0000135-1__dup1` (score=0.7000) - The gangliosidoses are a group of inherited metabolic diseases caused by a deficiency of the different proteins needed to break down fatty substances called lipids. Excess buildup of these fatty materials (oils, waxes, steroids, and othe...
2. `medquad_ans_0004525-1` (score=0.6004) - Oligoastrocytoma is a brain tumor that forms when two types of cells in the brain, called oligodendrocytes and astrocytes, rapidly increase in number to form a mass. These brain cells are known as glial cells, which normally protect and ...
3. `medquad_ans_0000737-1` (score=0.4643) - Paraganglioma is a type of noncancerous (benign) tumor that occurs in structures called paraganglia. Paraganglia are groups of cells that are found near nerve cell bunches called ganglia. Paragangliomas are usually found in the head, nec...
4. `medquad_ans_0002627-1` (score=0.4589) - GM1 gangliosidosis is an inherited lysosomal storage disorder that progressively destroys nerve cells (neurons) in the brain and spinal cord. The condition may be classified into three major types based on the general age that signs and ...
5. `medquad_ans_0002629-1` (score=0.4589) - GM1 gangliosidosis is an inherited lysosomal storage disorder that progressively destroys nerve cells (neurons) in the brain and spinal cord. The condition may be classified into three major types based on the general age that signs and ...

**Answer**:

The provided context does not mention "Gangliocytoma" or any condition with that name. Therefore, based on the given information, it is not possible to answer the question.

Answer: Insufficient context [medquad_ans_0000135-1, medquad_ans_0004525-1]

**Citations detected**: `medquad_ans_0000135-1, medquad_ans_0004525-1`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000135-1, medquad_ans_0004525-1`

## Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Top docs**:

1. `medquad_ans_0000130-1__dup1` (score=0.7000) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of the tendon reflexes....
2. `medquad_ans_0004068-1` (score=0.3582) - Miller syndrome is a rare condition that mainly affects the development of the face and limbs. Characteristic features include underdeveloped cheek bones, a very small lower jaw, cleft lip and/or palate, abnormalities of the eyes, absent...
3. `medquad_ans_0000658-2` (score=0.3117) - Miller syndrome is a rare disorder; it is estimated to affect fewer than 1 in 1 million newborns. At least 30 cases have been reported in the medical literature.
4. `medquad_ans_0000130-2__dup1` (score=0.3062) - Treatment for Miller Fisher syndrome is identical to treatment for Guillain-Barr syndrome: intravenous immunoglobulin (IVIg) or plasmapheresis (a procedure in which antibodies are removed from the blood) and supportive care.
5. `medquad_ans_0004069-1` (score=0.2707) - Miller-Dieker syndrome is a genetic condition characterized by lissencephaly, typical facial features, and severe neurologic abnormalities. Symptoms may include severe intellectual disability, developmental delay, seizures, muscle stiffn...

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease considered to be a variant of Guillain-Barré syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, absence of tendon reflexes, generalized muscle weakness, and may be preceded by a viral illness. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder. Treatment is identical to that of Guillain-Barré syndrome and includes intravenous immunoglobulin (IVIg) or plasmapheresis, along with supportive care [medquad_ans_0000130-1__dup1, medquad_ans_0000130-2__dup1].

**Citations detected**: `medquad_ans_0000130-1__dup1, medquad_ans_0000130-2__dup1`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000130-1__dup1, medquad_ans_0000130-2__dup1`

## Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Top docs**:

1. `medquad_ans_0001084-1` (score=0.7000) - 22q11.2 deletion syndrome (which is also known by several other names, listed below) is a disorder caused by the deletion of a small piece of chromosome 22. The deletion occurs near the middle of the chromosome at a location designated q...
2. `medquad_ans_0006570-3` (score=0.6628) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location designated q11.2. Most pe...
3. `medquad_ans_0001087-3` (score=0.6246) - 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at a location designated 2q37. The size of the deletion var...
4. `medquad_ans_0006579-3` (score=0.6031) - What causes 2q37 deletion syndrome? 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at a location designate...
5. `medquad_ans_0001086-2` (score=0.5563) - At least 500 cases of 22q13.3 deletion syndrome are known.

**Answer**:

22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in the region designated q11.2 [medquad_ans_0001084-1].

**Citations detected**: `medquad_ans_0001084-1`
**Supported citations**: `medquad_ans_0001084-1`
**Unsupported citations**: _none_

## Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Top docs**:

1. `medquad_ans_0000062-12` (score=0.7000) - Shingles is caused by a virus called the varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores and other symptoms healed, but the virus remained. It is with you for ...
2. `medquad_ans_0000062-2__dup3` (score=0.6938) - Caused By A Virus Shingles is caused by a virus called varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores healed and the other symptoms went away, but the virus r...
3. `medquad_ans_0000062-7__dup1` (score=0.6311) - Shingles -- also called varicella-zoster -- is a painful skin disease caused by a reactivation of the chickenpox virus. It is distinctive because it affects only one side of the body. The early signs of shingles usually develop in three ...
4. `medquad_ans_0000148-1__dup2` (score=0.6244) - Shingles (herpes zoster) is an outbreak of rash or blisters on the skin that is caused by the same virus that causes chickenpox the varicella-zoster virus. The first sign of shingles is often burning or tingling pain (which can be severe...
5. `medquad_ans_0000062-11` (score=0.5243) - The most common complication of shingles is pain -- a condition called post-herpetic neuralgia (PHN). People with PHN have severe pain in the areas where they had the shingles rash, even after the rash clears up. In most patients, the pa...

**Answer**:

Shingles is caused by the varicella-zoster virus, the same virus that causes chickenpox. After recovering from chickenpox, the virus remains in the body for life, hiding in nerve cells, usually in the spine. It can become active again when the immune system weakens, which may occur due to aging, certain medical conditions (such as cancer, HIV/AIDS, or transplant operations), or stress. When reactivated, the virus travels along a nerve path (dermatome) to the skin, causing a painful rash [medquad_ans_0000062-12][medquad_ans_0000062-2__dup3].

**Citations detected**: `medquad_ans_0000062-12`, `medquad_ans_0000062-2__dup3`
**Supported citations**: `medquad_ans_0000062-12`, `medquad_ans_0000062-2__dup3`
**Unsupported citations**: _none_

## Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Top docs**:

1. `medquad_ans_0002456-1` (score=0.7000) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the table below includes h...
2. `medquad_ans_0002458-2` (score=0.3019) - What are the signs and symptoms of Fryns syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns syndrome. If the information is available, the table below includes how often the symptom is seen...
3. `medquad_ans_0000333-1` (score=0.2966) - Fabry disease is an inherited disorder that results from the buildup of a particular type of fat, called globotriaosylceramide, in the body's cells. Beginning in childhood, this buildup causes signs and symptoms that affect many parts of...
4. `medquad_ans_0002200-2` (score=0.2342) - What are the signs and symptoms of Fabry disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Fabry disease. If the information is available, the table below includes how often the symptom is seen i...
5. `medquad_ans_0002458-1` (score=0.2148) - Fryns syndrome is a condition that affects the development of many parts of the body. Signs and symptoms vary widely among affected individuals. Many affected individuals have a defect in the diaphragm muscle such as a congenital diaphra...

**Answer**:

The symptoms of Fryns Hofkens Fabry syndrome include ulnar deviation of the finger (90%), distal ulnar hypoplasia, dysplastic radii, hypoplasia of the radius, mesomelic arm shortening, radial bowing, and ulnar deviation of the hand [medquad_ans_0002456-1].

**Citations detected**: `medquad_ans_0002456-1`
**Supported citations**: `medquad_ans_0002456-1`
**Unsupported citations**: _none_

## Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Top docs**:

1. `medquad_ans_0000050-8` (score=0.7000) - Scientists are pursuing a wide range of basic and clinical studies on osteoporosis. Significant advances in preventing and treating osteoporosis continue to be made. Such advances are the direct result of research focused on - determinin...
2. `medquad_ans_0000552-5` (score=0.3764) - These resources address the diagnosis or management of juvenile primary osteoporosis: - Lucile Packard Children's Hospital at Stanford: Juvenile Osteoporosis - MedlinePlus Encyclopedia: Bone Mineral Density Test - Merck Manual Home Healt...
3. `medquad_ans_0000050-34` (score=0.3121) - A comprehensive osteoporosis treatment program includes a focus on proper nutrition, exercise, and safety issues to prevent falls that may result in fractures. In addition, your doctor may prescribe a medication to slow or stop bone loss...
4. `medquad_ans_0000050-36` (score=0.2995) - Millions of Americans are able to lead healthy, productive lives while living with osteoporosis. If you have been feeling symptoms of depressionsuch as loss of appetite, hopelessness, feeling useless and helpless, or having thoughts of s...
5. `medquad_ans_0000050-6` (score=0.2450) - Who Treats Osteoporosis? Although there is no cure for osteoporosis, it can be treated. If your doctor does not specialize in osteoporosis, he or she can refer you to a specialist. There is not one type of doctor who cares for people wit...

**Answer**:

Scientists are pursuing a wide range of basic and clinical studies on osteoporosis, focusing on determining the causes and consequences of bone loss at the cellular and tissue levels, assessing risk factors, developing new strategies to maintain and even enhance bone density and reduce fracture risk, and exploring the roles of genetics, hormones, calcium, vitamin D, drugs, and exercise on bone mass [medquad_ans_0000050-8]. These advances are directly resulting from research efforts aimed at improving prevention and treatment options for osteoporosis. For more information on ongoing research, the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH provides resources [medquad_ans_0000050-8].

**Citations detected**: `medquad_ans_0000050-8`
**Supported citations**: `medquad_ans_0000050-8`
**Unsupported citations**: _none_

## Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Top docs**:

1. `medquad_ans_0000430-1__dup1` (score=0.7000) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. At worst, they're da...
2. `medquad_ans_0000338-1__dup1` (score=0.4885) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the source. If you use th...
3. `medquad_ans_0000927-1__dup2` (score=0.2725) - Summary : It seems to happen almost every day - you hear about the results of a new medical research study. Sometimes the results of one study seem to disagree with the results of another study. It's important to be critical when reading...
4. `medquad_ans_0000011-14` (score=0.2692) - Some parents did not tell their children about receiving treatment with pituitary hGH and the possible risk of CJD. These children are now adults. Although the HHS no longer sends annual information about the problem of CJD in pituitary ...
5. `medquad_ans_0000705-1__dup1` (score=0.1731) - Summary : You've probably seen your chart at your doctor's office. In fact, you may have charts at several doctors' offices. If you've been in the hospital, you have a chart there, too. These charts are your medical records. They may be ...

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. These scams often promise quick fixes, miracle cures, ancient remedies, or secret ingredients, which may sound too good to be true. Such scams can be dangerous, waste money, and prevent individuals from accessing effective treatments. They commonly target people with serious health conditions like cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. To protect yourself, recognize red flags and consult a doctor or healthcare professional before taking unproven treatments [medquad_ans_0000430-1__dup1].

**Citations detected**: `medquad_ans_0000430-1__dup1`
**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

## Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Top docs**:

1. `medquad_ans_0000275-1__dup1` (score=0.7000) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adults. Your retina is t...
2. `medquad_ans_0000055-3__dup2` (score=0.5616) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision - dark or floating sp...
3. `medquad_ans_0000016-6` (score=0.4209) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healthy retina is necessa...
4. `medquad_ans_0000015-17` (score=0.3922) - Diabetes is a very serious disease. Over time, diabetes that is not well managed causes serious damage to the eyes, kidneys, nerves, and heart, gums and teeth. If you have diabetes, you are more likely than someone who does not have diab...
5. `medquad_ans_0000054-1__dup4` (score=0.3476) - Diabetes problems are health problems that can happen when you have diabetes. If your diabetes is not under control, you will have too much glucose\*, also called sugar, in your blood. Having too much glucose in your blood for a long time...

**Answer**:

The provided context does not contain information about symptoms of preventing diabetes problems or specifically about how to keep eyes healthy in a preventive context. The context discusses symptoms of diabetic eye problems, such as blurry or double vision, rings or flashing lights, dark or floating spots, pain or pressure in the eyes, and trouble seeing out of the corners of the eyes, but these are symptoms of existing retinopathy, not preventive measures [medquad_ans_0000275-1\_\_dup1][medquad_ans_0000055-3__dup2][medquad_ans_0000016-6]. Therefore, based on the given information, the question cannot be answered.

**Citations detected**: `medquad_ans_0000275-1__dup1`, `medquad_ans_0000055-3__dup2`, `medquad_ans_0000016-6`
**Supported citations**: `medquad_ans_0000275-1__dup1`, `medquad_ans_0000055-3__dup2`, `medquad_ans_0000016-6`
**Unsupported citations**: _none_

## Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Top docs**:

1. `medquad_ans_0003743-4` (score=0.7000) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition which suggests that i...
2. `medquad_ans_0000983-3__dup1` (score=0.6830) - Is Castleman disease inherited? Although the exact underlying cause of Castleman disease is unknown, it is thought to occur sporadically in people with no family history of the condition.
3. `medquad_ans_0006460-4` (score=0.6644) - Is Wolff-Parkinson-White syndrome inherited?
4. `medquad_ans_0000557-4` (score=0.4174) - This condition is generally not inherited but arises from mutations in the body's cells that occur after conception. This alteration is called a somatic mutation and is present only in certain cells. Rarely, this condition is inherited i...
5. `medquad_ans_0000646-4` (score=0.3577) - Menkes syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy o...

**Answer**:

The provided context does not contain information about the inheritance of Aarskog-Scott syndrome. Therefore, it is not possible to determine based on the given text whether Aarskog-Scott syndrome is inherited. [no relevant context]

**Citations detected**: `no relevant context`
**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

## Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Top docs**:

1. `medquad_ans_0006525-1` (score=0.7000) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin. People with X-linked sideroblastic anemia have mature red blood cells that are smaller than no...
2. `medquad_ans_0001062-1` (score=0.6110) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin, which is the protein that carries oxygen in the blood. People with X-linked sideroblastic anem...
3. `medquad_ans_0001062-4` (score=0.5835) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of...
4. `medquad_ans_0001063-2` (score=0.5788) - X-linked sideroblastic anemia and ataxia is a rare disorder; only a few affected families have been reported.
5. `medquad_ans_0001063-1` (score=0.5486) - X-linked sideroblastic anemia and ataxia is a rare condition characterized by a blood disorder called sideroblastic anemia and movement problems known as ataxia. This condition occurs only in males. Sideroblastic anemia results when deve...

**Answer**:

The genetic changes related to X-linked sideroblastic anemia are mutations in the ALAS2 gene. In rare cases, mutations are also found in both the HFE gene and the ALAS2 gene, resulting in a more severe form of the disorder. The condition is inherited in an X-linked recessive pattern [medquad_ans_0006525-1].

**Citations detected**: `medquad_ans_0006525-1`
**Supported citations**: `medquad_ans_0006525-1`
**Unsupported citations**: _none_

## Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Top docs**:

1. `medquad_ans_0003601-1` (score=0.7000) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of years. Onset may occu...
2. `medquad_ans_0004514-4` (score=0.2684) - How is oculopharyngeal muscular dystrophy inherited?
3. `medquad_ans_0000602-1` (score=0.2661) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifically the muscles of...
4. `medquad_ans_0001935-1` (score=0.2220) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystrophin. Boys with Duchen...
5. `medquad_ans_0001537-1` (score=0.2118) - Congenital muscular dystrophy (CMD) refers to a group of inherited conditions that affect the muscles and are present at birth or in early infancy. The severity of the condition, the associated signs and symptoms and the disease progress...

**Answer**:

Limb-girdle muscular dystrophy is a group of disorders that affect the voluntary muscles around the hips and shoulders, causing weakness and wasting of the muscles, particularly the proximal muscles such as those in the shoulders, upper arms, pelvic area, and thighs [medquad_ans_0003601-1]. The conditions are progressive, leading to a loss of muscle strength and bulk over time, with onset possibly occurring in childhood, adolescence, young adulthood, or later in life [medquad_ans_0003601-1]. Males and females are affected equally, and most forms are inherited in an autosomal recessive manner, while several rare forms follow an autosomal dominant pattern [medquad_ans_0003601-1]. The severity, age of onset, and specific features vary among the subtypes and can differ even within the same family [medquad_ans_000

**Citations detected**: `medquad_ans_0003601-1`
**Supported citations**: `medquad_ans_0003601-1`
**Unsupported citations**: _none_

## Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Top docs**:

1. `medquad_ans_0000016-6` (score=0.7000) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healthy retina is necessa...
2. `medquad_ans_0000275-1__dup1` (score=0.6791) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adults. Your retina is t...
3. `medquad_ans_0000055-3__dup2` (score=0.5370) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision - dark or floating sp...
4. `medquad_ans_0000016-1__dup3` (score=0.4616) - Can Cause Vision Loss, Blindness Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina in the back of the eye. A healthy retina is ne...
5. `medquad_ans_0000016-8` (score=0.4391) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a comprehensive dilated e...

**Answer**:

Diabetic eye problems include diabetic retinopathy, cataracts, and glaucoma. Diabetic retinopathy is a complication of diabetes that occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye, and is a leading cause of blindness [medquad_ans_0000016-6]. Cataracts involve a clouding of the lens of the eye, and surgery can help restore clear vision [medquad_ans_0000275-1__dup1]. Glaucoma occurs when pressure builds up in the eye, damaging the main nerve, and can be managed with eye drops or surgery [medquad_ans_0000275-1__dup1]. People with diabetes should have a complete eye exam every year to detect and treat these conditions early [medquad_ans_0000275-1\_\_dup

**Citations detected**: `medquad_ans_0000016-6`, `medquad_ans_0000275-1__dup1`
**Supported citations**: `medquad_ans_0000016-6`, `medquad_ans_0000275-1__dup1`
**Unsupported citations**: _none_

## Example 88 (`0006449-1`)

**Question**: What is (are) Wilson disease ?

**Top docs**:

1. `medquad_ans_0001037-2` (score=0.7000) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
2. `medquad_ans_0006449-1` (score=0.5570) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver, brain and eyes. Sig...
3. `medquad_ans_0001037-5` (score=0.4653) - These resources address the diagnosis or management of Wilson disease: - Gene Review: Gene Review: Wilson Disease - Genetic Testing Registry: Wilson's disease - MedlinePlus Encyclopedia: Wilson's disease - National Human Genome Research ...
4. `medquad_ans_0006449-3` (score=0.3999) - What causes Wilson disease? Wilson disease is caused by changes (mutations) in the ATP7B gene. This gene encodes a protein that plays an important role in the transport of copper from the liver to the rest of the body. It also helps remo...
5. `medquad_ans_0000133-1__dup3` (score=0.3753) - Wilson disease is a genetic disease that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filters extra copper and...

**Answer**:

Wilson disease is a rare inherited disorder characterized by the accumulation of copper in the body due to mutations in the ATP7B gene. This gene encodes a protein responsible for transporting copper from the liver to other parts of the body and removing excess copper from the body. When the protein does not function properly, copper builds up in tissues and organs such as the liver, brain, and eyes, leading to damage. High levels of copper are toxic and can cause chronic liver disease, central nervous system abnormalities, and psychiatric disturbances. The disorder is inherited in an autosomal recessive manner and currently has no cure, but therapies exist to reduce or control copper accumulation [medquad_ans_0006449-1][medquad_ans_0003449-3][medquad_ans_0000133-1__dup3].

**Citations detected**: `medquad_ans_0006449-1`, `medquad_ans_0003449-3`, `medquad_ans_0000133-1__dup3`
**Supported citations**: `medquad_ans_0006449-1`, `medquad_ans_0000133-1__dup3`
**Unsupported citations**: `medquad_ans_0003449-3`

## Example 89 (`0001503-2`)

**Question**: What are the symptoms of Congenital dyserythropoietic anemia type 2 ?

**Top docs**:

1. `medquad_ans_0001503-2` (score=0.7000) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to absorb too much iron, w...
2. `medquad_ans_0001504-1` (score=0.3822) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital dyserythropoietic anemia type 3. If the information is available...
3. `medquad_ans_0001503-1` (score=0.2731) - Congenital dyserythropoietic anemia type 2 (CDA II) is an inherited blood disorder characterized by mild to severe anemia. It is usually diagnosed in adolescence or early adulthood. Many affected individuals have yellowing of the skin an...
4. `medquad_ans_0005622-4` (score=0.1626) - How is sideroblastic anemia diagnosed? The principle feature of sideroblastic anemia is slowly progressive, mild, life-long anemia which often goes unnoticed. Symptoms of iron overload may lead to the discovery of this underlying disorde...
5. `medquad_ans_0000223-1` (score=0.1557) - Congenital dyserythropoietic anemia (CDA) is an inherited blood disorder that affects the development of red blood cells. This disorder is one of many types of anemia, which is a condition characterized by a shortage of red blood cells. ...

**Answer**:

The symptoms of Congenital dyserythropoietic anemia type 2 include jaundice, gallstones, an enlarged liver and spleen (splenomegaly), anemia of inadequate production, endopolyploidy on chromosome studies of bone marrow, reduced activity of N-acetylglucosaminyltransferase II, reticulocytosis, and iron overload. Iron overload can lead to complications such as an abnormal heart rhythm (arrhythmia), congestive heart failure, diabetes, and chronic liver disease (cirrhosis). Rarely, mediastinal tumors may occur [medquad_ans_0001503-2].

**Citations detected**: `medquad_ans_0001503-2`
**Supported citations**: `medquad_ans_0001503-2`
**Unsupported citations**: _none_

## Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Top docs**:

1. `medquad_ans_0000038-11` (score=0.7000) - Kidney disease -- also known as chronic kidney disease (CKD) -- occurs when kidneys can no longer remove wastes and extra water from the blood or perform other functions as they should. According to the Centers for Disease Control and Pr...
2. `medquad_ans_0000140-2__dup2` (score=0.4694) - Chronic kidney disease is kidney damage that occurs slowly over many years, often due to diabetes or high blood pressure. Once damaged, the kidneys cant filter blood as they should. This damage can cause wastes to build up in the body an...
3. `medquad_ans_0000038-12` (score=0.3596) - Kidney disease is most often caused by diabetes or high blood pressure. Each kidney contains about one million tiny filters made up of blood vessels. These filters are called glomeruli. Diabetes and high blood pressure damage these blood...
4. `medquad_ans_0000033-1__dup4` (score=0.3528) - Diabetic kidney disease, also called diabetic nephropathy, is kidney disease caused by diabetes. Even when well controlled, diabetes can lead to chronic kidney disease (CKD) and kidney failure, described as end-stage kidney disease or ES...
5. `medquad_ans_0001009-1` (score=0.2289) - Uromodulin-associated kidney disease is an inherited condition that affects the kidneys. The signs and symptoms of this condition vary, even among members of the same family. Many individuals with uromodulin-associated kidney disease dev...

**Answer**:

Kidney disease, also known as chronic kidney disease (CKD), occurs when kidneys can no longer remove wastes and extra water from the blood or perform other functions as they should [medquad_ans_0000038-11]. Chronic kidney disease is kidney damage that occurs slowly over many years, often due to diabetes or high blood pressure [medquad_ans_0000140-2__dup2]. This damage impairs the kidneys' ability to filter blood properly, leading to a buildup of wastes in the body and other health problems, including mineral and bone disorder [medquad_ans_0000140-2__dup2]. The condition is most often caused by diabetes or high blood pressure, which damage the tiny blood vessel filters in the kidneys called glomeruli, reducing their ability to filter blood effectively [medquad_ans_0000038-12]. Diabetes

**Citations detected**: `medquad_ans_0000038-11`, `medquad_ans_0000140-2__dup2`, `medquad_ans_0000038-12`
**Supported citations**: `medquad_ans_0000038-11`, `medquad_ans_0000140-2__dup2`, `medquad_ans_0000038-12`
**Unsupported citations**: _none_

## Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Top docs**:

1. `medquad_ans_0000058-23` (score=0.7000) - There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But doctors also conside...
2. `medquad_ans_0000058-6__dup1` (score=0.5903) - Choosing Treatment There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But d...
3. `medquad_ans_0000058-26` (score=0.4979) - Surgery is a common treatment for early stage prostate cancer. It is used to remove the cancer. The surgeon may remove the entire prostate -- a type of surgery called radical prostatectomy -- or, in some cases, remove only part of it. So...
4. `medquad_ans_0000036_1-6` (score=0.4887) - Key Points - There are different types of treatment for patients with prostate cancer. - Seven types of standard treatment are used: - Watchful waiting or active surveillance - Surgery - Radiation therapy and radiopharmaceutical therapy ...
5. `medquad_ans_0000058-4__dup3` (score=0.3834) - If tests show that you have cancer, you should talk with your doctor in order to make treatment decisions. Working With a Team of Specialists A team of specialists often treats people with cancer. The team will keep the primary doctor in...

**Answer**:

The treatments for prostate cancer may involve watchful waiting, surgery, radiation therapy, or hormonal therapy. Some men receive a combination of these therapies. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor, as well as the man's age, general health, and feelings about the treatments and their possible side effects [medquad_ans_0000058-23]. A cure is probable for men whose prostate cancer is diagnosed early [medquad_ans_0000058-23]. Surgery, such as radical prostatectomy, may be used to remove the cancer, and can sometimes preserve sexual function [medquad_ans_0000058-6__dup1]. Radiation therapy and hormonal therapy also have potential side effects, including impacts on sexual desire or performance and urinary function [medquad_ans_0000058-6\_\_dup1

**Citations detected**: `medquad_ans_0000058-23`, `medquad_ans_0000058-6__dup1`
**Supported citations**: `medquad_ans_0000058-23`, `medquad_ans_0000058-6__dup1`
**Unsupported citations**: _none_

## Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Top docs**:

1. `medquad_ans_0000431-1__dup1` (score=0.7000) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when you need medical care....
2. `medquad_ans_0000036-2__dup3` (score=0.4971) - Health insurance helps pay for medical care, including the cost of diabetes care. Health insurance options include the following: - private health insurance, which includes group and individual health insurance - government health insura...
3. `medquad_ans_0000036-4__dup3` (score=0.4750) - Insurance companies sell private health insurance plans. Two types of private health insurance are - Group health insurance. People may be eligible to purchase group health insurance through their employer or union or through a family me...
4. `medquad_ans_0000045-9` (score=0.3460) - Medicare Part B is medical insurance. It helps cover medical services such as doctor's services, outpatient care and other medical services that Part A doesn't cover. Part B also covers some preventive services, such as flu shots and dia...
5. `medquad_ans_0000705-1__dup1` (score=0.2814) - Summary : You've probably seen your chart at your doctor's office. In fact, you may have charts at several doctors' offices. If you've been in the hospital, you have a chart there, too. These charts are your medical records. They may be ...

**Answer**:

Yes, there is information about health insurance. Health insurance helps protect individuals from high medical care costs by providing coverage for medical expenses, including diabetes care. It is a contract between the individual and an insurance company, where the company agrees to pay part of the expenses when medical care is needed. Health insurance can be obtained through employers, where the employer often helps pay for the insurance, or through individual purchases. Private health insurance includes group and individual plans, while government health insurance options include Medicare, Medicaid, the Children's Health Insurance Program (CHIP), TRICARE, and veterans health care programs. The Affordable Care Act (ACA) prevents insurers from denying coverage or charging higher premiums to people with preexisting conditions and requires most people to have health insurance or pay a fee, with some exemptions. The ACA also expanded coverage for many people in the U.S. [medquad_ans_0000431-1__dup1]medquad_ans

**Citations detected**: `medquad_ans_0000431-1__dup1`
**Supported citations**: `medquad_ans_0000431-1__dup1`
**Unsupported citations**: _none_

## Example 93 (`0003860-1`)

**Question**: What is (are) MECP2 duplication syndrome ?

**Top docs**:

1. `medquad_ans_0003860-1` (score=0.7000) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delayed motor milestones ...
2. `medquad_ans_0000635-1` (score=0.5290) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infancy, feeding difficul...
3. `medquad_ans_0000635-2` (score=0.5026) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of all cases of intell...
4. `medquad_ans_0000635-3` (score=0.4884) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q) arm of the X chromo...
5. `medquad_ans_0003860-3` (score=0.4722) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located on the long (q) arm...

**Answer**:

MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. It is caused by an extra copy (duplication) of the MECP2 gene in each cell, which results from a duplication of genetic material on the long (q) arm of the X chromosome. The duplication size varies from 100,000 to 900,000 base pairs (100 to 900 kilobases), and the MECP2 gene is always included; other genes may also be duplicated, but their extra copies do not appear to affect the severity of the condition. Other signs and symptoms include infantile hypotonia, delayed motor milestones (such as sitting up or crawling), recurrent infections, poor or absent speech, seizures, and/or spasticity. Some individuals experience developmental regression, and approximately one third cannot walk without assistance. Recurrent respiratory tract

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Top docs**:

1. `medquad_ans_0000594-1__dup2` (score=0.7000) - Melanoma is the most serious type of skin cancer. Often the first sign of melanoma is a change in the size, shape, color, or feel of a mole. Most melanomas have a black or black-blue area. Melanoma may also appear as a new mole. It may b...
2. `medquad_ans_0000063-6__dup1` (score=0.5563) - Many Treatment Options There are many treatment options for people with melanoma and non-melanoma skin cancer. The choice of treatment depends on your age and general health, the site of the cancer, the stage of the cancer, whether or no...
3. `medquad_ans_0000017_1-5` (score=0.4758) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - How the melanoma cells look under a microscope. - The size and thickness of ...
4. `medquad_ans_0000028_2-1` (score=0.4669) - Key Points - Melanoma is a disease in which malignant (cancer) cells form in melanocytes (cells that color the skin). - There are different types of cancer that start in the skin. - Melanoma can occur anywhere on the skin. - Unusual mole...
5. `medquad_ans_0000063-15` (score=0.4597) - Melanoma is the fastest growing and most invasive type of skin cancer. This cancer arises from overgrowth of melanocytes. Melanocytes are specialized skin cells that produce a pigment called melanin.

**Answer**:

The provided context does not contain information about the outlook or prognosis for melanoma. Therefore, it is not possible to answer the question based on the given text. [doc_id] is not applicable.

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Top docs**:

1. `medquad_ans_0000139-2__dup2` (score=0.7000) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the gene on to you. You...
2. `medquad_ans_0002274-3` (score=0.6001) - How is familial isolated hyperparathyroidism inherited? Familial isolated hyperparathyroidism (FIHP) is typically inherited in an autosomal dominant manner. This means that having only one changed (mutated) copy of the responsible gene i...
3. `medquad_ans_0000300-4` (score=0.0000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
4. `medquad_ans_0000677-4` (score=0.0000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000990-4` (score=0.0000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...

**Answer**:

The provided context does not contain information about the inheritance pattern of boomerang dysplasia. Therefore, it is not possible to determine whether boomerang dysplasia is inherited based on the given text. [no_context]

**Citations detected**: `no_context`
**Supported citations**: _none_
**Unsupported citations**: `no_context`

## Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Top docs**:

1. `medquad_ans_0000466-1` (score=0.7000) - Hereditary multiple osteochondromas is a condition in which people develop multiple benign (noncancerous) bone tumors called osteochondromas. The number of osteochondromas and the bones on which they are located vary greatly among affect...
2. `medquad_ans_0000466-2` (score=0.6275) - The incidence of hereditary multiple osteochondromas is estimated to be 1 in 50,000 individuals. This condition occurs more frequently in some isolated populations: the incidence is approximately 1 in 1,000 in the Chamorro population of ...
3. `medquad_ans_0002838-1` (score=0.5943) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called osteochondromas). T...
4. `medquad_ans_0002838-2` (score=0.5819) - What are the signs and symptoms of Hereditary multiple osteochondromas? The Human Phenotype Ontology provides the following list of signs and symptoms for Hereditary multiple osteochondromas. If the information is available, the table be...
5. `medquad_ans_0000466-5` (score=0.4055) - These resources address the diagnosis or management of hereditary multiple osteochondromas: - Gene Review: Gene Review: Hereditary Multiple Osteochondromas - Genetic Testing Registry: Multiple congenital exostosis - Genetic Testing Regis...

**Answer**:

The provided context does not mention specific genetic changes related to hereditary multiple osteochondromas. Therefore, based on the given information, the genetic changes are not specified [medquad_ans_0000466-1, medquad_ans_0002838-1].

**Citations detected**: `medquad_ans_0000466-1, medquad_ans_0002838-1`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000466-1, medquad_ans_0002838-1`

## Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Top docs**:

1. `medquad_ans_0000746-5` (score=0.7000) - These resources address the diagnosis or management of oculofaciocardiodental syndrome: - Genetic Testing Registry: Oculofaciocardiodental syndrome These resources from MedlinePlus offer information about the diagnosis and management of ...
2. `medquad_ans_0004510-1` (score=0.5777) - Oculofaciocardiodental syndrome is a genetic syndrome that affects the eyes, heart, face, and teeth. Common signs and symptoms include abnormally small deep-set eyes, cataracts, long narrow face, a broad nasal tip that is divided by a cl...
3. `medquad_ans_0004510-2` (score=0.4673) - What are the signs and symptoms of Oculofaciocardiodental syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Oculofaciocardiodental syndrome. If the information is available, the table below incl...
4. `medquad_ans_0000344-2` (score=0.3446) - It is estimated that 750,000 people in the United States have dilated cardiomyopathy; roughly half of these cases are familial.
5. `medquad_ans_0000151-2` (score=0.2565) - Cardiofaciocutaneous syndrome is a very rare condition whose incidence is unknown. Researchers estimate that 200 to 300 people worldwide have this condition.

**Answer**:

The provided context does not specify the total number of people affected by oculofaciocardiodental syndrome. While it lists the frequency of various signs and symptoms in individuals with the condition, such as "90% with abnormality of the cardiac septa," it does not provide a population-wide prevalence or total number of affected individuals. Therefore, the exact number of people affected cannot be determined from the given information.

[medquad_ans_0004510-2]

**Citations detected**: `medquad_ans_0004510-2`
**Supported citations**: `medquad_ans_0004510-2`
**Unsupported citations**: _none_

## Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Top docs**:

1. `medquad_ans_0000189-1` (score=0.7000) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused by mutations in diffe...
2. `medquad_ans_0001350-1` (score=0.5924) - Citrullinemia type I is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. This condition, also known as classic citrullinemia, belongs to a class of genetic diseases called urea cycle disord...
3. `medquad_ans_0004334-1` (score=0.4808) - Neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) is a liver condition is also known as neonatal-onset type II citrullinemia. NICCD blocks the flow of bile (a digestive fluid produced by the liver) and prevents the bo...
4. `medquad_ans_0000189-5` (score=0.4701) - These resources address the diagnosis or management of citrullinemia: - Baby's First Test: Citrullinemia, Type I - Baby's First Test: Citrullinemia, Type II - Gene Review: Gene Review: Citrin Deficiency - Gene Review: Gene Review: Citrul...
5. `medquad_ans_0001350-5` (score=0.4547) - What happens when citrullinemia type I is not treated? Untreated individuals with the severe form of citrullinemia type I have hyperammonemia (plasma ammonia concentration 1000-3000 mol/L). Without prompt intervention, hyperammonemia and...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described: Type I (also known as classic citrullinemia), which is caused by mutations in the ASS1 gene and inherited in an autosomal recessive pattern, and typically becomes evident in the first few days of life with symptoms such as lethargy, poor feeding, vomiting, seizures, and loss of consciousness; and Type II, which chiefly affects the nervous system, causing confusion, restlessness, memory loss, abnormal behaviors, seizures, and coma, and may appear during adulthood, often triggered by medications, infections, surgery, or alcohol intake. Type II can also develop in individuals who had neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) as infants [medquad_ans_0000189-1]medquad_ans

**Citations detected**: `medquad_ans_0000189-1`
**Supported citations**: `medquad_ans_0000189-1`
**Unsupported citations**: _none_

## Example 99 (`0000718-1__dup1`)

**Question**: What is (are) Bilateral frontal polymicrogyria ?

**Top docs**:

1. `medquad_ans_0000718-1__dup1` (score=0.7000) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area located behind the Sylvius...
2. `medquad_ans_0000720-1__dup1` (score=0.5969) - Bilateral generalized polymicrogyria is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). This is the most widespread form of polymicrogyria and typically affects the entire surface of the br...
3. `medquad_ans_0000719-1__dup1` (score=0.5633) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides of the brain (bilate...
4. `medquad_ans_0004956-1` (score=0.4540) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. Specifically, the surface of the brain develops too many folds which are unusually small. The signs and symptoms associated with the condition...
5. `medquad_ans_0000807-1` (score=0.4337) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. The surface of the brain normally has many ridges or folds, called gyri. In people with polymicrogyria, the brain develops too many folds, and...

**Answer**:

Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form that affects only the frontal lobes of the brain, without involving the areas located behind the Sylvius fissure or the Rolando sulcus. Signs and symptoms include delayed motor and language milestones; spastic hemiparesis or quadriparesis; and mild to moderate intellectual disability. Seizures may also be present. The cause of this form is still not known, unlike the frontoparietal form, which is associated with mutations in the GPR56 gene. Treatment is based on the signs and symptoms present in each individual [medquad_ans_0000718-1__dup1].

**Citations detected**: `medquad_ans_0000718-1__dup1`
**Supported citations**: `medquad_ans_0000718-1__dup1`
**Unsupported citations**: _none_

## Example 100 (`0000944-5`)

**Question**: What are the treatments for spondylocostal dysostosis ?

**Top docs**:

1. `medquad_ans_0000944-2` (score=0.7000) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.
2. `medquad_ans_0005860-4` (score=0.6493) - What treatment is available for spondylothoracic dysostosis? Many infants born with spondylothoracic dysostosis have difficulty breathing due to their small, malformed chests, and therefore are prone to repeated respiratory infections (p...
3. `medquad_ans_0000949-5` (score=0.5260) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from MedlinePlus offer informa...
4. `medquad_ans_0005820-1` (score=0.3932) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an abnormal side-to-si...
5. `medquad_ans_0005819-1` (score=0.3605) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an abnormal side-to-si...

**Answer**:

Treatment for spondylocostal dysostosis usually consists of intensive medical care, including treatment of respiratory infections, bone surgery, and orthopedic treatment. Respiratory support may also be provided, and surgery may be used to correct inguinal hernia and scoliosis [medquad_ans_0005860-4]. Treatment is generally symptomatic and supportive, depending on the specific complications such as breathing difficulties or spinal deformities [medquad_ans_0005820-1].

**Citations detected**: `medquad_ans_0005860-4`, `medquad_ans_0005820-1`
**Supported citations**: `medquad_ans_0005860-4`, `medquad_ans_0005820-1`
**Unsupported citations**: _none_

## Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Top docs**:

1. `medquad_ans_0004825-1` (score=0.7000) - What are the signs and symptoms of Perry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Perry syndrome. If the information is available, the table below includes how often the symptom is seen...
2. `medquad_ans_0000786-1` (score=0.5523) - Perry syndrome is a progressive brain disease that is characterized by four major features: a pattern of movement abnormalities known as parkinsonism, psychiatric changes, weight loss, and abnormally slow breathing (hypoventilation). The...
3. `medquad_ans_0002652-2` (score=0.2611) - What are the signs and symptoms of Gordon syndrome? Gordon syndrome belongs to a group of conditions known as the distal arthrogryposes, which are characterized by stiffness and impaired mobility of certain joints of the lower arms and l...
4. `medquad_ans_0002480-2` (score=0.2146) - What are the signs and symptoms of Galloway-Mowat syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Galloway-Mowat syndrome. If the information is available, the table below includes how often t...
5. `medquad_ans_0005016-2` (score=0.1651) - What are the signs and symptoms of Prader-Willi syndrome? In infancy, Prader-Willi syndrome (PWS) is characterized by weak muscle tone (hypotonia), feeding difficulties, poor growth, and delayed development. In later infancy or early chi...

**Answer**:

The symptoms of Perry syndrome include abnormality of extrapyramidal motor function (such as parkinsonism, bradykinesia, rigidity, tremor, and mask-like facies), respiratory insufficiency (including hypoventilation and central hypoventilation), sleep disturbance, weight loss, developmental regression, hallucinations, hypotension, apathy, inappropriate behavior, insomnia, dysarthria, weak voice, and psychiatric changes such as depression and suicidal thoughts [medquad_ans_0004825-1]. Parkinsonism and psychiatric changes are typically the earliest features of the disease, which usually appears in a person's forties or fifties [medquad_ans_0000786-1].

**Citations detected**: `medquad_ans_0004825-1`, `medquad_ans_0000786-1`
**Supported citations**: `medquad_ans_0004825-1`, `medquad_ans_0000786-1`
**Unsupported citations**: _none_

## Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Top docs**:

1. `medquad_ans_0006399-1` (score=0.7000) - What are the signs and symptoms of Warfarin syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin syndrome. If the information is available, the table below includes how often the symptom i...
2. `medquad_ans_0006398-2` (score=0.6400) - What are the signs and symptoms of Warfarin sensitivity? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin sensitivity. If the information is available, the table below includes how often the sym...
3. `medquad_ans_0001028-2` (score=0.5430) - The prevalence of warfarin sensitivity is unknown. However, it appears to be more common in people who are older, those with lower body weights, and individuals of Asian ancestry. Of the approximately 2 million people in the U.S. who are...
4. `medquad_ans_0001027-2` (score=0.3821) - Warfarin resistance is thought to be a rare condition, although its prevalence is unknown.
5. `medquad_ans_0006398-1` (score=0.3393) - Warfarin sensitivity is a condition that is characterized by a reduced tolerance for a "blood-thinning" medication called warfarin. Warfarin is an anticoagulant that is often prescribed to people who are at an increased risk for blood cl...

**Answer**:

The symptoms of Warfarin sensitivity include abnormality of blood and blood-forming tissues, abnormality of metabolism/homeostasis, and autosomal dominant inheritance. [medquad_ans_0006398-2]

**Citations detected**: `medquad_ans_0006398-2`
**Supported citations**: `medquad_ans_0006398-2`
**Unsupported citations**: _none_

## Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Top docs**:

1. `medquad_ans_0000701-2` (score=0.7000) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.
2. `medquad_ans_0004262-1` (score=0.6711) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement ...
3. `medquad_ans_0000701-1` (score=0.5751) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most have intellectual d...
4. `medquad_ans_0004262-2` (score=0.3467) - What are the signs and symptoms of Myhre syndrome? Myhre syndrome is a condition with features affecting many systems and functions of the body. Associated findings might include: Delayed development of language and motor skills such as ...
5. `medquad_ans_0000701-5` (score=0.2366) - These resources address the diagnosis or management of Myhre syndrome: - Centers for Disease Control and Prevention: Types of Hearing Loss - Genetic Testing Registry: Myhre syndrome - National Institute on Deafness and Other Communicatio...

**Answer**:

Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement of the muscles (muscle hypertrophy), and/or joint stiffness. The condition is caused by mutations in the SMAD4 gene and is inherited in an autosomal dominant pattern, with most cases due to a new mutation. Affected individuals often have delayed development of language and motor skills, such as crawling and walking, and may have behavioral issues like features of autism or related developmental disorders. Common physical features include hearing loss (sensorineural, conductive, or mixed), skin stiffness, thickened skull bones, flattened spine bones (platyspondyly), broad ribs, underdeveloped pelvic wings (hypoplastic iliac wings), short fingers and toes (brachydactyly), joint stiffness, and limited mobility. Typical facial features include narrow

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Top docs**:

1. `medquad_ans_0004451-2` (score=0.7000) - What are the signs and symptoms of Noonan syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome 1. If the information is available, the table below includes how often the symptom i...
2. `medquad_ans_0002523-1` (score=0.4654) - What are the signs and symptoms of Genitopatellar syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Genitopatellar syndrome. If the information is available, the table below includes how often t...
3. `medquad_ans_0004826-1` (score=0.4624) - What are the signs and symptoms of Persistence of mullerian derivatives with lymphangiectasia and postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Persistence of mullerian derivati...
4. `medquad_ans_0000673-2__dup1` (score=0.4296) - What are the signs and symptoms of Behcet's disease? Symptoms of Behcet's disease include recurrent ulcers in the mouth (resembling canker sores) and on the genitals, and eye inflammation (uveitis). The disorder may also cause various ty...
5. `medquad_ans_0005640-1` (score=0.4293) - What are the signs and symptoms of Single upper central incisor? The Human Phenotype Ontology provides the following list of signs and symptoms for Single upper central incisor. If the information is available, the table below includes h...

**Answer**:

The provided context does not contain information about the symptoms of Thanatophoric dysplasia. [medquad_ans_0004451-2] and [medquad_ans_0002523-1] discuss symptoms of Noonan syndrome 1 and Genitopatellar syndrome, respectively, but neither mentions Thanatophoric dysplasia. Therefore, the requested information is not available in the context.

**Citations detected**: `medquad_ans_0004451-2`, `medquad_ans_0002523-1`
**Supported citations**: `medquad_ans_0004451-2`, `medquad_ans_0002523-1`
**Unsupported citations**: _none_

## Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Top docs**:

1. `medquad_ans_0002417-2` (score=0.7000) - What are the signs and symptoms of Fragile X syndrome? Fragile X syndrome is characterized by developmental problems including intellectual disability and delayed speech and language development. Males are usually more severely affected ...
2. `medquad_ans_0000381-1__dup1` (score=0.4082) - Fragile X syndrome is the most common form of inherited developmental disability. A problem with a specific gene causes the disease. Normally, the gene makes a protein you need for brain development. But the problem causes a person to ma...
3. `medquad_ans_0000381-1` (score=0.3422) - Fragile X syndrome is a genetic condition that causes a range of developmental problems including learning disabilities and cognitive impairment. Usually, males are more severely affected by this disorder than females. Affected individua...
4. `medquad_ans_0002417-6` (score=0.2642) - How might fragile X syndrome be treated? There is no specific treatment available for fragile X syndrome. Management of this condition is generally supportive and may include: recognizing the need for special education and avoiding exces...
5. `medquad_ans_0000381-5` (score=0.2270) - These resources address the diagnosis or management of fragile X syndrome: - Gene Review: Gene Review: FMR1-Related Disorders - GeneFacts: Fragile X Syndrome: Diagnosis - GeneFacts: Fragile X Syndrome: Management - Genetic Testing Regist...

**Answer**:

The symptoms of Fragile X syndrome include cognitive impairment, joint hypermobility, macroorchidism, neurological speech impairment, otitis media, pes planus, abnormality of the pinna, attention deficit hyperactivity disorder, frontal bossing, intellectual disability, moderate, long face, macrocephaly, mandibular prognathia, muscular hypotonia, narrow face, sinusitis, abnormality of the mitral valve, autism, cerebral cortical atrophy, dilatation of the ascending aorta, seizures, self-injurious behavior, and strabismus [medquad_ans_0002417-2]. Additionally, males may have enlarged testicles (macroorchidism) after puberty, and physical features such as a long and narrow face, large ears, prominent jaw and forehead, unusually flexible fingers, and flat feet may become more apparent with age [medquad_ans_000241

**Citations detected**: `medquad_ans_0002417-2`
**Supported citations**: `medquad_ans_0002417-2`
**Unsupported citations**: _none_

## Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Top docs**:

1. `medquad_ans_0000166-2` (score=0.7000) - CADASIL is likely a rare condition; however, its prevalence is unknown.
2. `medquad_ans_0000166-5` (score=0.5815) - These resources address the diagnosis or management of CADASIL: - Butler Hospital: Treatment and Management of CADASIL - Gene Review: Gene Review: CADASIL - Genetic Testing Registry: Cerebral autosomal dominant arteriopathy with subcorti...
3. `medquad_ans_0000056-3__dup1` (score=0.5487) - Symptoms usually progress slowly. By age 65, the majority of persons with CADASIL have cognitive problems and dementia. Some will become dependent due to multiple strokes.
4. `medquad_ans_0000889-2__dup1` (score=0.3887) - What are the signs and symptoms of CADASIL? Strokes are the main feature of CADASIL and often occur repeatedly. Strokes may lead to severe disability such as an inability to walk and urinary incontinence. The average age at onset for str...
5. `medquad_ans_0000166-1` (score=0.2505) - Cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy, usually called CADASIL, is an inherited condition that causes stroke and other impairments. This condition affects blood flow in small blood vess...

**Answer**:

CADASIL is a rare condition characterized by recurrent strokes, which are the main feature, often occurring repeatedly and leading to severe disability such as inability to walk and urinary incontinence. It typically presents with a slow decline in cognitive function, with over half of affected individuals experiencing a decline in thinking ability, beginning as early as 35 years of age. Approximately 75% of people with CADASIL eventually develop dementia, including significant difficulties with reasoning and memory. Migraines with aura occur in about 35% of individuals, with the first attack typically occurring at an average age of 26 years. Psychiatric issues, such as personality changes or severe depression, affect about 30% of people. Other common signs and symptoms include abnormality of temperature regulation (90%), abnormality of the retinal vasculature (90%), and memory impairment (50%). The condition is associated with cerebral autosomal dominant arter

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Top docs**:

1. `medquad_ans_0000688-1` (score=0.7000) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cel...
2. `medquad_ans_0000688-2` (score=0.4527) - Multiple endocrine neoplasia type 1 affects about 1 in 30,000 people; multiple endocrine neoplasia type 2 affects an estimated 1 in 35,000 people. Among the subtypes of type 2, type 2A is the most common form, followed by FMTC. Type 2B i...
3. `medquad_ans_0004187-2` (score=0.2987) - What are the signs and symptoms of Multiple endocrine neoplasia type 2A? The Human Phenotype Ontology provides the following list of signs and symptoms for Multiple endocrine neoplasia type 2A. If the information is available, the table ...
4. `medquad_ans_0004185-2` (score=0.2617) - What are the signs and symptoms of Multiple endocrine neoplasia type 1? Multiple endocrine neoplasia, type 1 (MEN1) is characterized primarily by several different types of endocrine tumors. People affected by MEN1 typically develop tumo...
5. `medquad_ans_0000010-8` (score=0.2063) - - Multiple endocrine neoplasia type 1 (MEN1) is an inherited disorder that causes hormone-secreting tumors in the duodenum and the endocrine glands-most often the parathyroid, pancreas, and pituitary. - Overactive parathyroid glands can ...

**Answer**:

Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cells and tissues throughout the body. Multiple endocrine neoplasia typically involves tumors (neoplasia) in at least two endocrine glands; tumors can also develop in other organs and tissues. These growths can be noncancerous (benign) or cancerous (malignant). If the tumors become cancerous, the condition can be life-threatening. The major forms of multiple endocrine neoplasia are called type 1, type 2, and type 4. These types are distinguished by the genes involved, the types of hormones made, and the characteristic signs and symptoms. [medquad_ans_0000688-1]

**Citations detected**: `medquad_ans_0000688-1`
**Supported citations**: `medquad_ans_0000688-1`
**Unsupported citations**: _none_

## Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Top docs**:

1. `medquad_ans_0000200-1__dup2` (score=0.7000) - The mucopolysaccharidoses are a group of inherited metabolic diseases in which a defective or missing enzyme causes large amounts of complex sugar molecules to accumulate in harmful amounts in the body's cells and tissues. This accumulat...
2. `medquad_ans_0004160-1` (score=0.5432) - Mucopolysaccharidosis type IIIC (MPS IIIC) is an genetic disorder that makes the body unable to break down large sugar molecules called glycosaminoglycans (GAGs, formerly called mucopolysaccharides). Specifically, people with this condit...
3. `medquad_ans_0004155-1` (score=0.4689) - Mucopolysaccharidosis I (MPS I) is a condition that affects many parts of the body. It is a progressively debilitating disorder; however, the rate of progression varies among affected individuals. MPS I is caused by mutations in the IDUA...
4. `medquad_ans_0004155-3` (score=0.4166) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs)....
5. `medquad_ans_0004161-1` (score=0.3940) - Mucopolysaccharidosis type IIID (MPS IIID) is an genetic disorder that makes the body unable to break down large sugar molecules called glycosaminoglycans (GAGs, formerly called mucopolysaccharides). Specifically, people with this condit...

**Answer**:

Mucopolysaccharidosis type I (MPS I) is caused by mutations in the IDUA gene. These mutations lead to reduced levels or the complete lack of the IDUA enzyme. Without the proper amount of this enzyme, large sugar molecules called glycosaminoglycans (GAGs) accumulate within cells called lysosomes [medquad_ans_0004155-1].

**Citations detected**: `medquad_ans_0004155-1`
**Supported citations**: `medquad_ans_0004155-1`
**Unsupported citations**: _none_

## Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Top docs**:

1. `medquad_ans_0004784-4` (score=0.7000) - How is Pelizaeus-Merzbacher disease inherited?
2. `medquad_ans_0002911-3` (score=0.6950) - What causes Holt-Oram syndrome? Holt-Oram syndrome is caused by changes (mutations) in the TBX5 gene. This gene gives the body instructions for making a protein involved in the development of the heart and upper limbs before birth. In pa...
3. `medquad_ans_0000107-3__dup2` (score=0.4081) - Primary ciliary dyskinesia (PCD) is a rare, inherited disease. "Inherited" means the disease is passed from parents to children through genes. With PCD, this process is very complex. Researchers are still learning how the disease is inhe...
4. `medquad_ans_0004671-4` (score=0.4002) - How is pachyonychia congenita inherited? Pachyonychia congenita (PC) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of the responsible gene in each cell. ...
5. `medquad_ans_0006446-4` (score=0.3372) - Is Williams syndrome inherited?

**Answer**:

The provided context does not contain any information about the genetic changes related to primary spontaneous pneumothorax. Therefore, a definitive answer cannot be given based on the available information. [no relevant context]

**Citations detected**: `no relevant context`
**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

## Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Top docs**:

1. `medquad_ans_0000066-13` (score=0.7000) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blood vessel in the bra...
2. `medquad_ans_0000261-1__dup2` (score=0.5533) - A stroke occurs when the blood supply to part of the brain is suddenly interrupted or when a blood vessel in the brain bursts, spilling blood into the spaces surrounding brain cells. Brain cells die when they no longer receive oxygen and...
3. `medquad_ans_0000066-18` (score=0.5423) - Transient ischemic attacks, or TIAs, occur when the warning signs of stroke last only a few moments and then disappear. These brief episodes are also sometimes called "mini-strokes." Although brief, they identify an underlying serious co...
4. `medquad_ans_0000857-1__dup2` (score=0.4971) - A stroke is a medical emergency. Strokes happen when blood flow to your brain stops. Within minutes, brain cells begin to die. There are two kinds of stroke. The more common kind, called ischemic stroke, is caused by a blood clot that bl...
5. `medquad_ans_0000124-1__dup2` (score=0.4395) - A stroke occurs if the flow of oxygen-rich blood to a portion of the brain is blocked. Without oxygen, brain cells start to die after a few minutes. Sudden bleeding in the brain also can cause a stroke if it damages brain cells. If brain...

**Answer**:

A stroke is a medical emergency that occurs when the blood supply to part of the brain is suddenly interrupted or when a blood vessel in the brain bursts, spilling blood into the spaces surrounding brain cells. Brain cells die when they no longer receive oxygen and nutrients from the blood or there is sudden bleeding into or around the brain. There are two main types of stroke: ischemic stroke, which is caused by a blood clot that blocks or plugs a blood vessel in the brain and accounts for approximately 80 percent of all strokes, and hemorrhagic stroke, which is caused by a blood vessel that breaks and bleeds into the brain [medquad_ans_0000261-1__dup2]. Additionally, transient ischemic attacks (TIAs), also known as "mini-strokes," occur when the blood supply to the brain is briefly interrupted, resulting in temporary symptoms that resolve within a few moments [medquad_ans_0000

**Citations detected**: `medquad_ans_0000261-1__dup2`
**Supported citations**: `medquad_ans_0000261-1__dup2`
**Unsupported citations**: _none_

## Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Top docs**:

1. `medquad_ans_0006276-2` (score=0.7000) - What are the signs and symptoms of Uncombable hair syndrome? Uncombable hair syndrome (UHS) may first become apparent any time between the ages of 3 months and 12 years. It only affects the scalp hair. The quantity of hair remains normal...
2. `medquad_ans_0006276-1` (score=0.6369) - Uncombable hair syndrome (UHS) is a rare disorder of the hair shaft of the scalp. It usually is characterized by silvery-blond or straw-colored hair that is disorderly; stands out from the scalp; and cannot be combed flat. It may first b...
3. `medquad_ans_0006276-5` (score=0.6213) - How is uncombable hair syndrome diagnosed? A diagnosis of uncombable hair syndrome (UHS) is made by observing the characteristic symptoms of the condition, as well observing the hair shaft under a special microscope. When the individual ...
4. `medquad_ans_0002708-1` (score=0.5249) - HAIR-AN syndrome is a condition that affects women. It is characterized by hyperandrogenism, insulin resistance, and acanthosis nigricans. Insulin resistance is a condition in which the body produces insulin but does not use it properly....
5. `medquad_ans_0006276-6` (score=0.4991) - How might uncombable hair syndrome be treated? There is no definitive treatment for uncombable hair syndrome, but the condition usually improves or resolves on its own with the onset of puberty. Gentle hair care is generally recommended ...

**Answer**:

The cause of Uncombable hair syndrome (UHS) is not known. A responsible gene has not yet been identified, and the condition is likely inherited in an autosomal dominant manner with reduced penetrance [medquad_ans_0006276-1]. However, no specific causal factor or genetic mutation is currently established. Therefore, based on the provided context, the exact cause remains unidentified.

**Citations detected**: `medquad_ans_0006276-1`
**Supported citations**: `medquad_ans_0006276-1`
**Unsupported citations**: _none_

## Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000261-4` (score=0.7000) - In most cases of cyclic vomiting syndrome, affected people have no known history of the disorder in their family. However, many affected individuals have a family history of related conditions, such as migraines, irritable bowel syndrome...
2. `medquad_ans_0003743-4` (score=0.3232) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition which suggests that i...
3. `medquad_ans_0000557-4` (score=0.3173) - This condition is generally not inherited but arises from mutations in the body's cells that occur after conception. This alteration is called a somatic mutation and is present only in certain cells. Rarely, this condition is inherited i...
4. `medquad_ans_0000661-4` (score=0.2346) - Mitochondrial complex III deficiency is usually inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry o...
5. `medquad_ans_0000139-2__dup2` (score=0.2242) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the gene on to you. You...

**Answer**:

The provided context does not contain any information about Russell-Silver syndrome or its inheritance pattern. Therefore, based on the given context, it is not possible to determine whether Russell-Silver syndrome is inherited.

Answer: Insufficient context [doc_id].

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.7000) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are different types of weigh...
2. `medquad_ans_0000718-1__dup2` (score=0.2494) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts or reshape sagging ...
3. `medquad_ans_0000094-5__dup1` (score=0.1899) - Obesity hypoventilation syndrome (OHS) is diagnosed based on your medical history, signs and symptoms, and test results. Specialists Involved A critical care specialist, pulmonologist (lung specialist), and/or sleep specialist may diagno...
4. `medquad_ans_0000018-1__dup2` (score=0.1761) - After any operation, you'll have some side effects. There is usually some pain with surgery. There may also be swelling and soreness around the area that the surgeon cut. Your surgeon can tell you which side effects to expect. There can ...
5. `medquad_ans_0000022-5` (score=0.1130) - These resources address the diagnosis or management of adiposis dolorosa: - Genetic Testing Registry: Lipomatosis dolorosa - Merck Manual Consumer Version: Lipomas These resources from MedlinePlus offer information about the diagnosis an...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. It often limits the amount of food you can take in, and some types also affect how you digest and absorb nutrients. All types of weight loss surgery have risks and complications, such as infections, hernias, and blood clots. Many people lose weight quickly, but some regain weight later; however, following diet and exercise recommendations can help keep most of the weight off. Ongoing medical follow-up is required for the rest of your life [medquad_ans_0000969-1__dup2].

**Citations detected**: `medquad_ans_0000969-1__dup2`
**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

## Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Top docs**:

1. `medquad_ans_0000889-2` (score=0.7000) - Saethre-Chotzen syndrome has an estimated prevalence of 1 in 25,000 to 50,000 people.
2. `medquad_ans_0000889-5` (score=0.6750) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Saethre-Chotzen syndrom...
3. `medquad_ans_0000889-4` (score=0.4958) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one affected parent. Ot...
4. `medquad_ans_0001186-4` (score=0.2474) - How is chorea-acanthocytosis treated? There are currently no treatments to prevent or slow the progression of chorea-acanthocytosis; treatment is symptomatic and supportive. Management may include: botulinum toxin for decreasing the oro-...
5. `medquad_ans_0000889-1` (score=0.2251) - Saethre-Chotzen syndrome is a genetic condition characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the head and face. Most ...

**Answer**:

The provided context does not mention specific treatments for Saethre-Chotzen syndrome. While it details the diagnosis, inheritance pattern, and clinical features, no information is given about therapeutic interventions or management strategies. Therefore, based on the available context, the treatments for Saethre-Chotzen syndrome cannot be determined.

[medquad_ans_0000889-5] mentions resources related to diagnosis and management, such as genetic counseling, surgery and rehabilitation, and drug therapy, but does not specify treatments for Saethre-Chotzen syndrome. Thus, the exact treatments are not explicitly stated in the context.

**Citations detected**: `medquad_ans_0000889-5`
**Supported citations**: `medquad_ans_0000889-5`
**Unsupported citations**: _none_

## Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Top docs**:

1. `medquad_ans_0000512-5` (score=0.7000) - These resources address the diagnosis or management of idiopathic pulmonary fibrosis: - Gene Review: Gene Review: Pulmonary Fibrosis, Familial - Genetic Testing Registry: Idiopathic fibrosing alveolitis, chronic form These resources from...
2. `medquad_ans_0000512-1` (score=0.6948) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effectively. The disease u...
3. `medquad_ans_0000078-1__dup2` (score=0.6941) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens, your lungs can't p...
4. `medquad_ans_0000758-1__dup2` (score=0.6186) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough oxygen. Causes of pu...
5. `medquad_ans_0000512-2` (score=0.5630) - Idiopathic pulmonary fibrosis has an estimated prevalence of 13 to 20 per 100,000 people worldwide. About 100,000 people are affected in the United States, and 30,000 to 40,000 new cases are diagnosed each year. Familial pulmonary fibros...

**Answer**:

Idiopathic pulmonary fibrosis is a chronic, progressive lung disease in which scar tissue (fibrosis) builds up in the lungs, impairing their ability to transport oxygen into the bloodstream effectively [medquad_ans_0000512-1]. It typically affects individuals between the ages of 50 and 70 and is characterized by symptoms such as shortness of breath, a persistent dry, hacking cough, loss of appetite, and gradual weight loss. Some individuals may also develop clubbing of the fingers and toes due to oxygen shortage. The disease progresses over time, leading to reduced lung function and potentially serious complications like lung cancer, pulmonary emboli, pneumonia, or pulmonary hypertension. Most people with idiopathic pulmonary fibrosis survive 3 to 5 years after diagnosis, although the disease course is highly variable, with some living with it for a decade or longer. While most cases are sporadic, a small percentage have a familial

**Citations detected**: `medquad_ans_0000512-1`
**Supported citations**: `medquad_ans_0000512-1`
**Unsupported citations**: _none_

## Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Top docs**:

1. `medquad_ans_0000213-3__dup1` (score=0.7000) - Early identification of and treatment for individuals with neuroleptic malignant syndrome improves outcome. If clinically indicated, a low potency neuroleptic can be reintroduced very slowly when the individual recovers, although there i...
2. `medquad_ans_0004379-1` (score=0.6649) - Neuroleptic malignant syndrome is a rare neurological condition that is caused by an adverse reaction to neuroleptic (tranquilizer) or antipsychotic drugs. These drugs are commonly prescribed for the treatment of schizophrenia and other ...
3. `medquad_ans_0000213-4__dup1` (score=0.6229) - The NINDS supports research on neurological disorders such as neuroleptic malignant syndrome. Much of this research focuses on finding ways to prevent and treat the disorder.
4. `medquad_ans_0000273-4__dup1` (score=0.2189) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts TSC research in its laboratories at the National Institutes of Health (NIH) and also supports TSC research through grants to major medical institutions across t...
5. `medquad_ans_0000022-4__dup1` (score=0.2001) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system, and to use that knowledge to reduce the burden of neurological disease. The NINDS pur...

**Answer**:

The National Institute of Neurological Disorders and Stroke (NINDS) supports research on neurological disorders such as neuroleptic malignant syndrome, focusing on finding ways to prevent and treat the disorder [medquad_ans_0000213-4__dup1]. However, specific details about ongoing clinical trials or particular research studies on neuroleptic malignant syndrome are not provided in the context.

**Citations detected**: `medquad_ans_0000213-4__dup1`
**Supported citations**: `medquad_ans_0000213-4__dup1`
**Unsupported citations**: _none_

## Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Top docs**:

1. `medquad_ans_0000203-11` (score=0.7000) - - A kidney stone is a solid piece of material that forms in a kidney when substances that are normally found in the urine become highly concentrated. - Kidney stones are one of the most common disorders of the urinary tract. - Certain fo...
2. `medquad_ans_0000540-1__dup2` (score=0.6849) - A kidney stone is a solid piece of material that forms in the kidney from substances in the urine. It may be as small as a grain of sand or as large as a pearl. Most kidney stones pass out of the body without help from a doctor. But some...
3. `medquad_ans_0000221-8` (score=0.6210) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other health problems. The d...
4. `medquad_ans_0000221-12` (score=0.6142) - - A kidney stone is a solid piece of material that forms in a kidney when there are high levels of certain substances in the urine. These substances are normally found in the urine and do not cause problems at lower levels. - Kidney ston...
5. `medquad_ans_0000204-11` (score=0.3531) - - A kidney stone is a solid piece of material that forms in a kidney when some substances that are normally found in the urine become highly concentrated. - Kidney stones occur in infants, children, and teenagers from all races and ethni...

**Answer**:

To diagnose kidney stones in adults, a healthcare provider will perform a physical exam and take a medical history, including questions about family history of kidney stones, diet, digestive problems, and other health issues. The provider may also perform urine, blood, and imaging tests to complete the diagnosis. Urine tests can show whether there is an infection or the presence of substances that form stones. Blood tests can identify problems that may lead to kidney stones. Imaging tests are used to visualize the kidneys and urinary tract to confirm the presence of stones [medquad_ans_0000203-11][medquad_ans_0000221-8][medquad_ans_0000540-1__dup2].

**Citations detected**: `medquad_ans_0000203-11`, `medquad_ans_0000221-8`, `medquad_ans_0000540-1__dup2`
**Supported citations**: `medquad_ans_0000203-11`, `medquad_ans_0000221-8`, `medquad_ans_0000540-1__dup2`
**Unsupported citations**: _none_

## Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Top docs**:

1. `medquad_ans_0000041_1-1` (score=0.7000) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of bladder cancer can a...
2. `medquad_ans_0000099-1__dup2` (score=0.4656) - The bladder is a hollow organ in your lower abdomen that stores urine. Bladder cancer occurs in the lining of the bladder. It is the sixth most common type of cancer in the United States. Symptoms include - Blood in your urine - A freque...
3. `medquad_ans_0006289-1` (score=0.4228) - Urachal cancer is a rare type of bladder cancer, making up less than 1% of all bladder cancers. Only about 350 cases have been described in the medical literature to date. The urachus is a primitive structure which before birth connected...
4. `medquad_ans_0000041_1-2` (score=0.3520) - Signs of urethral cancer include bleeding or trouble with urination. These and other signs and symptoms may be caused by urethral cancer or by other conditions. There may be no signs or symptoms in the early stages. Check with your docto...
5. `medquad_ans_0000744-1__dup1` (score=0.3450) - Bladder cancer is a form of cancer that occurs due to abnormal and uncontrolled cell growth in the bladder. Signs and symptoms of the condition may include abdominal pain, blood in the urine, fatigue, painful urination, frequent urinatio...

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. The urethra is the tube that carries urine from the bladder to outside the body. In women, the urethra is about 1 inch long and is just above the vagina. In men, the urethra is about 8 inches long, and goes through the prostate gland and the penis to the outside of the body. In men, the urethra also carries semen. Urethral cancer is a rare cancer that occurs more often in men than in women. There are different types of urethral cancer that begin in cells that line the urethra, named for the types of cells that become malignant: squamous cell carcinoma, transitional cell carcinoma, and adenocarcinoma. Squamous cell carcinoma is the most common type and forms in cells near the bladder in women and in the lining

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Top docs**:

1. `medquad_ans_0004671-4` (score=0.7000) - How is pachyonychia congenita inherited? Pachyonychia congenita (PC) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of the responsible gene in each cell. ...
2. `medquad_ans_0006446-4` (score=0.5548) - Is Williams syndrome inherited?
3. `medquad_ans_0004750-4` (score=0.4598) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is passed down through fami...
4. `medquad_ans_0004784-4` (score=0.3847) - How is Pelizaeus-Merzbacher disease inherited?
5. `medquad_ans_0005548-4` (score=0.3416) - Is septo-optic dysplasia inherited?

**Answer**:

The provided context does not contain information about the inheritance of paramyotonia congenita. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Citations detected**: `no_context`
**Supported citations**: _none_
**Unsupported citations**: `no_context`

## Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Top docs**:

1. `medquad_ans_0000735-2__dup1` (score=0.7000) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound biotinidase deficiency,...
2. `medquad_ans_0000735-1__dup1` (score=0.5007) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. The disorder may become apparent in the first few months of life, or later in childhood. The more severe form of the disorder is c...
3. `medquad_ans_0000125-1` (score=0.4602) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. If this condition is not recognized and treated, its signs and symptoms typically appear within the first few months of life, alth...
4. `medquad_ans_0000125-2` (score=0.3343) - Profound or partial biotinidase deficiency occurs in approximately 1 in 60,000 newborns
5. `medquad_ans_0000484-1` (score=0.2418) - Holocarboxylase synthetase deficiency is an inherited disorder in which the body is unable to use the vitamin biotin effectively. This disorder is classified as a multiple carboxylase deficiency, a group of disorders characterized by imp...

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), reduced consciousness/confusion (7.5%), respiratory insufficiency (7.5%), skin ulcer (7.5%), and visual field defect (7.5%). Other associated symptoms may include feeding difficulties in infancy, recurrent skin infections, seborrheic dermatitis, sensorineural hearing impairment, hepatomegaly, hyperammonemia

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Top docs**:

1. `medquad_ans_0000966-1__dup2` (score=0.7000) - Summary : We all need clean water. People need it to grow crops and to operate factories, and for drinking and recreation. Fish and wildlife depend on it to survive. Many different pollutants can harm our rivers, streams, lakes, and ocea...
2. `medquad_ans_0000298-1__dup1` (score=0.6145) - Summary : We all need to drink water. How much you need depends on your size, activity level, and the weather where you live. The water you drink is a combination of surface water and groundwater. Surface water includes rivers, lakes and...
3. `medquad_ans_0000331-1__dup1` (score=0.5808) - Summary : Our environment affects our health. If parts of the environment, like the air, water, or soil become polluted, it can lead to health problems. For example, asthma pollutants and chemicals in the air or in the home can trigger a...
4. `medquad_ans_0000019-1__dup3` (score=0.3553) - Summary : Air pollution is a mixture of solid particles and gases in the air. Car emissions, chemicals from factories, dust, pollen and mold spores may be suspended as particles. Ozone, a gas, is a major part of air pollution in cities. ...
5. `medquad_ans_0000661-1__dup1` (score=0.2579) - Summary : Oil spills often happen because of accidents, when people make mistakes or equipment breaks down. Other causes include natural disasters or deliberate acts. Oil spills have major environmental and economic effects. Oil spills c...

**Answer**:

Yes, there is information about water pollution. The three most common pollutants affecting rivers, streams, lakes, and oceans are soil, nutrients, and bacteria. Rain washes soil into water bodies, which can kill fish eggs, clog fish gills, and block light, causing plant death. Nutrients from fertilizers cause algae growth, turning water green, especially in lakes, ponds, and reservoirs. Bacteria from sewage spills can pollute both fresh and salt water. To help protect water supplies, one should not pour household products like cleansers, medicines, or grease down the drain, and instead dispose of them at hazardous waste sites. Excess grease, diapers, condoms, and pet waste should be thrown in the garbage. Pet waste contains nutrients and germs that can contribute to water pollution [medquad_ans_0000966-1__dup2].

**Citations detected**: `medquad_ans_0000966-1__dup2`
**Supported citations**: `medquad_ans_0000966-1__dup2`
**Unsupported citations**: _none_

## Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Top docs**:

1. `medquad_ans_0000132-2__dup1` (score=0.7000) - Treatment depends on the specific cause of foot drop. The most common treatment is to support the foot with light-weight leg braces and shoe inserts, called ankle-foot orthotics. Exercise therapy to strengthen the muscles and maintain jo...
2. `medquad_ans_0000132-3__dup1` (score=0.6854) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue ...
3. `medquad_ans_0000132-1__dup1` (score=0.5745) - Foot drop describes the inability to raise the front part of the foot due to weakness or paralysis of the muscles that lift the foot. As a result, individuals with foot drop scuff their toes along the ground or bend their knees to lift t...
4. `medquad_ans_0000132-4__dup1` (score=0.5253) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to the neurological conditions that cause foot drop in its laboratories at the National Institutes of Health (NIH), and also supports additiona...
5. `medquad_ans_0006002-1` (score=0.3542) - Talipes equinovarus is a congenital (present from birth) condition where the foot turns inward and downward. The cause of this condition is not known, although it may be passed down through families in some cases. This condition occurs i...

**Answer**:

The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1].

**Citations detected**: `medquad_ans_0000132-3__dup1`
**Supported citations**: `medquad_ans_0000132-3__dup1`
**Unsupported citations**: _none_

## Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Top docs**:

1. `medquad_ans_0000145-2` (score=0.7000) - The prevalence of Camurati-Engelmann disease is unknown. Approximately 200 cases have been reported worldwide.
2. `medquad_ans_0000916-5__dup1` (score=0.4246) - How is Camurati-Engelmann disease diagnosed? Diagnosis of Camurati-Engelmann disease is based on physical examination and radiographic findings and can be confirmed by molecular genetic testing. TGFB1 is the only gene known to be associa...
3. `medquad_ans_0000145-5` (score=0.3353) - These resources address the diagnosis or management of Camurati-Engelmann disease: - Gene Review: Gene Review: Camurati-Engelmann Disease - Genetic Testing Registry: Diaphyseal dysplasia These resources from MedlinePlus offer information...
4. `medquad_ans_0000145-1` (score=0.3225) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are also af...
5. `medquad_ans_0000916-2__dup1` (score=0.2645) - What are the signs and symptoms of Camurati-Engelmann disease? People with Camurati-Engelmann disease have increased bone density, particularly affecting the long bones of the arms and legs (tibia, femur, humerus, ulna, radius). In some ...

**Answer**:

Camurati-Engelmann disease is a condition that mainly affects the bones, characterized by increased bone density, particularly in the long bones of the arms and legs. It can also affect the skull and hip bones. The thickened bones may lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. Increased skull density can cause increased pressure on the brain, resulting in neurological problems such as headaches, hearing loss, vision problems, dizziness (vertigo), ringing in the ears (tinnitus), and facial paralysis. Additional features include abnormal curvature of the spine (scoliosis), joint deformities (contractures), knock knees, flat feet (pes planus), abnormally long limbs in proportion to height, decreased muscle mass and body fat, and delayed puberty. The age of symptom onset varies, with most individuals developing pain or weakness by adolescence. In some cases, individuals may have the gene mutation

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Top docs**:

1. `medquad_ans_0000316-2` (score=0.7000) - X-linked Emery-Dreifuss muscular dystrophy is the most common form of this condition, affecting an estimated 1 in 100,000 people. The autosomal recessive type of this disorder appears to be very rare; only a few cases have been reported ...
2. `medquad_ans_0002080-2` (score=0.6801) - What are the signs and symptoms of Emery-Dreifuss muscular dystrophy, X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Emery-Dreifuss muscular dystrophy, X-linked. If the information is availab...
3. `medquad_ans_0000316-1` (score=0.6501) - Emery-Dreifuss muscular dystrophy is a condition that chiefly affects muscles used for movement (skeletal muscles) and heart (cardiac) muscle. Among the earliest features of this disorder are joint deformities called contractures, which ...
4. `medquad_ans_0000316-5` (score=0.4580) - These resources address the diagnosis or management of Emery-Dreifuss muscular dystrophy: - Gene Review: Gene Review: Emery-Dreifuss Muscular Dystrophy - Genetic Testing Registry: Emery-Dreifuss muscular dystrophy - Genetic Testing Regis...
5. `medquad_ans_0002213-2` (score=0.1833) - What are the signs and symptoms of Facioscapulohumeral muscular dystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Facioscapulohumeral muscular dystrophy. If the information is available, the ta...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked include abnormality of the neck, Achilles tendon contracture, atrioventricular block, childhood onset, decreased cervical spine flexion due to contractures of posterior cervical muscles, elbow flexion contracture, elevated serum creatine phosphokinase, juvenile onset, pectus excavatum, primary atrial arrhythmia, slow progression, sudden cardiac death, type 1 muscle fiber atrophy, waddling gait, and X-linked recessive inheritance [medquad_ans_0002080-2]. Among the earliest features are joint deformities called contractures, which restrict movement and become noticeable in early childhood, most often involving the elbows, ankles, and neck [medquad_ans_0000316-1]. Most affected individuals also experience slowly progressive muscle weakness and wasting, beginning in the muscles of the upper arms and lower legs

**Citations detected**: `medquad_ans_0002080-2`, `medquad_ans_0000316-1`
**Supported citations**: `medquad_ans_0002080-2`, `medquad_ans_0000316-1`
**Unsupported citations**: _none_

## Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Top docs**:

1. `medquad_ans_0000072-2` (score=0.7000) - Arterial tortuosity syndrome is a rare disorder; its prevalence is unknown. About 100 cases have been reported in the medical literature.
2. `medquad_ans_0000072-5` (score=0.6423) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins McKusick-Nathans Insti...
3. `medquad_ans_0000072-1` (score=0.4865) - Arterial tortuosity syndrome is a disorder that affects connective tissue. Connective tissue provides strength and flexibility to structures throughout the body, including blood vessels, skin, joints, and the gastrointestinal tract. As i...
4. `medquad_ans_0003631-5` (score=0.1588) - How might lipodermatosclerosis be treated? Lipodermatosclerosis is primarily treated with compression therapy to improve venous insufficiency. Other strategies for managing venous insufficiency include leg elevation; not sitting or stand...
5. `medquad_ans_0000089-6` (score=0.1271) - Marfan syndrome has no cure. However, treatments can help delay or prevent complications, especially when started early. Marfan syndrome can affect many parts of your body, including your heart, bones and joints, eyes, nervous system, an...

**Answer**:

The provided context does not specify treatments for arterial tortuosity syndrome. While it mentions resources that address the diagnosis or management of the disorder, such as genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, and palliative care, it does not detail specific treatments [medquad_ans_0000072-5]. Therefore, based on the given information, the exact treatments are not available.

**Citations detected**: `medquad_ans_0000072-5`
**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

## Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Top docs**:

1. `medquad_ans_0001093-1__dup1` (score=0.7000) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is available, the table be...
2. `medquad_ans_0001092-1__dup1` (score=0.6896) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is available, the table be...
3. `medquad_ans_0001094-2__dup1` (score=0.3969) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1E? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1E. If the information is available, the table be...
4. `medquad_ans_0001088-2__dup1` (score=0.3663) - What are the signs and symptoms of Charcot-Marie-Tooth disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease. If the information is available, the table below includes how...
5. `medquad_ans_0006492-1` (score=0.3412) - What are the signs and symptoms of X-linked Charcot-Marie-Tooth disease type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked Charcot-Marie-Tooth disease type 1. If the information is availab...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 1C include decreased motor nerve conduction velocity, distal amyotrophy, distal muscle weakness, distal sensory impairment, hypertrophic nerve changes, hyporeflexia, juvenile onset, onion bulb formation, pes cavus, and segmental peripheral demyelination/remyelination [medquad_ans_0001092-1__dup1].

**Citations detected**: `medquad_ans_0001092-1__dup1`
**Supported citations**: `medquad_ans_0001092-1__dup1`
**Unsupported citations**: _none_

## Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Top docs**:

1. `medquad_ans_0005321-1` (score=0.7000) - What are the signs and symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal arterial macroaneurysm with supravalvular ...
2. `medquad_ans_0000870-1` (score=0.4937) - Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) is a disorder that affects blood vessels in the eyes and heart. The condition generally becomes apparent in infancy or childhood. RAMSVPS damages the arteries ...
3. `medquad_ans_0000870-5` (score=0.2397) - These resources address the diagnosis or management of RAMSVPS: - Calgary Retina Consultants: Retinal Arterial Macroaneurysm - Genetic Testing Registry: Retinal arterial macroaneurysm with supravalvular pulmonic stenosis - MedlinePlus En...
4. `medquad_ans_0005330-2` (score=0.1868) - What are the signs and symptoms of Retinal vasculopathy with cerebral leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal vasculopathy with cerebral leukodystrophy. If the informatio...
5. `medquad_ans_0003142-2` (score=0.1843) - What are the signs and symptoms of idiopathic juxtafoveal retinal telangiectasia? Signs and symptoms of idiopathic juxtafoveal retinal telangiectasia may include slow loss of vision, distorted vision, trouble reading, and scotomata (a sp...

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) include exudative retinal detachment and pulmonic stenosis. Retinal arterial macroaneurysms can rupture, leading to bleeding within the eye that may cause vision loss. Additionally, supravalvular pulmonic stenosis results in narrowing of the pulmonary artery above the pulmonic valve, impairing blood flow to the lungs and reducing oxygenation of the blood. This leads to symptoms such as shortness of breath. [medquad_ans_0005321-1][medquad_ans_0000870-1]

**Citations detected**: `medquad_ans_0005321-1`, `medquad_ans_0000870-1`
**Supported citations**: `medquad_ans_0005321-1`, `medquad_ans_0000870-1`
**Unsupported citations**: _none_

## Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Top docs**:

1. `medquad_ans_0001838-2` (score=0.7000) - What are the signs and symptoms of Dicarboxylic aminoaciduria? There are no common signs or symptoms of dicarboxylic aminoaciduria. Hypoglycemia, developmental and neurological abnormalities, and obsessive compulsive tendencies were desc...
2. `medquad_ans_0000050-1` (score=0.3776) - Aminoacylase 1 deficiency is an inherited disorder that can cause neurological problems; the pattern and severity of signs and symptoms vary widely among affected individuals. Individuals with this condition typically have delayed develo...
3. `medquad_ans_0000067-3` (score=0.2983) - Mutations in the ASL gene cause argininosuccinic aciduria. Argininosuccinic aciduria belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occur in liver cells. It processes ex...
4. `medquad_ans_0002894-1` (score=0.1863) - What are the signs and symptoms of Histidinuria renal tubular defect? The Human Phenotype Ontology provides the following list of signs and symptoms for Histidinuria renal tubular defect. If the information is available, the table below ...
5. `medquad_ans_0000168-7` (score=0.1522) - - Proteinuria is a condition in which urine contains a detectable amount of protein. - Proteinuria is a sign of chronic kidney disease (CKD). - Groups at risk for proteinuria include African Americans, American Indians, Hispanics/Latinos...

**Answer**:

Dicarboxylic aminoaciduria is a condition characterized by the presence of dicarboxylic amino acids in the urine. It is associated with no common signs or symptoms; however, in individual cases, hypoglycemia, developmental and neurological abnormalities, and obsessive compulsive tendencies have been reported. Some individuals diagnosed with this condition have virtually no signs or symptoms. Aminoaciduria, with autosomal recessive inheritance, and fasting hypoglycemia, intellectual disability, are among the reported signs and symptoms, with frequencies estimated based on studies [medquad_ans_0001838-2]. The Human Phenotype Ontology (HPO) compiles information on the occurrence of these features in patients with the condition [medquad_ans_0001838-2].

**Citations detected**: `medquad_ans_0001838-2`
**Supported citations**: `medquad_ans_0001838-2`
**Unsupported citations**: _none_

## Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Top docs**:

1. `medquad_ans_0005859-1` (score=0.7000) - Spondylospinal thoracic dysostosis is an extremely rare skeletal disorder characterized by a short, curved spine and fusion of the spinous processes, short thorax with 'crab-like' configuration of the ribs, underdevelopment of the lungs ...
2. `medquad_ans_0000949-5` (score=0.6747) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from MedlinePlus offer informa...
3. `medquad_ans_0000944-2` (score=0.4839) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.
4. `medquad_ans_0005819-1` (score=0.4823) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an abnormal side-to-si...
5. `medquad_ans_0005822-1` (score=0.4691) - Spondylocostal dysostosis is a group of conditions characterized by abnormal development of the bones in the spine and ribs. In the spine, the vertebrae are misshapen and fused. Many people with this condition have an abnormal side-to-si...

**Answer**:

Spondylothoracic dysostosis is believed to be inherited in an autosomal recessive manner [medquad_ans_0005859-1]. However, it is important to note that spondylothoracic dysostosis and spondylocostal dysostosis are distinct conditions, and the provided context does not explicitly confirm that spondylothoracic dysostosis is the same as spondylocostal dysostosis. While spondylocostal dysostosis is described as being inherited in an autosomal recessive manner, the direct evidence for spondylothoracic dysostosis is limited to the statement that it is believed to be inherited in an autosomal recessive manner [medquad_ans_0005859-1]. Therefore, based on the provided context, spondylothoracic dysostosis is believed to be inherited in

**Citations detected**: `medquad_ans_0005859-1`
**Supported citations**: `medquad_ans_0005859-1`
**Unsupported citations**: _none_

## Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Top docs**:

1. `medquad_ans_0003296-1` (score=0.7000) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. It is inherited in an...
2. `medquad_ans_0000542-1` (score=0.6917) - Jervell and Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). This disorder is a form of long QT syndrome, which is a heart condition that cause...
3. `medquad_ans_0000542-2` (score=0.6618) - Jervell and Lange-Nielsen syndrome is uncommon; it affects an estimated 1.6 to 6 per 1 million people worldwide. This condition has a higher prevalence in Denmark, where it affects at least 1 in 200,000 people.
4. `medquad_ans_0000542-5` (score=0.4281) - These resources address the diagnosis or management of Jervell and Lange-Nielsen syndrome: - Gene Review: Gene Review: Jervell and Lange-Nielsen Syndrome - Genetic Testing Registry: Jervell and Lange-Nielsen syndrome - MedlinePlus Encycl...
5. `medquad_ans_0000241-1` (score=0.1929) - Cornelia de Lange syndrome is a developmental disorder that affects many parts of the body. The features of this disorder vary widely among affected individuals and range from relatively mild to severe. Cornelia de Lange syndrome is char...

**Answer**:

Jervell Lange-Nielsen syndrome is a form of long QT syndrome that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. The condition is inherited in an autosomal recessive fashion and has two different types: type 1 and type 2. It is uncommon, affecting an estimated 1.6 to 6 per 1 million people worldwide, with a higher prevalence in Denmark, where it affects at least 1 in 200,000 people [medquad_ans_0003296-1, medquad_ans_0000542-1, medquad_ans_0000542-2].

**Citations detected**: `medquad_ans_0003296-1, medquad_ans_0000542-1, medquad_ans_0000542-2`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0003296-1, medquad_ans_0000542-1, medquad_ans_0000542-2`

## Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Top docs**:

1. `medquad_ans_0000218-5` (score=0.7000) - These resources address the diagnosis or management of CCFDN: - Gene Review: Gene Review: Congenital Cataracts, Facial Dysmorphism, and Neuropathy - Genetic Testing Registry: Congenital Cataracts, Facial Dysmorphism, and Neuropathy - Med...
2. `medquad_ans_0000218-1` (score=0.6711) - Congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is a rare disorder that affects several parts of the body. It is characterized by a clouding of the lens of the eyes at birth (congenital cataracts) and other eye abnormali...
3. `medquad_ans_0000507-2` (score=0.5395) - The prevalence of hypomyelination and congenital cataract is unknown.
4. `medquad_ans_0001004-1__dup1` (score=0.4093) - What are the signs and symptoms of Cataract, total congenital? The Human Phenotype Ontology provides the following list of signs and symptoms for Cataract, total congenital. If the information is available, the table below includes how o...
5. `medquad_ans_0000998-1__dup1` (score=0.3618) - What are the signs and symptoms of Cataract, autosomal recessive congenital 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Cataract, autosomal recessive congenital 2. If the information is available...

**Answer**:

The provided context does not specify the number of people affected by congenital cataracts, facial dysmorphism, and neuropathy (CCFDN). [medquad_ans_0000218-1] describes the condition's features and complications but does not include population statistics or prevalence data. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Citations detected**: `medquad_ans_0000218-1`
**Supported citations**: `medquad_ans_0000218-1`
**Unsupported citations**: _none_

## Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Top docs**:

1. `medquad_ans_0000259-3__dup1` (score=0.7000) - How is alpha-thalassemia x-linked intellectual disability syndrome inherited? Alpha-thalassemia x-linked intellectual disability (ATRX) syndrome is caused by a mutation in the ATRX gene and is inherited in an x-linked way. The chance tha...
2. `medquad_ans_0000127-1__dup2` (score=0.4833) - Thalassemias (thal-a-SE-me-ahs) are inherited blood disorders. "Inherited" means that the disorder is passed from parents to children through genes. Thalassemias cause the body to make fewer healthy red blood cells and less hemoglobin (H...
3. `medquad_ans_0003225-3` (score=0.4199) - How is intellectual disability-developmental delay-contractures syndrome inherited? Intellectual disability-developmental delay-contractures syndrome syndrome is inherited in an X-linked recessive manner and is caused by mutations in the...
4. `medquad_ans_0000127-4__dup2` (score=0.2613) - Family history and ancestry are the two risk factors for thalassemias. Family History Thalassemias are inheritedthat is, the genes for the disorders are passed from parents to their children. If your parents have missing or altered hemog...
5. `medquad_ans_0000381-1__dup1` (score=0.2589) - Fragile X syndrome is the most common form of inherited developmental disability. A problem with a specific gene causes the disease. Normally, the gene makes a protein you need for brain development. But the problem causes a person to ma...

**Answer**:

Alpha-thalassemia x-linked intellectual disability (ATRX) syndrome is caused by a mutation in the ATRX gene and is inherited in an x-linked manner [medquad_ans_0000259-3__dup1]. The inheritance pattern depends on whether the mutation was inherited from the mother or occurred de novo. If the mutation is de novo, the risk to relatives is very low. If inherited from the mother, each of her sisters has a 50% chance of being a carrier. A carrier woman has a 25% chance of having a son affected with ATRX syndrome, a 25% chance of having a son without the mutation, a 25% chance of having a daughter who is a carrier, and a 25% chance of having a daughter who is not a carrier [medquad_ans_0000259-3__dup1].

**Citations detected**: `medquad_ans_0000259-3__dup1`
**Supported citations**: `medquad_ans_0000259-3__dup1`
**Unsupported citations**: _none_

## Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Top docs**:

1. `medquad_ans_0002622-1` (score=0.7000) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepatomegaly), or an inc...
2. `medquad_ans_0002622-2` (score=0.6684) - What are the signs and symptoms of Glycogen storage disease type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 6. If the information is available, the table below incl...
3. `medquad_ans_0002615-2` (score=0.5337) - What are the signs and symptoms of Glycogen storage disease type 13? Glycogen storage disease type 13 causes muscle pain (myalgia). Individuals with GSD13 also experience exercise intolerance, which means they have difficulty exercising ...
4. `medquad_ans_0002614-1` (score=0.4771) - What are the signs and symptoms of Glycogen storage disease type 12? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 12. If the information is available, the table below in...
5. `medquad_ans_0002623-2` (score=0.4152) - What are the signs and symptoms of Glycogen storage disease type 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 7. If the information is available, the table below incl...

**Answer**:

The symptoms of Glycogen storage disease type 6 include low blood sugar (hypoglycemia), an enlarged liver (hepatomegaly), and increased lactic acid in the blood (lactic acidosis), particularly when an individual does not eat for a long time. Short stature and postnatal growth retardation are also associated with the condition. Hypoglycemia and short stature are reported in approximately 90% of patients [medquad_ans_0002622-2]. The condition is caused by mutations in the PYGL gene and is inherited in an autosomal recessive manner [medquad_ans_0002622-1].

**Citations detected**: `medquad_ans_0002622-2`, `medquad_ans_0002622-1`
**Supported citations**: `medquad_ans_0002622-2`, `medquad_ans_0002622-1`
**Unsupported citations**: _none_

## Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Top docs**:

1. `medquad_ans_0000032-1__dup4` (score=0.7000) - Diabetes is a disorder of metabolismthe way our bodies use digested food for energy. Most of the food we eat is broken down into glucose, the form of sugar in the blood. Glucose is the body's main source of fuel. After digestion, glucose...
2. `medquad_ans_0000035-1__dup4` (score=0.5256) - Diabetes is a complex group of diseases with a variety of causes. People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia. Diabetes is a disorder of metabolismthe way the body uses digested food for en...
3. `medquad_ans_0000070-1__dup6` (score=0.4941) - Diabetes is a complex group of diseases with a variety of causes. People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia. Diabetes is a disorder of metabolismthe way the body uses digested food for en...
4. `medquad_ans_0000027-1__dup5` (score=0.4941) - Diabetes is a complex group of diseases with a variety of causes. People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia. Diabetes is a disorder of metabolismthe way the body uses digested food for en...
5. `medquad_ans_0000037-1__dup4` (score=0.4941) - Diabetes is a complex group of diseases with a variety of causes. People with diabetes have high blood glucose, also called high blood sugar or hyperglycemia. Diabetes is a disorder of metabolismthe way the body uses digested food for en...

**Answer**:

Diabetes is caused by the body not making enough insulin or being unable to use insulin effectively, or both [medquad_ans_0000032-1__dup4]. The pancreas, which produces insulin, may fail to generate sufficient amounts of the hormone, or the cells in the muscles, liver, and fat may not respond properly to insulin [medquad_ans_0000035-1__dup4]. This results in high blood glucose levels (hyperglycemia) as glucose remains in the bloodstream instead of being absorbed by cells [medquad_ans_0000035-1__dup4].

**Citations detected**: `medquad_ans_0000032-1__dup4`, `medquad_ans_0000035-1__dup4`
**Supported citations**: `medquad_ans_0000032-1__dup4`, `medquad_ans_0000035-1__dup4`
**Unsupported citations**: _none_

## Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Top docs**:

1. `medquad_ans_0000016-21` (score=0.7000) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the laboratory and with pat...
2. `medquad_ans_0000016-8` (score=0.5906) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a comprehensive dilated e...
3. `medquad_ans_0000016-6` (score=0.5748) - Diabetic retinopathy is a complication of diabetes and a leading cause of blindness. It occurs when diabetes damages the tiny blood vessels inside the retina, the light-sensitive tissue at the back of the eye. A healthy retina is necessa...
4. `medquad_ans_0000055-7` (score=0.5631) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need other treatments. You wi...
5. `medquad_ans_0000016-4__dup2` (score=0.5248) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels...

**Answer**:

Research being done for diabetic retinopathy includes studies in the laboratory and with patients to find better ways to detect, treat, and prevent vision loss in people with diabetes. Specifically, researchers are studying drugs for the treatment of proliferative retinopathy that may reduce the need for laser surgery [medquad_ans_0000016-21]. Additionally, clinical trials are exploring treatments such as medicines injected into the eye to block proteins that cause abnormal blood vessel growth and fluid leakage, which can help treat macular edema [medquad_ans_0000055-7]. These treatments are administered by ophthalmologists and may include focal or scatter laser treatments depending on the condition [medquad_ans_0000055-7].

**Citations detected**: `medquad_ans_0000016-21`, `medquad_ans_0000055-7`
**Supported citations**: `medquad_ans_0000016-21`, `medquad_ans_0000055-7`
**Unsupported citations**: _none_

## Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.7000) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart and lung diseases wors...
2. `medquad_ans_0000007-4__dup1` (score=0.6031) - People at risk for ARDS have a condition or illness that can directly or indirectly injure their lungs. Direct Lung Injury Conditions that can directly injure the lungs include: Pneumonia. This is an infection in the lungs. Breathing in ...
3. `medquad_ans_0000007-3__dup1` (score=0.3838) - Many conditions or factors can directly or indirectly injure the lungs and lead to ARDS. Some common ones are: Sepsis. This is a condition in which bacteria infect the bloodstream. Pneumonia. This is an infection in the lungs. Severe ble...
4. `medquad_ans_0000137-1__dup1` (score=0.2762) - Acute respiratory distress syndrome (ARDS) is a life-threatening lung condition that prevents enough oxygen from getting to the lungs and into the blood. People who develop ARDS often are very ill with another disease or have major injur...
5. `medquad_ans_0000007-1__dup1` (score=0.2673) - ARDS, or acute respiratory distress syndrome, is a lung condition that leads to low oxygen levels in the blood. ARDS can be life threatening because your body's organs need oxygen-rich blood to work well. People who develop ARDS often ar...

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles in the air from fires or toxic fumes. These substances can damage the eyes and respiratory system and may worsen chronic heart and lung diseases. Symptoms of inhalation injuries include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, it may worsen [medquad_ans_0000516-1__dup1].

**Citations detected**: `medquad_ans_0000516-1__dup1`
**Supported citations**: `medquad_ans_0000516-1__dup1`
**Unsupported citations**: _none_

## Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Top docs**:

1. `medquad_ans_0000134-4__dup2` (score=0.7000) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has a parent with MEN1 i...
2. `medquad_ans_0000134-9` (score=0.5903) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing Zollinger-Ellison syndrome.
3. `medquad_ans_0000195-2` (score=0.3506) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0000134-3__dup2` (score=0.3321) - Experts do not know the exact cause of Zollinger-Ellison syndrome. About 25 to 30 percent of gastrinomas are caused by an inherited genetic disorder called multiple endocrine neoplasia type 1 (MEN1).1 MEN1 causes hormone-releasing tumors...
5. `medquad_ans_0000955-2` (score=0.2739) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.

**Answer**:

The incidence of Zollinger-Ellison syndrome is estimated to affect 1 in 40,000 to 50,000 people [medquad_ans_0000195-2]. Additionally, it is rare and occurs in about one in every 1 million people [medquad_ans_0000134-4__dup2]. However, the prevalence of the condition is unknown [medquad_ans_0000955-2].

**Citations detected**: `medquad_ans_0000195-2`, `medquad_ans_0000134-4__dup2`, `medquad_ans_0000955-2`
**Supported citations**: `medquad_ans_0000195-2`, `medquad_ans_0000134-4__dup2`, `medquad_ans_0000955-2`
**Unsupported citations**: _none_

## Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Top docs**:

1. `medquad_ans_0000974-1` (score=0.7000) - Tetrahydrobiopterin deficiency is a rare disorder characterized by a shortage (deficiency) of a molecule called tetrahydrobiopterin or BH4. This condition alters the levels of several substances in the body, including phenylalanine. Phen...
2. `medquad_ans_0006044-2` (score=0.5831) - What are the signs and symptoms of Tetrahydrobiopterin deficiency? Infants with tetrahydrobiopterin (BH4) deficiency typically appear normal and healthy at birth. Neurological signs and symptoms usually become apparent over time, and can...
3. `medquad_ans_0006044-1` (score=0.4625) - Tetrahydrobiopterin (BH4) deficiency is a neurological condition caused by an inborn error of metabolism. BH4 is a substance in the body that enhances the action of other enzymes. Deficiency of BH4 leads to abnormally high blood levels o...
4. `medquad_ans_0000974-5` (score=0.4552) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Genetic Testing Regist...
5. `medquad_ans_0000974-3` (score=0.3981) - Tetrahydrobiopterin deficiency can be caused by mutations in one of several genes, including GCH1, PCBD1, PTS, and QDPR. These genes provide instructions for making enzymes that help produce and recycle tetrahydrobiopterin in the body. T...

**Answer**:

The provided context does not mention any treatments for tetrahydrobiopterin deficiency. Therefore, based on the available information, it is not possible to answer the question about treatments. [medquad_ans_0000974-1] [medquad_ans_0006044-2]

**Citations detected**: `medquad_ans_0000974-1`, `medquad_ans_0006044-2`
**Supported citations**: `medquad_ans_0000974-1`, `medquad_ans_0006044-2`
**Unsupported citations**: _none_

## Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Top docs**:

1. `medquad_ans_0000736-2` (score=0.7000) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
2. `medquad_ans_0000150-2__dup1` (score=0.2581) - There is no standard course of treatment for holoprosencephaly. Treatment is symptomatic and supportive.
3. `medquad_ans_0000736-1` (score=0.2251) - Nonsyndromic holoprosencephaly is an abnormality of brain development that also affects the head and face. Normally, the brain divides into two halves (hemispheres) during early development. Holoprosencephaly occurs when the brain fails ...
4. `medquad_ans_0000195-2` (score=0.2068) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
5. `medquad_ans_0000955-2` (score=0.1811) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.

**Answer**:

Holoprosencephaly affects an estimated 1 in 10,000 newborns, and nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases [medquad_ans_0000736-2]. Therefore, the number of people affected by nonsyndromic holoprosencephaly can be estimated as 25 to 50 percent of 1 in 10,000 newborns, which translates to approximately 1 in 20,000 to 1 in 40,000 newborns. However, the exact number of individuals affected is not specified in the provided context. [medquad_ans_0000736-2]

**Citations detected**: `medquad_ans_0000736-2`
**Supported citations**: `medquad_ans_0000736-2`
**Unsupported citations**: _none_

## Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Top docs**:

1. `medquad_ans_0000525-4` (score=0.7000) - Susceptibility to intrahepatic cholestasis of pregnancy is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to increase the risk of developing the disorder. Some women with a...
2. `medquad_ans_0000118-5` (score=0.6211) - These resources address the diagnosis or management of benign recurrent intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Benign recurrent intrahepatic cholestasis 1 - Genetic Testing Reg...
3. `medquad_ans_0000829-5` (score=0.5913) - These resources address the diagnosis or management of progressive familial intrahepatic cholestasis: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Progressive familial intrahepatic cholestasis 2 - Genetic Tes...
4. `medquad_ans_0005084-1` (score=0.5767) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching, jaundice, failure ...
5. `medquad_ans_0003240-1` (score=0.4222) - Intrahepatic cholestasis of pregnancy (ICP) is a disorder of the liver that occurs in women during pregnancy. Cholestasis is a condition that impairs the release of bile (a digestive juice) from liver cells. The bile then builds up in th...

**Answer**:

The genetic change related to progressive familial intrahepatic cholestasis type 2 (PFIC2) is a mutation in the ABCB11 gene. This condition is inherited in an autosomal recessive manner [medquad_ans_0005084-1]. The provided context does not specify genetic changes for other types of progressive familial intrahepatic cholestasis beyond PFIC2.

**Citations detected**: `medquad_ans_0005084-1`
**Supported citations**: `medquad_ans_0005084-1`
**Unsupported citations**: _none_

## Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.7000) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below includes how often the...
2. `medquad_ans_0003814-2` (score=0.3741) - What are the signs and symptoms of Marshall-Smith syndrome? Marshall-Smith syndrome is characterized by accelerated skeletal maturation, relative failure to thrive, respiratory difficulties, mental retardation, and unusual facies, includ...
3. `medquad_ans_0003845-2` (score=0.3431) - What are the signs and symptoms of McCune Albright syndrome? People with McCune Albright syndrome (MAS) may have symptoms related to bones, the endocrine system, and/or skin. The symptoms can range from mild to severe. Bone symptoms may ...
4. `medquad_ans_0001983-1` (score=0.1271) - What are the signs and symptoms of Dyssegmental dysplasia Silverman-Handmaker type? The Human Phenotype Ontology provides the following list of signs and symptoms for Dyssegmental dysplasia Silverman-Handmaker type. If the information is...
5. `medquad_ans_0002658-1` (score=0.1177) - What are the signs and symptoms of Gracile bone dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Gracile bone dysplasia. If the information is available, the table below includes how often the...

**Answer**:

The signs and symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Citations detected**: `medquad_ans_0005678-1`
**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

## Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Top docs**:

1. `medquad_ans_0000453-1` (score=0.7000) - Hepatic veno-occlusive disease with immunodeficiency (also called VODI) is a hereditary disorder of the liver and immune system. Its signs and symptoms appear after the first few months of life. Hepatic veno-occlusive disease is a condit...
2. `medquad_ans_0000453-5` (score=0.4405) - These resources address the diagnosis or management of VODI: - Gene Review: Gene Review: Hepatic Veno-Occlusive Disease with Immunodeficiency - Genetic Testing Registry: Hepatic venoocclusive disease with immunodeficiency These resources...
3. `medquad_ans_0002804-1` (score=0.2729) - What are the signs and symptoms of Hepatic venoocclusive disease with immunodeficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Hepatic venoocclusive disease with immunodeficiency. If the inform...
4. `medquad_ans_0000324-4` (score=0.1267) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
5. `medquad_ans_0005558-3` (score=0.1069) - How is severe combined immunodeficiency (SCID) diagnosed? A diagnosis of severe combined immunodeficiency (SCID) may be suspected if a baby shows any of the following persistent symptoms within the first year of life: Eight or more ear i...

**Answer**:

Yes, hepatic veno-occlusive disease with immunodeficiency is inherited in an autosomal recessive manner. [medquad_ans_0002804-1]

**Citations detected**: `medquad_ans_0002804-1`
**Supported citations**: `medquad_ans_0002804-1`
**Unsupported citations**: _none_

## Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Top docs**:

1. `medquad_ans_0005902-1` (score=0.7000) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how often the symptom is ...
2. `medquad_ans_0001768-1` (score=0.5380) - What are the signs and symptoms of Dehydrated hereditary stomatocytosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Dehydrated hereditary stomatocytosis. If the information is available, the table ...
3. `medquad_ans_0000013_3-2__dup4` (score=0.2679) - Signs and symptoms of juvenile myelomonocytic leukemia include fever, weight loss, and feeling very tired. These and other signs and symptoms may be caused by JMML or by other conditions. Check with your doctor if you have any of the fol...
4. `medquad_ans_0002784-1` (score=0.2326) - What are the signs and symptoms of Hemophagocytic lymphohistiocytosis, familial, 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Hemophagocytic lymphohistiocytosis, familial, 4. If the information is...
5. `medquad_ans_0002781-2` (score=0.2264) - What are the signs and symptoms of Hemophagocytic lymphohistiocytosis? The signs and symptoms of hemophagocytic lymphohistiocytosis typically develop during the first months or years of life. However, in rare cases, affected people may n...

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. These signs and symptoms are based on the Human Phenotype Ontology, which compiles data from sources such as Orphanet [medquad_ans_0005902-1].

**Citations detected**: `medquad_ans_0005902-1`
**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

## Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Top docs**:

1. `medquad_ans_0000159-8` (score=0.7000) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed through urine and b...
2. `medquad_ans_0003688-1` (score=0.5760) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, high blood pressure, ...
3. `medquad_ans_0000159-6` (score=0.5586) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephritis develop kidney f...
4. `medquad_ans_0000159-7` (score=0.5214) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may need to make to their...
5. `medquad_ans_0000159-3__dup2` (score=0.5122) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time or shortly after lu...

**Answer**:

For lupus nephritis, treatment involves medications that suppress the immune system to prevent damage to the kidneys. Standard treatment typically includes a corticosteroid, such as prednisone, to reduce inflammation and an immunosuppressive medication, such as cyclophosphamide or mycophenolate mofetil, used in combination with prednisone. If high blood pressure is present, medications to lower blood pressure are also prescribed, as this can slow the progression of kidney disease. In some cases, treatment may involve dialysis or a kidney transplant. Early diagnosis and treatment are crucial to prevent chronic kidney damage and complications such as kidney failure, which affects 10 to 30 percent of people with lupus nephritis [medquad_ans_0000159-8][medquad_ans_0003688-1]. People with lupus nephritis are also at a high risk for certain

**Citations detected**: `medquad_ans_0000159-8`, `medquad_ans_0003688-1`
**Supported citations**: `medquad_ans_0000159-8`, `medquad_ans_0003688-1`
**Unsupported citations**: _none_

## Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Top docs**:

1. `medquad_ans_0000114-6` (score=0.7000) - Restless legs syndrome (RLS) has no cure. If a condition or medicine triggers RLS, it may go away or get better if the trigger is relieved or stopped. RLS can be treated. The goals of treatment are to: Prevent or relieve symptoms Increas...
2. `medquad_ans_0005317-1` (score=0.4953) - Restless legs syndrome is a neurological condition that causes an irresistible urge to move the legs. The movement is triggered by strange or uncomfortable feelings, which occur mostly while the affected person is sitting or lying down a...
3. `medquad_ans_0000776-1__dup2` (score=0.3803) - Restless legs syndrome (RLS) causes a powerful urge to move your legs. Your legs become uncomfortable when you are lying down or sitting. Some people describe it as a creeping, crawling, tingling, or burning sensation. Moving makes your ...
4. `medquad_ans_0000114-5__dup1` (score=0.3509) - Your doctor will diagnose restless legs syndrome (RLS) based on your signs and symptoms, your medical and family histories, a physical exam, and test results. Your doctor will use this information to rule out other conditions that have s...
5. `medquad_ans_0000246-1__dup2` (score=0.3264) - Restless legs syndrome (RLS) is a neurological disorder characterized by unpleasant sensations in the legs and an uncontrollable, and sometimes overwhelming, urge to move them for relief. Individuals affected with the disorder often desc...

**Answer**:

The treatments for Restless Legs Syndrome (RLS) include lifestyle changes and, in more severe cases, daily medicines. For mild cases, lifestyle changes may be the only treatment needed. These include avoiding substances that trigger symptoms, such as alcohol and tobacco, and informing your doctor about all medications you are taking, as some can cause or worsen RLS. Good sleep habits, such as keeping the sleep environment cool, quiet, dark, and comfortable, going to bed and waking up at the same time every day, and avoiding long periods of lying awake, can also help. Engaging in regular, moderate physical activity and doing mentally challenging tasks before bedtime may ease symptoms. Certain activities that relieve symptoms include walking, stretching, taking a hot or cold bath, massaging the affected limb(s), or using heat or ice packs. If lifestyle changes are insufficient, medicines may be needed, as no single medicine works for all people, and it may take several adjustments

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Top docs**:

1. `medquad_ans_0002575-1` (score=0.7000) - Gliomatosis cerebri is a type of brain cancer. It is a variant form of glioblastoma multiforme. It is characterized by scattered and widespread tumor cells that can cause the cerebrum, cerebellum, or brain stem to enlarge. Signs and symp...
2. `medquad_ans_0002574-1` (score=0.6656) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of glial cell is involved...
3. `medquad_ans_0004525-1` (score=0.4044) - Oligoastrocytoma is a brain tumor that forms when two types of cells in the brain, called oligodendrocytes and astrocytes, rapidly increase in number to form a mass. These brain cells are known as glial cells, which normally protect and ...
4. `medquad_ans_0004363-1` (score=0.3251) - Neuroblastoma is a tumor that develops from a nerve in a child, usually before the age of 5. It occurs in the abdomen near the adrenal glands, but it can also occur in other parts of the body. It is considered an aggressive tumor because...
5. `medquad_ans_0000006_5-1` (score=0.2704) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - There are two types o...

**Answer**:

The provided context does not contain information about glioblastoma. While it mentions gliomatosis cerebri as a variant form of glioblastoma multiforme, it does not define or describe glioblastoma itself. Therefore, based on the available context, the answer cannot be determined.

[medquad_ans_0002575-1]

**Citations detected**: `medquad_ans_0002575-1`
**Supported citations**: `medquad_ans_0002575-1`
**Unsupported citations**: _none_

## Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Top docs**:

1. `medquad_ans_0005496-1` (score=0.7000) - Schizencephaly is a rare congenital (present from birth) brain malformation in which abnormal slits or clefts form in the cerebral hemispheres of the brain. The signs and symptoms of this condition may include developmental delay, seizur...
2. `medquad_ans_0000251-1__dup2` (score=0.5396) - Schizencephaly is an extremely rare developmental birth defect characterized by abnormal slits, or clefts, in the cerebral hemispheres of the brain. Babies with clefts in both hemispheres (called bilateral clefts) commonly have developme...
3. `medquad_ans_0000251-3__dup1` (score=0.2715) - The prognosis for individuals with schizencephaly varies depending on the size of the clefts and the extent of neurological disabilities.
4. `medquad_ans_0000064-1__dup1` (score=0.1874) - Cephalic disorders are congenital conditions that stem from damage to or abnormal development of the budding nervous system. Most cephalic disorders are caused by a disturbance that occurs very early in the development of the fetal nervo...
5. `medquad_ans_0000251-4__dup1` (score=0.1401) - The NINDS conducts and supports a wide range of studies that explore the mechanisms of normal brain development. The knowledge gained from these fundamental studies provides the foundation for understanding how to prevent or treat develo...

**Answer**:

The exact cause of schizencephaly is unknown, but it has been linked to a variety of genetic and non-genetic factors [medquad_ans_0005496-1]. Additionally, some cephalic disorders, which include conditions affecting the development of the nervous system, may be influenced by hereditary or genetic conditions or by environmental exposures during pregnancy, such as medication taken by the mother, maternal infection, or exposure to radiation [medquad_ans_0000064-1]. However, specific environmental or genetic causes directly tied to schizencephaly are not detailed in the provided context.

**Citations detected**: `medquad_ans_0005496-1`, `medquad_ans_0000064-1`
**Supported citations**: `medquad_ans_0005496-1`
**Unsupported citations**: `medquad_ans_0000064-1`

## Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Top docs**:

1. `medquad_ans_0000309-1__dup1` (score=0.7000) - Amyotrophic lateral sclerosis (ALS), also referred to as "Lou Gehrig's disease," is a progressive motor neuron disease which leads to problems with muscle control and movement. There are various types of ALS, which are distinguished by t...
2. `medquad_ans_0000309-3__dup1` (score=0.5559) - What causes amyotrophic lateral sclerosis? In approximately 90-95% of cases the cause of amyotrophic lateral sclerosis (ALS) is unknown and is sporadic (occurring in individuals with no history of the condition in the family). The remain...
3. `medquad_ans_0000052-5` (score=0.5402) - These resources address the diagnosis or management of amyotrophic lateral sclerosis: - Gene Review: Gene Review: ALS2-Related Disorders - Gene Review: Gene Review: Amyotrophic Lateral Sclerosis Overview - Gene Review: Gene Review: C9orf...
4. `medquad_ans_0000052-3` (score=0.5315) - Mutations in several genes can cause familial ALS and contribute to the development of sporadic ALS. Mutations in the C9orf72 gene account for 30 to 40 percent of familial ALS in the United States and Europe. Worldwide, SOD1 gene mutatio...
5. `medquad_ans_0003331-3` (score=0.5277) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but it...

**Answer**:

Genetic changes related to amyotrophic lateral sclerosis (ALS) include mutations in several genes, such as C9orf72, SOD1, TARDBP, FUS, ANG, ALS2, SETX, and VAPB. These mutations can cause familial ALS and may also contribute to the development of sporadic ALS. Approximately 60% of individuals with familial ALS have an identifiable genetic mutation, while the genetic cause in the remaining cases is unknown. The genes associated with ALS appear to play a role in neuronal function or in regulating the production of various proteins [doc_id: medquad_ans_0000309-3__dup1].

**Citations detected**: `doc_id: medquad_ans_0000309-3__dup1`
**Supported citations**: _none_
**Unsupported citations**: `doc_id: medquad_ans_0000309-3__dup1`

## Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Top docs**:

1. `medquad_ans_0003473-6` (score=0.7000) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injections of insulin-like g...
2. `medquad_ans_0000581-5` (score=0.4541) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Testing Registry: Laro...
3. `medquad_ans_0003473-1` (score=0.4262) - Laron syndrome is a condition that occurs when the body is unable to utilize growth hormone. It is primarily characterized by short stature. Other signs and symptoms vary but may include reduced muscle strength and endurance; hypoglycemi...
4. `medquad_ans_0000581-2` (score=0.4130) - Laron syndrome is a rare disorder. About 350 people have been diagnosed with the condition worldwide. The largest single group of affected individuals (about 100 people) lives in an area of southern Ecuador.
5. `medquad_ans_0003455-6` (score=0.3492) - How might Lafora disease be treated? Unfortunately, there is currently no cure for Lafora disease or way to slow the progression of the condition. Treatment is based on the signs and symptoms present in each person. For example, certain ...

**Answer**:

The only specific treatment available for Laron syndrome is subcutaneous injections of insulin-like growth factor 1 (IGF-1), which stimulates linear growth, improves brain growth, and addresses metabolic abnormalities caused by long-term IGF-1 deficiency. IGF-1 has also been shown to raise blood glucose levels, reduce cholesterol, and increase muscle growth. Treatment involves closely monitoring IGF-1 and growth hormone levels to avoid overdosage, which can cause health problems [medquad_ans_0003473-6]. Additionally, treatment is focused on improving growth and may include IGF-1 injections as part of a broader management approach [medquad_ans_0003473-1].

**Citations detected**: `medquad_ans_0003473-6`, `medquad_ans_0003473-1`
**Supported citations**: `medquad_ans_0003473-6`, `medquad_ans_0003473-1`
**Unsupported citations**: _none_

## Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Top docs**:

1. `medquad_ans_0000016-15` (score=0.7000) - During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels of blood sugar, blood pressure...
2. `medquad_ans_0000016-4__dup2` (score=0.6375) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels...
3. `medquad_ans_0000055-7` (score=0.6269) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need other treatments. You wi...
4. `medquad_ans_0000016-8` (score=0.3533) - All people with diabetes -- both type 1 and type 2 -- are at risk for diabetic retinopathy. People with diabetes are also at increased risk for cataract and glaucoma. That's why everyone with diabetes should get a comprehensive dilated e...
5. `medquad_ans_0000016-19` (score=0.3005) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although both treatments have h...

**Answer**:

The treatments for diabetic retinopathy include:

- For the first three stages, no treatment is needed unless macular edema is present. To prevent progression, people with diabetes should control their blood sugar, blood pressure, and blood cholesterol levels [medquad_ans_0000016-15].

- For macular edema, prompt treatment with anti-VEGF drugs—either alone or in combination with laser treatment—results in better vision than laser treatment alone or steroid injections. These drugs reduce fluid leakage and inhibit the growth of new blood vessels in the retina. In some cases, focal laser treatment is used alongside injections, where up to several hundred small laser burns are placed around the macula to slow fluid leakage [medquad_ans_0000016-4__dup2].

- Proliferative retinopathy is treated with scatter laser treatment, which involves placing 1,000 to

**Citations detected**: `medquad_ans_0000016-15`, `medquad_ans_0000016-4__dup2`
**Supported citations**: `medquad_ans_0000016-15`, `medquad_ans_0000016-4__dup2`
**Unsupported citations**: _none_

## Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Top docs**:

1. `medquad_ans_0001421-3` (score=0.7000) - How might collagenous colitis be treated? Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases the condition may resolve on its own (spontaneous remission), although m...
2. `medquad_ans_0001421-1` (score=0.4216) - Collagenous colitis is a type of inflammatory bowel disease that affects the colon. It is a form of microscopic colitis, which means that the inflammation is only visible when a biopsy is examined under a microscope; the inflammation can...
3. `medquad_ans_0000122-7` (score=0.2807) - A health care provider treats ulcerative colitis with - medications - surgery Which treatment a person needs depends on the severity of the disease and the symptoms. Each person experiences ulcerative colitis differently, so health care ...
4. `medquad_ans_0000116-8` (score=0.2438) - To help reduce symptoms, a health care provider may recommend the following dietary changes: - avoid foods and drinks that contain caffeine or artificial sugars - drink plenty of liquids to prevent dehydration during episodes of diarrhea...
5. `medquad_ans_0000116-7` (score=0.2046) - Treatment depends on the severity of symptoms. The gastroenterologist will - review the medications the person is taking - make recommendations to change or stop certain medications - recommend that the person quit smoking The gastroente...

**Answer**:

Treatments for collagenous colitis vary depending on the symptoms and severity in each individual. In some cases, the condition may resolve on its own (spontaneous remission), although most people continue to experience ongoing or occasional diarrhea. Dietary changes are usually tried first to alleviate symptoms, including a reduced-fat diet, eliminating caffeine and lactose, and avoiding over-the-counter pain relievers such as ibuprofen or aspirin. If dietary changes are insufficient, medications may be used to control symptoms. Prescription anti-inflammatory medications such as mesalamine and sulfasalazine may help reduce swelling. Steroids like budesonide and prednisone can reduce inflammation but are typically reserved for sudden attacks of diarrhea due to the risk of unwanted side effects with long-term use. Anti-diarrheal medications such as bismuth subsalicylate, diphenoxylate with atropine, and loperamide can provide short-term relief.

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Top docs**:

1. `medquad_ans_0000418-6` (score=0.7000) - Safe and effective prescription drugs are available to treat both Trichinella infection and the symptoms that occur as a result of infection. Treatment should begin as soon as possible; a doctor will make the decision to treat based upon...
2. `medquad_ans_0000418-1__dup2` (score=0.6638) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (meat-eating) animals...
3. `medquad_ans_0000418-2__dup1` (score=0.5672) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or cooking puts you at...
4. `medquad_ans_0000418-7` (score=0.4152) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause of many cases of tri...
5. `medquad_ans_0000418-5__dup1` (score=0.3385) - A diagnosis of trichinellosis is made in patients whose signs and symptoms are compatible with trichinellosis, have a positive laboratory test for Trichinella, and who can recall eating raw or undercooked pork or wild game meat. Laborato...

**Answer**:

To prevent Trichinellosis (also known as Trichinosis), wash your hands with warm water and soap after handling raw meat [medquad_ans_0000418-7]. Additionally, avoid consuming raw or undercooked meat, particularly from wild game such as bear or cougar, or from domestic pigs and wild boar, as these can harbor the Trichinella parasite [medquad_ans_0000418-1\_\_dup2][medquad_ans_0000418-2__dup1]. Curing (salting), drying, smoking, or microwaving meat alone may not be sufficient for prevention; proper cooking is essential to kill the parasite [medquad_ans_0000418-7]. Public awareness of the risks associated with eating raw or undercooked meat products also plays a key role in reducing the incidence of infection [medquad_ans

**Citations detected**: `medquad_ans_0000418-7`, `medquad_ans_0000418-1__dup2`, `medquad_ans_0000418-2__dup1`
**Supported citations**: `medquad_ans_0000418-7`, `medquad_ans_0000418-1__dup2`, `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

## Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Top docs**:

1. `medquad_ans_0000182-1__dup3` (score=0.7000) - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the normal components, or cells, of the liver and causes inflammation and liver damage. The immune system normally protects people from inf...
2. `medquad_ans_0000182-3__dup1` (score=0.6355) - A combination of autoimmunity, environmental triggers, and a genetic predisposition can lead to autoimmune hepatitis.
3. `medquad_ans_0000182-5__dup1` (score=0.4798) - Autoimmune hepatitis is classified into several types. Type 1 autoimmune hepatitis is the most common form in North America. Type 1 can occur at any age; however, it most often starts in adolescence or young adulthood. About 70 percent o...
4. `medquad_ans_0000182-10` (score=0.3717) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing autoimmune hepatitis.
5. `medquad_ans_0000182-11` (score=0.3697) - - Autoimmune hepatitis is a chronicor long lastingdisease in which the body's immune system attacks the liver and causes inflammation and damage. - Autoimmune hepatitis is a serious condition that may worsen over time if not treated. Aut...

**Answer**:

Autoimmune hepatitis is caused by a combination of autoimmunity, environmental triggers, and a genetic predisposition [doc_id: medquad_ans_0000182-3__dup1].

**Citations detected**: `doc_id: medquad_ans_0000182-3__dup1`
**Supported citations**: _none_
**Unsupported citations**: `doc_id: medquad_ans_0000182-3__dup1`

## Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Top docs**:

1. `medquad_ans_0000006_3-1` (score=0.7000) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central nervous system contr...
2. `medquad_ans_0000006_3-4` (score=0.4465) - The signs and symptoms of astrocytomas are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain or spinal cord. - The size of the tumor. - How fast the tumor grows. - The child's a...
3. `medquad_ans_0000324-1__dup1` (score=0.3073) - Anaplastic astrocytoma is a rare, cancerous (malignant) type of brain tumor that arises from star-shaped brain cells called astrocytes. These cells surround and protect nerve cells in the brain and spinal cord. An anaplastic astrocytoma ...
4. `medquad_ans_0004883-4` (score=0.2685) - Are pilocytic astrocytomas inherited? Pilocytic astrocytomas are typically sporadic, occurring by chance in individuals with no history of the condition in the family. Sporadic abnormalities are not inherited from a parent and are not li...
5. `medquad_ans_0004883-1` (score=0.2253) - Pilocytic astrocytoma is an often benign, slow-growing tumor of the brain or spinal cord. The tumor may be in the form of a cyst and usually does not spread to nearby tissues. Symptoms vary depending upon the size and location of the tum...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They are tumors that start in star-shaped brain cells called astrocytes, which are a type of glial cell that holds nerve cells in place, brings food and oxygen to them, and helps protect them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Citations detected**: `medquad_ans_0000006_3-1`
**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

## Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Top docs**:

1. `medquad_ans_0000013_3-2__dup3` (score=0.7000) - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising or bleeding and feeling tired and weak. These and other signs and symptoms may be caused by atypical CML or by other conditions. Check with your doctor if ...
2. `medquad_ans_0000013_3-1__dup3` (score=0.5719) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising ...
3. `medquad_ans_0000001_4-1` (score=0.4720) - Key Points - Chronic myelogenous leukemia is a disease in which the bone marrow makes too many white blood cells. - Leukemia may affect red blood cells, white blood cells, and platelets. - Signs and symptoms of chronic myelogenous leukem...
4. `medquad_ans_0000013_2-1__dup3` (score=0.4343) - Chronic myelogenous leukemia is a disease in which too many white blood cells are made in the bone marrow. See the PDQ summary on Chronic Myelogenous Leukemia Treatment for information on diagnosis, staging, and treatment.
5. `medquad_ans_0000013_2-2__dup3` (score=0.4049) - See the PDQ summary about Chronic Myelogenous Leukemia Treatment for information.

**Answer**:

Atypical chronic myelogenous leukemia (CML) is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. In this condition, the body tells too many blood stem cells to become granulocytes, and some of these stem cells never become mature white blood cells; instead, they remain as immature cells called blasts. Over time, these granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. Atypical CML is similar to typical CML in that the leukemia cells look alike under a microscope, but it does not have the "Philadelphia chromosome" chromosome change that is typically present in typical CML [medquad_ans_0000013_3-1__dup3]. Signs and symptoms may include easy bruising or bleeding, feeling tired and weak, shortness of breath, pale skin, petechiae, and pain or a

**Citations detected**: `medquad_ans_0000013_3-1__dup3`
**Supported citations**: `medquad_ans_0000013_3-1__dup3`
**Unsupported citations**: _none_

## Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Top docs**:

1. `medquad_ans_0000300-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
2. `medquad_ans_0000676-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000990-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
4. `medquad_ans_0000685-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000770-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...

**Answer**:

The provided context does not contain any information about genetic changes related to trisomy 13. Therefore, the question cannot be answered based on the given context. [no_context]

**Citations detected**: `no_context`
**Supported citations**: _none_
**Unsupported citations**: `no_context`

## Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Top docs**:

1. `medquad_ans_0004415-3` (score=0.7000) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respiratory problems A cardio...
2. `medquad_ans_0000218-2__dup1` (score=0.6591) - There is currently no cure for Niemann-Pick disease. Treatment is supportive. Children usually die from infection or progressive neurological loss. There is currently no effective treatment for persons with type A. Bone marrow transplant...
3. `medquad_ans_0004415-1` (score=0.4003) - Niemann-Pick disease is an inherited condition involving lipid metabolism, which is the breakdown, transport, and use of fats and cholesterol in the body. In people with this condition, abnormal lipid metabolism causes harmful amounts of...
4. `medquad_ans_0000218-1__dup1` (score=0.3288) - Niemann-Pick disease (NP) refers to a group of inherited metabolic disorders known as lipid storage diseases. Lipids (fatty materials such as waxes, fatty acids, oils, and cholesterol) and proteins are usually broken down into smaller co...
5. `medquad_ans_0000731-1` (score=0.3056) - Niemann-Pick disease is a condition that affects many body systems. It has a wide range of symptoms that vary in severity. Niemann-Pick disease is divided into four main types: type A, type B, type C1, and type C2. These types are classi...

**Answer**:

There is no specific treatment for Niemann-Pick disease type A. Treatment is supportive and may include care from specialists such as a pulmonologist for respiratory problems, a cardiologist for heart problems, liver and spleen specialists, nutritionists, physical therapists, and gastroenterologists, as well as learning specialists [medquad_ans_0004415-3]. There is currently no effective treatment for persons with type A, and no cure exists for the disease [medquad_ans_0000218-2__dup1]. Bone marrow transplantation has been attempted in a few individuals with type B, and enzyme replacement and gene therapies might be helpful for type B, but these are not applicable to type A [medquad_ans_0000218-2__dup1]. Restricting one's diet does not prevent the buildup of lipids in cells and tissues [medquad_ans_00

**Citations detected**: `medquad_ans_0004415-3`, `medquad_ans_0000218-2__dup1`
**Supported citations**: `medquad_ans_0004415-3`, `medquad_ans_0000218-2__dup1`
**Unsupported citations**: _none_

## Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Top docs**:

1. `medquad_ans_0001714-1` (score=0.7000) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial polydactyly. If the info...
2. `medquad_ans_0001708-1` (score=0.3548) - What are the signs and symptoms of Dandy-Walker like malformation with atrioventricular septal defect? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker like malformation with atrioventricula...
3. `medquad_ans_0001710-1` (score=0.2137) - What are the signs and symptoms of Dandy-Walker malformation with facial hemangioma? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with facial hemangioma. If the information ...
4. `medquad_ans_0000270-5` (score=0.1577) - These resources address the diagnosis or management of Dandy-Walker malformation: - Genetic Testing Registry: Dandy-Walker syndrome - National Hydrocephalus Foundation: Treatment of Hydrocephalus These resources from MedlinePlus offer in...
5. `medquad_ans_0003881-2` (score=0.1567) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speech Drooling

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, autosomal recessive inheritance, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, and wide nasal bridge [

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Top docs**:

1. `medquad_ans_0003789-1` (score=0.7000) - Mantle cell lymphoma (MCL) belongs to a group of diseases known as non-Hodgkins lymphomas (NHL), which are cancers that affect the the lymphatic system (part of the immune system). MCL accounts for 6% of all non-Hodgkin lymphomas and is ...
2. `medquad_ans_0003789-5` (score=0.4527) - How might Mantle cell lympoma be treated? Various treatmentsare currently available for Mantle cell lymphomas. Rare cases of early stage mantle cell lymphomas may be treated with radiation therapy. For more advance stagestreatment includ...
3. `medquad_ans_0003789-4` (score=0.3921) - How is Mantle cell lymphoma diagnosed? Mantle cell lymphoma is diagnosed by a biopsy (surgical removal) of the lymph nodes. If lymph nodes are not easily accessible to be biopsied, a fine needle aspiration may be performed, but the diagn...
4. `medquad_ans_0003789-2` (score=0.3692) - What are the signs and symptoms of Mantle cell lymphoma? Common symptoms of Mantle cell lymphoma include fatigue, loss of appetite, and enlarged lymph nodes, spleen, and/or liver. Other symptoms may include night sweats, unexplained high...
5. `medquad_ans_0003471-1` (score=0.1477) - Large granular lymphocyte (LGL) leukemia is a rare cancer of a type of white blood cells called lymphocytes. LGL leukemia causes a slow increase in white blood cells called T lymphocytes, or T cells, which originate in the lymph system a...

**Answer**:

The provided context does not specify the cause of Mantle cell lymphoma. [medquad_ans_0003789-1], [medquad_ans_0003789-5], [medquad_ans_0003789-4], and [medquad_ans_0003789-2] discuss diagnosis, treatment, symptoms, and classification but do not address the underlying cause of the disease. Therefore, based on the given information, the cause of Mantle cell lymphoma cannot be determined.

**Citations detected**: `medquad_ans_0003789-1`, `medquad_ans_0003789-5`, `medquad_ans_0003789-4`, `medquad_ans_0003789-2`
**Supported citations**: `medquad_ans_0003789-1`, `medquad_ans_0003789-5`, `medquad_ans_0003789-4`, `medquad_ans_0003789-2`
**Unsupported citations**: _none_

## Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Top docs**:

1. `medquad_ans_0000589-5` (score=0.7000) - These resources address the diagnosis or management of Legg-Calv-Perthes disease: - National Osteonecrosis Foundation - Seattle Children's Hospital These resources from MedlinePlus offer information about the diagnosis and management of ...
2. `medquad_ans_0000589-1` (score=0.6511) - Legg-Calv-Perthes disease is a bone disorder that affects the hips. Usually, only one hip is involved, but in about 10 percent of cases, both hips are affected. Legg-Calv-Perthes disease begins in childhood, typically between ages 4 and ...
3. `medquad_ans_0000589-3` (score=0.6115) - Legg-Calv-Perthes disease is usually not caused by genetic factors. The cause in these cases is unknown. In a small percentage of cases, mutations in the COL2A1 gene cause the bone abnormalities characteristic of Legg-Calv-Perthes diseas...
4. `medquad_ans_0006597-4` (score=0.2728) - Can 48,XXYY syndrome be inherited?
5. `medquad_ans_0003078-4` (score=0.1278) - How is hypophosphatemic rickets inherited? Hypophosphatemic rickets is most often inherited in an X-linked dominant manner. This means that the gene responsible for the condition is located on the X chromosome, and having only one mutate...

**Answer**:

Legg-Calv-Perthes disease is usually not caused by genetic factors, and the cause in most cases is unknown. However, in a small percentage of cases, mutations in the COL2A1 gene are associated with the bone abnormalities characteristic of the disease [medquad_ans_0000589-3]. Therefore, while it is not typically inherited, there can be a genetic component in rare instances. [medquad_ans_0000589-3]

**Citations detected**: `medquad_ans_0000589-3`
**Supported citations**: `medquad_ans_0000589-3`
**Unsupported citations**: _none_

## Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Top docs**:

1. `medquad_ans_0000647-2` (score=0.7000) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigrated to Israel from s...
2. `medquad_ans_0000647-5` (score=0.6243) - These resources address the diagnosis or management of metachromatic leukodystrophy: - Gene Review: Gene Review: Arylsulfatase A Deficiency - Genetic Testing Registry: Metachromatic leukodystrophy - Genetic Testing Registry: Sphingolipid...
3. `medquad_ans_0000647-1` (score=0.6170) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the substance that insulate...
4. `medquad_ans_0003566-1` (score=0.5586) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to defects of the myelin, ...
5. `medquad_ans_0000192-1__dup1` (score=0.5470) - Metachromatic leukodystrophy (MLD) is one of a group of genetic disorders called the leukodystrophies, which are characterized by the toxic buildup of lipids (fatty materials such as oils and waxes) and other storage materials in cells i...

**Answer**:

Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells, particularly affecting cells in the nervous system that produce myelin, the substance that insulates and protects nerves. This accumulation leads to progressive destruction of white matter (leukodystrophy) throughout the nervous system, including the brain, spinal cord, and peripheral nerves. The damage results in progressive deterioration of intellectual functions and motor skills, such as the ability to walk, loss of sensation in the extremities (peripheral neuropathy), incontinence, seizures, paralysis, inability to speak, blindness, and hearing loss. Eventually, individuals may lose awareness of their surroundings and become unresponsive. While neurological problems are the primary feature, sulfatide accumulation can also affect other organs, most commonly the gallbladder. The most common form, affecting about 50 to 60 percent of individuals, is the late infantile form,

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Top docs**:

1. `medquad_ans_0005135-1` (score=0.7000) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the table below includes how...
2. `medquad_ans_0001983-1` (score=0.4144) - What are the signs and symptoms of Dyssegmental dysplasia Silverman-Handmaker type? The Human Phenotype Ontology provides the following list of signs and symptoms for Dyssegmental dysplasia Silverman-Handmaker type. If the information is...
3. `medquad_ans_0005835-1` (score=0.3228) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia with multiple dislocations? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia with multiple dislocation...
4. `medquad_ans_0000094-1__dup1` (score=0.3129) - What are the signs and symptoms of Acromesomelic dysplasia Maroteaux type? The Human Phenotype Ontology provides the following list of signs and symptoms for Acromesomelic dysplasia Maroteaux type. If the information is available, the ta...
5. `medquad_ans_0000077-1__dup1` (score=0.2720) - What are the signs and symptoms of Acrofacial dysostosis Catania type? The Human Phenotype Ontology provides the following list of signs and symptoms for Acrofacial dysostosis Catania type. If the information is available, the table belo...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies (90%), elbow dislocation (90%), hypoplasia of the zygomatic bone (90%), scoliosis (90%), omphalocele (7.5%), hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, tongue-like lumbar vertebral deformities [medquad_ans_0005135-1].

**Citations detected**: `medquad_ans_0005135-1`
**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

## Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Top docs**:

1. `medquad_ans_0001664-3` (score=0.7000) - What causes cri du chat syndrome? Cri du chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected individuals but stu...
2. `medquad_ans_0000220-1__dup3` (score=0.1998) - Interstitial cystitis\*painful bladder syndrome (IC/PBS) is one of several conditions that causes bladder pain and a need to urinate frequently and urgently. Some doctors have started using the term bladder pain syndrome (BPS) to describe...
3. `medquad_ans_0006007-1` (score=0.1111) - TAR syndrome is characterized by the absence of a bone called the radius in each forearm, short stature, and thrombocytopenia. The thrombocytopenia often appears first in infancy but becomes less severe or returns to normal over time. In...
4. `medquad_ans_0004775-1` (score=0.0769) - Pearson syndrome is a mitochondrial disorder characterized by transfusion-dependent sideroblastic anemia and pancreatic dysfunction resulting in in malabsorption and chronic diarrhea. The features of this progressive disorder may change ...
5. `medquad_ans_0004143-1` (score=0.0733) - Mowat-Wilson syndrome (MWS) is a rare genetic disorder that affects many systems of the body. The main features include moderate to severe intellectual disability, distinctive facial features, and epilepsy. Other features may include Hir...

**Answer**:

Cri-du-chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5, written as 5p-. The size of the deletion varies among affected individuals, with larger deletions generally leading to more severe intellectual disability and developmental delay than smaller deletions. The signs and symptoms are likely due to the loss of multiple genes on the short arm of chromosome 5. Researchers believe that the loss of a specific gene, CTNND2, is associated with severe intellectual disability in some individuals with this condition. [medquad_ans_0001664-3]

**Citations detected**: `medquad_ans_0001664-3`
**Supported citations**: `medquad_ans_0001664-3`
**Unsupported citations**: _none_

## Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Top docs**:

1. `medquad_ans_0000719-2__dup1` (score=0.7000) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmental delay Seizures D...
2. `medquad_ans_0000718-1__dup1` (score=0.5255) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area located behind the Sylvius...
3. `medquad_ans_0003881-2` (score=0.3436) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speech Drooling
4. `medquad_ans_0000719-1__dup1` (score=0.3324) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides of the brain (bilate...
5. `medquad_ans_0004960-2` (score=0.2265) - What are the symptoms of polymyositis? Polymyositis is characterized by chronic muscle inflammation and weakness involving the skeletal muscles (those involved with making movement) on both sides of the body. Weakness generally starts in...

**Answer**:

The signs and symptoms of bilateral frontoparietal polymicrogyria may include moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze, ataxia, strabismus, increased muscle tone, finger dysmetria, ankle clonus, Babinski sign, broad-based gait, cerebellar hypoplasia, esotropia, exotropia, hyperreflexia, hypertonia, hypoplasia of the brainstem, nystagmus, polymicrogyria with an anterior to posterior gradient, and truncal ataxia [medquad_ans_0000719-2__dup1].

**Citations detected**: `medquad_ans_0000719-2__dup1`
**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

## Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Top docs**:

1. `medquad_ans_0005818-1` (score=0.7000) - Spondylocarpotarsal synostosis (SCT) syndrome is an inherited syndrome characterized by disproportionate short stature, abnormalities of the vertebrae in the spine, scoliosis and lordosis, carpal and tarsal fusion (fusion of the bones in...
2. `medquad_ans_0006375-1` (score=0.5589) - Vohwinkel syndrome is an inherited condition that affects the skin. People with the "classic form" generally have honeycomb-patterned calluses on the palms of the hands and the soles of the feet (palmoplantar keratoses); constricting ban...
3. `medquad_ans_0001143-1` (score=0.4300) - CHILD syndrome, also known as congenital hemidysplasia with ichthyosiform erythroderma and limb defects, is a genetic condition that is typically characterized by large patches of skin that are red and inflamed (erythroderma) and covered...
4. `medquad_ans_0000840-1__dup1` (score=0.3507) - Branchiootorenal syndrome is characterized by birth defects or anomalies of tissues in the neck, malformations of the external ear, hearing loss, and kidney malformations. Symptom and symptom severity can vary greatly from person to pers...
5. `medquad_ans_0003384-1` (score=0.3477) - Keutel syndrome is an inherited condition characterized by cartilage calcification in the ears, nose, larnyx, trachea (voice box), and ribs; pulmonary artery stenoses; brachytelephalangism (short fingers and nails that resemble drumstick...

**Answer**:

The provided context does not mention "Konigsmark Knox Hussels syndrome" or any information related to this condition. Therefore, the cause cannot be determined from the given text.

Answer: Insufficient context [doc_id].

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Top docs**:

1. `medquad_ans_0004590-1` (score=0.7000) - What are the signs and symptoms of Orofaciodigital syndrome 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 3. If the information is available, the table below includes how o...
2. `medquad_ans_0004584-2` (score=0.6590) - What are the signs and symptoms of Orofaciodigital syndrome 1? Oral features of OFD1 may include a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormalities. Facial feature...
3. `medquad_ans_0004593-1` (score=0.6266) - What are the signs and symptoms of Orofaciodigital syndrome 8? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 8. If the information is available, the table below includes how o...
4. `medquad_ans_0004589-2` (score=0.4625) - What are the signs and symptoms of Orofaciodigital syndrome 2? Although the signs and symptoms that occur in people with orofaciodigital syndrome type 2 may vary, the following findings may be present:Facial findings Nodules (bumps) of t...
5. `medquad_ans_0004586-1` (score=0.4337) - What are the signs and symptoms of Orofaciodigital syndrome 11? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 11. If the information is available, the table below includes how...

**Answer**:

Oral features of Orofaciodigital syndrome 1 may include a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormalities. Facial features may include hypertelorism (increased width between the eyes), a small nose, micrognathia (small jaw), and other features. The fingers and toes may be short (brachydactyly), webbed or joined together (syndactyly), abnormally curved (clinodactyly), or have other abnormalities. There may be brain abnormalities (such as cysts) and kidney problems (such as polycystic kidney disease) [medquad_ans_0004584-2].

**Citations detected**: `medquad_ans_0004584-2`
**Supported citations**: `medquad_ans_0004584-2`
**Unsupported citations**: _none_

## Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Top docs**:

1. `medquad_ans_0000863-1__dup1` (score=0.7000) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For example, a surgeon may ...
2. `medquad_ans_0000718-1__dup2` (score=0.6453) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts or reshape sagging ...
3. `medquad_ans_0000969-1__dup2` (score=0.5359) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are different types of weigh...
4. `medquad_ans_0000010-23` (score=0.4588) - Several types of surgery are available for someone with colorectal cancer. If the cancer is found at a very early stage, the doctor may remove it without cutting through the abdominal wall. Instead, the doctor may put a tube up the rectu...
5. `medquad_ans_0000029-18` (score=0.4241) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.

**Answer**:

Yes, there is information about surgery. Surgery can relieve or prevent pain, reduce symptoms, improve body function, or find a problem such as through a biopsy. Some surgeries, like heart surgery, can save a life. Laparoscopic surgery allows for smaller incisions using a thin tube with a camera and small tools. After surgery, there are risks such as infection, bleeding, reaction to anesthesia, or accidental injury, and pain is almost always present [medquad_ans_0000863-1__dup1]. Cosmetic surgery can reshape body parts, including breasts, ears, eyes, face, hair, nose, and tummy [medquad_ans_0000718-1__dup2]. Weight loss surgery helps people with extreme obesity, especially when diet and exercise are ineffective or when obesity causes serious health problems; it often limits food intake and affects digestion and nutrient absorption, with risks including infections, hernias

**Citations detected**: `medquad_ans_0000863-1__dup1`, `medquad_ans_0000718-1__dup2`
**Supported citations**: `medquad_ans_0000863-1__dup1`, `medquad_ans_0000718-1__dup2`
**Unsupported citations**: _none_

## Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Top docs**:

1. `medquad_ans_0002935-1` (score=0.7000) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affected people may late...
2. `medquad_ans_0002935-4` (score=0.6093) - How is human T-cell leukemia virus, type 2 diagnosed? Human T-cell leukemia virus, type 2 (HTLV-2) is usually diagnosed based on blood tests that detect antibodies to the virus. However, HTLV-2 is often never suspected or diagnosed since...
3. `medquad_ans_0002935-5` (score=0.5321) - How might human T-cell leukemia virus, type 2 be treated? No cure or treatment exists for human T-cell leukemia virus, type 2 (HTLV-2). Management is focused on early detection and preventing the spread of HTLV-2 to others. Screening blo...
4. `medquad_ans_0002934-1` (score=0.4798) - Human T-cell leukemia virus, type 1 (HTLV-1) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, some affected people may later develop adult T-cell l...
5. `medquad_ans_0002934-4` (score=0.4529) - How is human T-cell leukemia virus, type 1 diagnosed? Human T-cell leukemia virus, type 1 (HTLV-1) is usually diagnosed based on blood tests that detect antibodies to the virus. However, HTLV-1 is often never suspected or diagnosed since...

**Answer**:

Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affects the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affected people may later develop neurological problems and/or chronic lung infections. HTLV-2 is spread by blood transfusions, sexual contact, and sharing needles. It can also be spread from mother to child during birth or breastfeeding. There is no cure or treatment for HTLV-2, and it is considered a lifelong condition; however, most infected people remain asymptomatic throughout life [medquad_ans_0002935-1].

**Citations detected**: `medquad_ans_0002935-1`
**Supported citations**: `medquad_ans_0002935-1`
**Unsupported citations**: _none_

## Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Top docs**:

1. `medquad_ans_0000522-1__dup1` (score=0.7000) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing in dust or other pa...
2. `medquad_ans_0000038-1__dup2` (score=0.5805) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chronic cough, rapid breat...
3. `medquad_ans_0000038-5__dup1` (score=0.3529) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interstitial lung disease o...
4. `medquad_ans_0000038-6` (score=0.1539) - Childhood interstitial lung disease (chILD) has many signs and symptoms because the disease has many forms. Signs and symptoms may include: Fast breathing, which also is called tachypnea (tak-ip-NE-ah) Labored breathing, which also is ca...
5. `medquad_ans_0000758-1__dup2` (score=0.1469) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough oxygen. Causes of pu...

**Answer**:

Interstitial lung diseases (ILD) are a large group of diseases that inflame or scar the lungs, making it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Causes include breathing in dust or other particles, such as coal dust (black lung disease), farm dust (farmer's lung), asbestos fibers (asbestosis), iron from mines or welding fumes (siderosis), and silica dust (silicosis). Other causes include autoimmune diseases or occupational exposures to molds, gases, or fumes. Some types have no known cause. Treatment depends on the type of exposure and disease stage and may involve medicines, oxygen therapy, or lung transplant in severe cases [medquad_ans_0000522-1__dup1]. In children, childhood interstitial lung disease (chILD) is a broad term for rare lung diseases affecting babies, children, and teens, with similar symptoms like chronic cough, rapid

**Citations detected**: `medquad_ans_0000522-1__dup1`
**Supported citations**: `medquad_ans_0000522-1__dup1`
**Unsupported citations**: _none_

## Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Top docs**:

1. `medquad_ans_0003743-4` (score=0.7000) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition which suggests that i...
2. `medquad_ans_0002735-4` (score=0.5642) - Is Hashimoto's encephalitis inherited? We are aware of only one instance when more than one person in the same family had Hashimoto's encephalitis (HE). To our knowledge, no other cases of familial HE have been reported; HE typically occ...
3. `medquad_ans_0005695-4` (score=0.4952) - Is spastic diplegia cerebral palsy inherited? Scientists have found that family members of people with cerebral palsy, including spastic diplegia cerebral palsy, have an increased risk of developing the condition. The exact risk depends ...
4. `medquad_ans_0005004-4` (score=0.4801) - Is postural orthostatic tachycardia syndrome inherited? Most cases of postural orthostatic tachycardia syndrome (POTS) are not thought to be inherited. Although the condition generally occurs sporadically, some people with POTS do report...
5. `medquad_ans_0000261-4` (score=0.4653) - In most cases of cyclic vomiting syndrome, affected people have no known history of the disorder in their family. However, many affected individuals have a family history of related conditions, such as migraines, irritable bowel syndrome...

**Answer**:

The provided context does not contain information about whether Melnick-Needles syndrome is inherited. Therefore, based on the given sources, it is not possible to determine if Melnick-Needles syndrome is inherited.

Answer: Insufficient context [no_source].

**Citations detected**: `no_source`
**Supported citations**: _none_
**Unsupported citations**: `no_source`

## Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Top docs**:

1. `medquad_ans_0006140-1` (score=0.7000) - Tracheobronchopathia osteoplastica (TO) is a rare condition of the large airways. It is characterized by the presence of multiple growths (nodules) made of bone and cartilage tissue, in the submucosa of the tracheobronchial wall. The nod...
2. `medquad_ans_0006138-5` (score=0.6315) - How is tracheobronchomalacia diagnosed? A diagnosis of tracheobronchomalacia (TBM) may be suspected based on the presence of characteristic signs and symptoms or abnormal pulmonary function tests. Additional testing such as CT scan and b...
3. `medquad_ans_0006138-1` (score=0.5362) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of TBM: a congenital fo...
4. `medquad_ans_0006140-3` (score=0.4838) - What causes tracheobronchopathia osteoplastica? The underlying cause of tracheobronchopathia osteoplastica (TO) remains unknown. Several theories have been proposed, including chronic airway inflammation, exostosis (formation of new bone...
5. `medquad_ans_0006138-3` (score=0.4430) - What causes tracheobronchomalacia? The underlying cause of tracheobronchomalacia (TBM) varies by subtype. Most cases of primary TBM (also called congenital TBM) are caused by genetic conditions that weaken the walls of the airway (specif...

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse [medquad_ans_0006138-1]. There are two forms: a congenital form (primary TBM), which typically develops during infancy or early childhood, and an acquired form (secondary TBM), which is usually seen in adults and often results from trauma, chronic inflammation, or prolonged compression of the airways [medquad_ans_0006138-1]. The condition is typically progressive and may lead to symptoms such as shortness of breath, cough, sputum retention, and wheezing or stridor with breathing [medquad_ans_0006138-1]. A diagnosis may be suspected based on symptoms or abnormal pulmonary function tests

**Citations detected**: `medquad_ans_0006138-1`
**Supported citations**: `medquad_ans_0006138-1`
**Unsupported citations**: _none_

## Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Top docs**:

1. `medquad_ans_0006030-1` (score=0.7000) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, ...
2. `medquad_ans_0000880-1__dup2` (score=0.1760) - The temporomandibular joint (TMJ) connects your jaw to the side of your head. When it works well, it enables you to talk, chew, and yawn. For people with TMJ dysfunction, problems with the joint and muscles around it may cause - Pain tha...
3. `medquad_ans_0004514-1` (score=0.0733) - Oculopharyngeal muscular dystrophy (OPMD) is a genetic muscle disorder with onset during adulthood, most often between 40 and 60 years of age. This condition is characterized by slowly progressive muscle disease (myopathy) affecting the ...
4. `medquad_ans_0000389-1` (score=0.0704) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which also includes otopalat...
5. `medquad_ans_0000810-1` (score=0.0654) - Popliteal pterygium syndrome is a condition that affects the development of the face, skin, and genitals. Most people with this disorder are born with a cleft lip, a cleft palate (an opening in the roof of the mouth), or both. Affected i...

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene, and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Citations detected**: `medquad_ans_0006030-1`
**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

## Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Top docs**:

1. `medquad_ans_0003845-4` (score=0.7000) - Is McCune Albright syndrome inherited? McCune Albright syndrome (MAS) is not inherited. It is caused by a random change (mutation) in the GNAS gene that occurs very early in development. As a result, some of the body's cells have a norma...
2. `medquad_ans_0000631-4` (score=0.2852) - McCune-Albright syndrome is not inherited. Instead, it is caused by a random mutation in the GNAS gene that occurs very early in development. As a result, some of the body's cells have a normal version of the GNAS gene, while other cells...
3. `medquad_ans_0004143-4` (score=0.2783) - How is Mowat-Wilson inherited? Mowat-Wilson syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of Mowat-Wilson syndrome result fro...
4. `medquad_ans_0000615-4` (score=0.0795) - Maffucci syndrome is not inherited. The mutations that cause this disorder are somatic, which means they occur during a person's lifetime. A somatic mutation occurs in a single cell. As that cell continues to grow and divide, the cells d...
5. `medquad_ans_0000181-5__dup1` (score=0.0318) - Alagille syndrome occurs in about one of every 30,000 live births.3 The disorder affects both sexes equally and shows no geographical, racial, or ethnic preferences. JAG1 and NOTCH2 gene mutations are inherited in an autosomal dominant w...

**Answer**:

The provided context does not contain information about the inheritance of Kallmann syndrome. Therefore, it is not possible to determine based on the given text whether Kallmann syndrome is inherited.

Answer: Insufficient context [medquad_ans_0003845-4, medquad_ans_0000631-4, medquad_ans_0004143-4, medquad_ans_0000615-4, medquad_ans_0000181-5__dup1].

**Citations detected**: `medquad_ans_0003845-4, medquad_ans_0000631-4, medquad_ans_0004143-4, medquad_ans_0000615-4, medquad_ans_0000181-5__dup1`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0003845-4, medquad_ans_0000631-4, medquad_ans_0004143-4, medquad_ans_0000615-4, medquad_ans_0000181-5__dup1`

## Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Top docs**:

1. `medquad_ans_0005967-2` (score=0.7000) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how often the symptom i...
2. `medquad_ans_0005970-1` (score=0.3564) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how often the symptom i...
3. `medquad_ans_0000908-1__dup1` (score=0.3245) - What are the signs and symptoms of Camptodactyly syndrome Guadalajara type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Camptodactyly syndrome Guadalajara type 3. If the information is available, ...
4. `medquad_ans_0006385-1` (score=0.3050) - What are the signs and symptoms of Waardenburg syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Waardenburg syndrome type 3. If the information is available, the table below includes how...
5. `medquad_ans_0005022-1` (score=0.2341) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the table below includes how...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and autosomal dominant inheritance [medquad_ans_0005967-2].

**Citations detected**: `medquad_ans_0005967-2`
**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

## Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Top docs**:

1. `medquad_ans_0000197-1__dup3` (score=0.7000) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tightness. There are tw...
2. `medquad_ans_0000853-1__dup1` (score=0.5112) - Bronchiolitis obliterans is an inflammatory obstruction of the lung's tiniest airways, the bronchioles. The bronchioles may become damaged and inflamed after inhalation of toxic fumes, as a result of respiratory infections, in associatio...
3. `medquad_ans_0000133-1__dup2` (score=0.4874) - The bronchi are two tubes that branch off the trachea, or windpipe. The bronchi carry air to your lungs. The most common problem with the bronchi is bronchitis, an inflammation of the tubes. Bronchitis can be acute or chronic. Other prob...
4. `medquad_ans_0000021-1__dup3` (score=0.4450) - Espaol Bronchitis (bron-KI-tis) is a condition in which the bronchial tubes become inflamed. These tubes carry air to your lungs. (For more information about the bronchial tubes and airways, go to the Diseases and Conditions Index How th...
5. `medquad_ans_0000008-1__dup3` (score=0.3109) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tightness. There are tw...

**Answer**:

Chronic bronchitis is an inflammation of the bronchial tubes, the airways that carry air to the lungs. It causes a cough that often brings up mucus and can also lead to shortness of breath, wheezing, a low fever, and chest tightness. It is one type of COPD (chronic obstructive pulmonary disease) and is characterized by the inflamed bronchial tubes producing a lot of mucus, leading to persistent coughing and difficulty breathing. Cigarette smoking is the most common cause, and long-term exposure to air pollution, fumes, or dust may also contribute. Chronic bronchitis is a long-term condition that tends to recur or never goes away completely. Diagnosis involves evaluating symptoms, listening to breathing, and possibly other tests. Treatment includes medicines to open airways and clear mucus, oxygen therapy if needed, and pulmonary rehabilitation to improve daily management [medquad_ans_000

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Top docs**:

1. `medquad_ans_0000840-1__dup2` (score=0.7000) - Spinal muscular atrophy (SMA) is a genetic disease that attacks nerve cells, called motor neurons, in the spinal cord. These cells communicate with your voluntary muscles - the ones you can control, like in your arms and legs. As the neu...
2. `medquad_ans_0000934-5` (score=0.3949) - These resources address the diagnosis or management of spinal and bulbar muscular atrophy: - Gene Review: Gene Review: Spinal and Bulbar Muscular Atrophy - Genetic Testing Registry: Bulbo-spinal atrophy X-linked - MedlinePlus Encyclopedi...
3. `medquad_ans_0000935-2` (score=0.3583) - Spinal muscular atrophy affects 1 in 6,000 to 1 in 10,000 people.
4. `medquad_ans_0006267-3` (score=0.2909) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation often becomes necessa...
5. `medquad_ans_0000233-2__dup1` (score=0.2428) - Generally, treatment for the disorder begins with stretching exercises and massage. Anti-inflammatory drugs may be prescribed. Cessation of running, bicycling, or similar activities may be advised. A corticosteroid injection near where t...

**Answer**:

Treatments for Spinal Muscular Atrophy (SMA) include machines to help with breathing, nutritional support, physical therapy, and medicines. There is no cure, but these treatments help with symptoms and prevent complications [medquad_ans_0000840-1__dup2].

**Citations detected**: `medquad_ans_0000840-1__dup2`
**Supported citations**: `medquad_ans_0000840-1__dup2`
**Unsupported citations**: _none_

## Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Top docs**:

1. `medquad_ans_0000040-19` (score=0.7000) - Standard treatments for leukemia include chemotherapy, biological therapy, radiation therapy, and immunotherapy. Some patients receive a combination of treatments. Learn more about treatments for acute myeloid leukemia. Learn more about ...
2. `medquad_ans_0000001_6-7` (score=0.5789) - Key Points - There are different types of treatment for childhood acute lymphoblastic leukemia (ALL). - Children with ALL should have their treatment planned by a team of doctors who are experts in treating childhood leukemia. - Children...
3. `medquad_ans_0000001_1-7` (score=0.4377) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - Chemotherapy with ste...
4. `medquad_ans_0000040-4__dup2` (score=0.4077) - There are many treatment options for people with leukemia. The choice of treatment depends on your age and general health, the type of leukemia you have, whether or not it has spread outside the bone marrow, and other factors. If tests s...
5. `medquad_ans_0000001_1-1` (score=0.3696) - Key Points - Adult acute lymphoblastic leukemia (ALL) is a type of cancer in which the bone marrow makes too many lymphocytes (a type of white blood cell). - Leukemia may affect red blood cells, white blood cells, and platelets. - Previo...

**Answer**:

The provided context does not contain information about treatments for Adult Acute Lymphoblastic Leukemia. While it details treatments for childhood acute lymphoblastic leukemia (ALL), including chemotherapy, radiation therapy, chemotherapy with stem cell transplant, targeted therapy, and clinical trials, there is no mention of adult-specific treatments for acute lymphoblastic leukemia [medquad_ans_0000001_6-7]. Therefore, based on the given context, the treatments for Adult Acute Lymphoblastic Leukemia cannot be determined.

**Citations detected**: `medquad_ans_0000001_6-7`
**Supported citations**: `medquad_ans_0000001_6-7`
**Unsupported citations**: _none_

## Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Top docs**:

1. `medquad_ans_0000388-2` (score=0.7000) - Friedreich ataxia is estimated to affect 1 in 40,000 people. This condition is found in people with European, Middle Eastern, or North African ancestry. It is rarely identified in other ethnic groups.
2. `medquad_ans_0000382-1__dup2` (score=0.6881) - Friedreich's ataxia is an inherited disease that damages your nervous system. The damage affects your spinal cord and the nerves that control muscle movement in your arms and legs. Symptoms usually begin between the ages of 5 and 15. The...
3. `medquad_ans_0000388-1` (score=0.6682) - Friedreich ataxia is a genetic condition that affects the nervous system and causes movement problems. People with this condition develop impaired muscle coordination (ataxia) that worsens over time. Other features of this condition incl...
4. `medquad_ans_0000133-1__dup1` (score=0.6363) - Friedreich's ataxia is a rare inherited disease that causes progressive damage to the nervous system and movement problems. Neurological symptoms include awkward, unsteady movements, impaired sensory function, speech problems, and vision...
5. `medquad_ans_0000388-5` (score=0.4849) - These resources address the diagnosis or management of Friedreich ataxia: - Friedreich's Ataxia Research Alliance: Clinical Care Guidelines - Gene Review: Gene Review: Friedreich Ataxia - Genetic Testing Registry: Friedreich ataxia 1 - M...

**Answer**:

The provided context does not specify the cause of Friedreich ataxia. While it describes the symptoms, inheritance, and progression of the condition, the underlying genetic or biological cause is not mentioned in the given texts. [medquad_ans_0000388-2], [medquad_ans_0000382-1__dup2], [medquad_ans_0000388-1], and [medquad_ans_0000133-1__dup1] all describe the condition as inherited and affecting the nervous system, but they do not identify the specific cause. Therefore, based on the available information, the cause of Friedreich ataxia cannot be determined.

**Citations detected**: `medquad_ans_0000388-2`, `medquad_ans_0000382-1__dup2`, `medquad_ans_0000388-1`, `medquad_ans_0000133-1__dup1`
**Supported citations**: `medquad_ans_0000388-2`, `medquad_ans_0000382-1__dup2`, `medquad_ans_0000388-1`, `medquad_ans_0000133-1__dup1`
**Unsupported citations**: _none_

## Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Top docs**:

1. `medquad_ans_0000013-4__dup2` (score=0.7000) - Atrial fibrillation (AF) occurs if the heart's electrical signals don't travel through the heart in a normal way. Instead, they become very rapid and disorganized. Damage to the heart's electrical system causes AF. The damage most often ...
2. `medquad_ans_0000013-7` (score=0.6422) - Atrial fibrillation (AF) is diagnosed based on your medical and family histories, a physical exam, and the results from tests and procedures. Sometimes AF doesn't cause signs or symptoms. Thus, it may be found during a physical exam or E...
3. `medquad_ans_0000070-1__dup4` (score=0.6159) - An arrhythmia is a problem with the speed or rhythm of the heartbeat. Atrial fibrillation (AF) is the most common type of arrhythmia. The cause is a disorder in the heart's electrical system. Often, people who have AF may not even feel s...
4. `medquad_ans_0000013-8` (score=0.5479) - Treatment for atrial fibrillation (AF) depends on how often you have symptoms, how severe they are, and whether you already have heart disease. General treatment options include medicines, medical procedures, and lifestyle changes. Goals...
5. `medquad_ans_0000013-6` (score=0.4881) - Atrial fibrillation (AF) usually causes the heart's lower chambers, the ventricles, to contract faster than normal. When this happens, the ventricles can't completely fill with blood. Thus, they may not be able to pump enough blood to th...

**Answer**:

The provided context does not define or describe familial atrial fibrillation. While it mentions family medical history as part of the diagnostic process for atrial fibrillation [medquad_ans_0000013-7], it does not specify whether familial atrial fibrillation exists as a condition or provide details about its causes, inheritance, or prevalence. Therefore, based on the given information, familial atrial fibrillation cannot be answered.

Answer: Insufficient context to determine what familial atrial fibrillation is.

**Citations detected**: `medquad_ans_0000013-7`
**Supported citations**: `medquad_ans_0000013-7`
**Unsupported citations**: _none_

## Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.7000) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary ...
2. `medquad_ans_0004627-2` (score=0.3100) - What are the signs and symptoms of Osteopathia striata cranial sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopathia striata cranial sclerosis. If the information is available, the tabl...
3. `medquad_ans_0004625-1` (score=0.1547) - Osteomesopyknosis is a bone disorder characterized by abnormal hardening of bone (osteosclerosis). It is generally limited to the axial spine, pelvis, and proximal part of the long bones, which is what distinguishes this condition from o...
4. `medquad_ans_0000260-1__dup1` (score=0.1461) - Striatonigral degeneration is a neurological disorder caused by a disruption in the connection between two areas of the brain-the striatum and the substantia nigra. These two areas work together to enable balance and movement. Striatonig...
5. `medquad_ans_0000238-1__dup1` (score=0.1054) - Primary lateral sclerosis (PLS) is a rare neuromuscular disease with slowly progressive weakness in voluntary muscle movement. PLS belongs to a group of disorders known as motor neuron diseases. PLS affects the upper motor neurons (also ...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary significantly from person to person, even within a single family. Features of the condition are generally present at birth and may include skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected people may also have developmental delay, hearing loss, heart defects, and/or ophthalmoplegia (paralysis of the muscles surrounding the eyes). Osteopathia striata cranial sclerosis is caused by changes (mutations) in the WTX gene and is inherited in an X-linked dominant manner. Treatment is based on the signs and symptoms present in each person [medquad_ans_0004

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Top docs**:

1. `medquad_ans_0004097-5` (score=0.7000) - How is mixed connective tissue disease diagnosed? Mixed connective tissue disease (MCTD) is often suspected after a physical examination reveals signs and symptoms associated with the condition. The diagnosis is supported by a blood test...
2. `medquad_ans_0004097-2` (score=0.6842) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyosi...
3. `medquad_ans_0004097-1` (score=0.4490) - Mixed connective tissue disease (MCTD) is a rare autoimmune disorder that is characterized by features commonly seen in three different connective tissue disorders: systemic lupus erythematosus, scleroderma, and polymyositis. Some affect...
4. `medquad_ans_0000230-1__dup2` (score=0.4134) - Connective tissue is the material inside your body that supports many of its parts. It is the "cellular glue" that gives your tissues their shape and helps keep them strong. It also helps some of your tissues do their work. Cartilage and...
5. `medquad_ans_0004172-2` (score=0.1783) - What are the signs and symptoms of multicentric Castleman disease? The signs and symptoms of multicentric Castleman disease (MCD) are often nonspecific and blamed on other, more common conditions. They can vary but may include: Fever Enl...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. A condition called Raynaud's phenomenon sometimes occurs months or years before other symptoms of MCTD develop. Most people with MCTD have pain in multiple joints, and/or inflammation of joints (arthritis). Muscle weakness, fevers, and fatigue are also common. Other signs and symptoms may include: accumulation of fluid in the tissue of the hands that causes puffiness and swelling (edema), skin findings including lupus-like rashes (including reddish brown patches), reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia), dysfunction of the esophagus

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Top docs**:

1. `medquad_ans_0000004-5__dup2` (score=0.7000) - Medications Can Treat Symptoms There is no known cure for Alzheimer's disease, but there are medicines that can treat symptoms of the disease. Most Alzheimers medicines work best for people in the mild or moderate stages of the disease. ...
2. `medquad_ans_0000024-1__dup2` (score=0.6315) - Caring for someone who has Alzheimer's disease (AD) can be stressful and overwhelming. It's important to take care of yourself. Ask for and accept help. Talk to the doctor. Find out what treatments might help control symptoms or address ...
3. `medquad_ans_0000003-6__dup1` (score=0.6280) - Most people with Alzheimers disease are cared for at home by family members. Within families, caregiving is provided most often by wives and husbands, followed by daughters. As Alzheimers disease gets worse, the person will need more and...
4. `medquad_ans_0000098-2__dup1` (score=0.5977) - Drugs to specifically treat Alzheimers disease and some other progressive dementias are now available. Although these drugs do not halt the disease or reverse existing brain damage, they can improve symptoms and slow the progression of t...
5. `medquad_ans_0000003-30` (score=0.2446) - As Alzheimers disease gets worse, you will need more help to care for the person. It's okay to seek help whenever you need it. Several kinds of help are available. - Home health care agencies send a home health aide or nurse to your home...

**Answer**:

The provided context does not contain specific information about treatments for Alzheimer's caregiving. It discusses medications for treating symptoms, memory aids, and managing behavioral symptoms, but does not address treatments or strategies specifically for caregivers. Therefore, based on the given context, it is not possible to provide a supported answer regarding treatments for Alzheimer's caregiving. [medquad_ans_0000004-5__dup2]

**Citations detected**: `medquad_ans_0000004-5__dup2`
**Supported citations**: `medquad_ans_0000004-5__dup2`
**Unsupported citations**: _none_

## Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Top docs**:

1. `medquad_ans_0000293-4` (score=0.7000) - Distal myopathy 2 is inherited in an autosomal dominant pattern, which means one copy of the altered MATR3 gene in each cell is sufficient to cause the disorder.
2. `medquad_ans_0003179-1` (score=0.5366) - Inclusion body myopathy 2, also known as hereditary inclusion body myopathy (HIBM), GNE-related myopathy, distal myopathy with rimmed vacuoles, and Nonaka myopathy, is an inherited condition that primarily affects the skeletal muscles (t...
3. `medquad_ans_0000707-1__dup1` (score=0.4017) - Bethlem myopathy is an inherited movement disorder characterized by progressive muscle weakness and joint stiffness (contractures) in the fingers, wrists, elbows, and ankles. Due to a progressive course, up to two-thirds of people with t...
4. `medquad_ans_0006236-4` (score=0.3888) - Is tubular aggregate myopathy genetic? It is evident from family history studies that the condition can be passed through families in either an autosomal dominant or autosomal recessive fashion. Some cases appear to be due to dominant mu...
5. `medquad_ans_0006500-1` (score=0.3390) - X-linked scapuloperoneal myopathy is an inherited muscular dystrophy characterized by weakness and wasting of the muscles in the lower legs and the area of the shoulder blades. In some individuals, facial muscles may also be affected. Wh...

**Answer**:

The provided context does not contain any information about the genetic changes related to idiopathic inflammatory myopathy. Therefore, a definitive answer cannot be provided based on the given text. [no context]

**Citations detected**: `no context`
**Supported citations**: _none_
**Unsupported citations**: `no context`

## Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Top docs**:

1. `medquad_ans_0000855-2` (score=0.7000) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
2. `medquad_ans_0000855-5` (score=0.3307) - These resources address the diagnosis or management of pyruvate dehydrogenase deficiency: - Genetic Testing Registry: Pyruvate dehydrogenase E1-beta deficiency - Genetic Testing Registry: Pyruvate dehydrogenase E2 deficiency - Genetic Te...
3. `medquad_ans_0000855-1` (score=0.2924) - Pyruvate dehydrogenase deficiency is characterized by the buildup of a chemical called lactic acid in the body and a variety of neurological problems. Signs and symptoms of this condition usually first appear shortly after birth, and the...
4. `medquad_ans_0005223-2` (score=0.2864) - What are the signs and symptoms of Pyruvate dehydrogenase deficiency? Pyruvate dehydrogenase (PDH) deficiency can have a significant effect on fetal development, which may become apparent during late pregnancy with poor fetal weight gain...
5. `medquad_ans_0005223-5` (score=0.2854) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for pyruvate dehydrogen...

**Answer**:

Diagnosis of pyruvate dehydrogenase deficiency can be approached through genetic testing, which identifies specific deficiencies in components of the pyruvate dehydrogenase complex, such as E1-beta, E2, E3-binding protein, or phosphatase deficiencies [medquad_ans_0000855-5]. Additionally, diagnostic resources such as those provided by MedlinePlus, including Diagnostic Tests, Genetic Counseling, and Drug Therapy, may support the evaluation and management of the condition [medquad_ans_0000855-5]. Clinical signs and symptoms, including lactic acidosis, neurological problems, delayed development, and abnormal brain structures, also contribute to the diagnostic process [medquad_ans_0000855-1]. However, specific diagnostic procedures or tests are not detailed in the provided context.

**Citations detected**: `medquad_ans_0000855-5`, `medquad_ans_0000855-1`
**Supported citations**: `medquad_ans_0000855-5`, `medquad_ans_0000855-1`
**Unsupported citations**: _none_

## Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Top docs**:

1. `medquad_ans_0006562-2` (score=0.7000) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb abnormalities. All in...
2. `medquad_ans_0006559-2` (score=0.4003) - What are the signs and symptoms of 16q24.3 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 16q24.3 microdeletion syndrome. If the information is available, the table below includ...
3. `medquad_ans_0006554-2` (score=0.4000) - What are the signs and symptoms of 15q11.2 microdeletion? The signs and symptoms in people with a 15q11.2 microdeletion can vary widely. Some people with the microdeletion don't have any apparent features, while others are more severely ...
4. `medquad_ans_0006565-2` (score=0.2617) - What are the signs and symptoms of 1q21.1 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 1q21.1 microdeletion syndrome. If the information is available, the table below includes...
5. `medquad_ans_0006566-1` (score=0.1926) - What are the signs and symptoms of 1q44 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 1q44 microdeletion syndrome. If the information is available, the table below includes how...

**Answer**:

The symptoms of 17q23.1q23.2 microdeletion syndrome include developmental delay, microcephaly, short stature, heart defects, and hand, foot, and limb abnormalities. Specific signs and symptoms reported in the literature include:

- Developmental delay, particularly delays in speech [medquad_ans_0006562-2]
- Microcephaly (seen in 50% of patients) [medquad_ans_0006562-2]
- Short stature (seen in 50% of patients) [medquad_ans_0006562-2]
- Heart defects, including patent ductus arteriosus or atrial septal defects (each seen in 50% of patients) [medquad_ans_0006562-2]
- Arachnodactyly (long, thin fingers and toes)

**Citations detected**: `medquad_ans_0006562-2`
**Supported citations**: `medquad_ans_0006562-2`
**Unsupported citations**: _none_

## Example 186 (`0003860-3`)

**Question**: What causes MECP2 duplication syndrome ?

**Top docs**:

1. `medquad_ans_0003860-1` (score=0.7000) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delayed motor milestones ...
2. `medquad_ans_0003860-3` (score=0.6724) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located on the long (q) arm...
3. `medquad_ans_0000635-3` (score=0.6399) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q) arm of the X chromo...
4. `medquad_ans_0000635-1` (score=0.6003) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infancy, feeding difficul...
5. `medquad_ans_0000635-2` (score=0.5708) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of all cases of intell...

**Answer**:

MECP2 duplication syndrome is caused by an extra copy (duplication) of the MECP2 gene in each cell, which is generally due to a duplication of genetic material located on the long (q) arm of the X chromosome [medquad_ans_0003860-1]. The duplication can vary in size, ranging from 100,000 to 900,000 base pairs (100 to 900 kilobases), and the MECP2 gene is always included in this duplication [medquad_ans_0000635-3]. This results in the production of excess MeCP2 protein, which disrupts the normal regulation of other genes in the brain, leading to irregular brain activity and the associated signs and symptoms [medquad_ans_0003860-3].

**Citations detected**: `medquad_ans_0003860-1`, `medquad_ans_0000635-3`, `medquad_ans_0003860-3`
**Supported citations**: `medquad_ans_0003860-1`, `medquad_ans_0000635-3`, `medquad_ans_0003860-3`
**Unsupported citations**: _none_

## Example 187 (`0004633-2`)

**Question**: What are the symptoms of Osteopetrosis autosomal dominant type 2 ?

**Top docs**:

1. `medquad_ans_0004633-2` (score=0.7000) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 2. If the information is available, the ...
2. `medquad_ans_0004637-2` (score=0.4442) - What are the signs and symptoms of Osteopetrosis autosomal recessive 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 4. If the information is available, the table be...
3. `medquad_ans_0004632-2` (score=0.4289) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 1. If the information is available, the ...
4. `medquad_ans_0004640-2` (score=0.3662) - What are the signs and symptoms of Osteopetrosis autosomal recessive 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 7. If the information is available, the table be...
5. `medquad_ans_0000759-1` (score=0.3295) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of inheritance: autosom...

**Answer**:

The symptoms of Osteopetrosis autosomal dominant type 2 include abnormality of epiphysis morphology, abnormality of the metacarpal bones, abnormality of the metaphyses, aseptic necrosis, bone pain, facial palsy, frontal bossing, joint dislocation, macrocephaly, osteoarthritis, osteomyelitis, recurrent fractures, short distal phalanx of finger, anemia, genu valgum, optic atrophy, short stature, visual impairment, abnormality of leukocytes, carious teeth, hearing impairment, hydrocephalus, hypocalcemia, bone marrow hypocellularity, abnormality of pelvic girdle bone morphology, abnormality of the vertebral endplates, generalized osteosclerosis, hip osteoarthritis, juvenile onset, mandibular osteomyelitis, and fractures of the long bones [medquad_ans_0004633

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 188 (`0003459-4`)

**Question**: What are the treatments for Lambert Eaton myasthenic syndrome ?

**Top docs**:

1. `medquad_ans_0003459-4` (score=0.7000) - How might Lambert-Eaton myasthenic syndrome be treated? Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, plasmapheresis (wher...
2. `medquad_ans_0006133-6` (score=0.3190) - Is there treatment for Townes-Brocks syndrome? Treatment is directed towards the specific symptoms, including immediate surgical intervention for imperforate anus; surgery for severe malformations of the hands; routine management of cong...
3. `medquad_ans_0003459-1` (score=0.3020) - Lambert Eaton myasthenic syndrome (LEMS) is a disorder of the neuromuscular junction. The neuromuscular junction is the site where nerve cells meet muscle cells and help activate the muscles. This syndrome occurs when antibodies interfer...
4. `medquad_ans_0003459-2` (score=0.2235) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficulty lifting objects ...
5. `medquad_ans_0000200-2__dup2` (score=0.1785) - Currently there is no cure for these disease syndromes.Medical care is directed at treating systemic conditions and improving the person's quality of life. Physical therapy and daily exercise may delay joint problems and improve the abil...

**Answer**:

Treatments for Lambert-Eaton myasthenic syndrome (LEMS) may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, plasmapheresis, IV immunoglobulins, steroids (e.g., prednisone), azathioprine or cyclosporine, and/or 3,4-diaminopyridine. While some evidence suggests that 3,4-diaminopyridine or IV immunoglobulin can improve muscle strength and nerve to muscle cell communication, the degree of benefit still needs to be determined. Treatment of the underlying disorder or cancer is the first priority [medquad_ans_0003459-4, medquad_ans_0003459-1].

**Citations detected**: `medquad_ans_0003459-4, medquad_ans_0003459-1`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0003459-4, medquad_ans_0003459-1`

## Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Top docs**:

1. `medquad_ans_0000445-1__dup1` (score=0.7000) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of arginine in the blo...
2. `medquad_ans_0000445-2__dup1` (score=0.6156) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes how often the sympt...
3. `medquad_ans_0000065-2` (score=0.4782) - Arginase deficiency is a very rare disorder; it has been estimated to occur once in every 300,000 to 1,000,000 individuals.
4. `medquad_ans_0000065-1` (score=0.4525) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxi...
5. `medquad_ans_0003472-2` (score=0.2287) - What are the signs and symptoms of L-arginine:glycine amidinotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for L-arginine:glycine amidinotransferase deficiency. If the information ...

**Answer**:

The symptoms of Arginase deficiency include growth deficiency, spasticity (abnormal tensing of the muscles), developmental delay, loss of developmental milestones, intellectual disability, seizures, microcephaly, hyperammonemia, aminoaciduria, behavioral abnormality, cognitive impairment, neurological speech impairment, EEG abnormality, hemiplegia/hemiparesis, hyperactivity, irritability, oroticaciduria, postnatal growth retardation, progressive spastic quadriplegia, and vomiting [medquad_ans_0000445-1__dup1]. Additionally, approximately 90% of patients exhibit aminoaciduria, behavioral abnormality, cognitive impairment, and neurological speech impairment, while 50% show EEG abnormality, hyperammonemia, seizures, hemiplegia/hemiparesis, and other listed symptoms [medquad_ans_0000445-2\_\_dup

**Citations detected**: `medquad_ans_0000445-1__dup1`
**Supported citations**: `medquad_ans_0000445-1__dup1`
**Unsupported citations**: _none_

## Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Top docs**:

1. `medquad_ans_0006376-6` (score=0.7000) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but are still small so the...
2. `medquad_ans_0000961-1__dup1` (score=0.5765) - Von Hippel-Lindau disease (VHL) is a rare, genetic disease that causes tumors and cysts to grow in your body. The tumors can be either cancerous or benign. They can grow in your brain and spinal cord, kidneys, pancreas and, in men, their...
3. `medquad_ans_0006376-1` (score=0.4806) - Von Hippel-Lindau (VHL) disease is an inherited disorder characterized by the abnormal growth of both benign and cancerous tumors and cysts in many parts of the body. Tumors usually first appear in young adulthood. The types of tumors as...
4. `medquad_ans_0006376-5` (score=0.4601) - How is von Hippel-Lindau (VHL) disease diagnosed? The diagnosis of von Hippel-Lindau (VHL) disease can be made based on specific clinical criteria (signs and symptoms), or when molecular genetic testing reveals a mutation in the VHL gene...
5. `medquad_ans_0000022-1__dup1` (score=0.4391) - von Hippel-Lindau disease (VHL) is a rare, genetic multi-system disorder in which non-cancerous tumors grow in certain parts of the body. Slow-growing hemgioblastomas -- benign tumors with many blood vessels -- may develop in the brain, ...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. The goal is to treat growths when they cause symptoms but are still small to prevent permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All people with VHL disease should be carefully followed by a physician or medical team familiar with the disorder [medquad_ans_0006376-6]. Surgery or radiation therapy are commonly used, with the aim of addressing tumors early [medquad_ans_0000961-1__dup1].

**Citations detected**: `medquad_ans_0006376-6`, `medquad_ans_0000961-1__dup1`
**Supported citations**: `medquad_ans_0006376-6`, `medquad_ans_0000961-1__dup1`
**Unsupported citations**: _none_

## Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Top docs**:

1. `medquad_ans_0000738-2` (score=0.7000) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
2. `medquad_ans_0000738-1` (score=0.5479) - Noonan syndrome is a condition that affects many areas of the body. It is characterized by mildly unusual facial characteristics, short stature, heart defects, bleeding problems, skeletal malformations, and many other signs and symptoms....
3. `medquad_ans_0004452-2` (score=0.3618) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) are generally standa...
4. `medquad_ans_0004456-3` (score=0.3618) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) are generally standa...
5. `medquad_ans_0004453-3` (score=0.3618) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) are generally standa...

**Answer**:

The provided context does not mention "Noonan syndrome 3" or any such classification. Therefore, based on the available information, it is not possible to answer the question. [doc_id]

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Top docs**:

1. `medquad_ans_0005865-2` (score=0.7000) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may not have noticeable...
2. `medquad_ans_0005865-1` (score=0.3254) - Sprengel deformity is a congenital condition characterized by abnormal development and elevation of the shoulder blade (scapula). Severity can range considerably from being almost invisible when covered with clothes, to the shoulder bein...
3. `medquad_ans_0005865-3` (score=0.2266) - How might Sprengel deformity be treated? Treatment of Sprengel deformity depends on the degree of functional impairment and/or cosmetic disfigurement. Many people with Sprengel deformity do not need surgery and may have physical therapy ...
4. `medquad_ans_0001315-2` (score=0.0902) - What are the signs and symptoms of Chronic atypical neutrophilic dermatosis with lipodystrophy and elevated temperature? The Human Phenotype Ontology provides the following list of signs and symptoms for Chronic atypical neutrophilic der...
5. `medquad_ans_0000003-2__dup3` (score=0.0876) - Signs and symptoms of Cushing's syndrome vary, but most people with the disorder have upper body obesity, a rounded face, increased fat around the neck, and relatively slender arms and legs. Children tend to be obese with slowed growth r...

**Answer**:

The symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additional features that may be present include abnormality of the shoulder (seen in approximately 90% of patients), cleft palate (in about 7.5% of patients), scoliosis, hemivertebrae, neck muscle hypoplasia, rib segmentation abnormalities, cervical segmentation defects, spina bifida occulta, and abnormalities of the collarbone or chest organs [medquad_ans_0005865-2]. The deformity more commonly occurs on the left side but can be bilateral [medquad_ans_0005865-2].

**Citations detected**: `medquad_ans_0005865-2`
**Supported citations**: `medquad_ans_0005865-2`
**Unsupported citations**: _none_

## Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Top docs**:

1. `medquad_ans_0000882-4` (score=0.7000) - As discussed above, Robinow syndrome can have either an autosomal recessive or an autosomal dominant pattern of inheritance. Autosomal recessive inheritance means both copies of the gene in each cell have mutations. The parents of an ind...
2. `medquad_ans_0000882-5` (score=0.4393) - These resources address the diagnosis or management of Robinow syndrome: - Gene Review: Gene Review: Autosomal Dominant Robinow Syndrome - Gene Review: Gene Review: ROR2-Related Robinow Syndrome - Genetic Testing Registry: Robinow syndro...
3. `medquad_ans_0005415-4` (score=0.4138) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of Robinow syndrome have...
4. `medquad_ans_0005415-1` (score=0.4016) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant Robinow syndrome. T...
5. `medquad_ans_0000882-2` (score=0.3689) - Both the autosomal recessive and autosomal dominant forms of Robinow syndrome are rare. Fewer than 200 people with autosomal recessive Robinow syndrome have been described in the medical literature. This form of the condition has been id...

**Answer**:

Genetic changes related to Robinow syndrome include mutations in the genes that cause either autosomal recessive or autosomal dominant inheritance patterns. In autosomal recessive Robinow syndrome, both copies of the gene in each cell have mutations, while in autosomal dominant Robinow syndrome, one copy of the altered gene in each cell is sufficient to cause the disorder. Specific gene-related changes are associated with conditions such as ROR2-related Robinow syndrome [medquad_ans_0000882-4]. However, not all individuals with Robinow syndrome have identifiable mutations in known genes, and in some cases, the cause remains unknown [medquad_ans_0005415-4]. Genetic testing is available for both forms of the syndrome, and the Genetic Testing Registry provides information on these tests [medquad_ans_0005415-4].

**Citations detected**: `medquad_ans_0000882-4`, `medquad_ans_0005415-4`
**Supported citations**: `medquad_ans_0000882-4`, `medquad_ans_0005415-4`
**Unsupported citations**: _none_

## Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Top docs**:

1. `medquad_ans_0000180-1__dup3` (score=0.7000) - Kidney failure means your kidneys no longer work well enough to do their job. You need treatment to replace the work your damaged kidneys have stopped doing. The treatments for kidney failure are - hemodialysis - peritoneal dialysis - a ...
2. `medquad_ans_0000179-12` (score=0.4824) - Eating healthy foods can help you keep up your energy and strength. All dialysis and transplant centers have a dietitian. The dietitian helps people with kidney failure learn about healthy food choices. You should talk with your centers ...
3. `medquad_ans_0000158-14` (score=0.4648) - - Kidney failure can affect a persons health in several ways. - When the kidneys stop working, waste products build up in the blood, a condition known as uremia. - People with kidney failure can avoid most of the problems of uremia by ha...
4. `medquad_ans_0000180-8` (score=0.4626) - Eating the right foods can help you feel better when you are on dialysis or have a kidney transplant. Staying healthy with kidney failure requires watching how much of these elements are included in your diet: - Protein is in many foods ...
5. `medquad_ans_0000158-13` (score=0.4415) - For people who are on dialysis or approaching total kidney failure, adequate nutrition is important for maintaining energy, strength, healthy sleep patterns, bone health, heart health, and good mental health. A persons treatment will dic...

**Answer**:

The provided context does not contain information about "Kidney Failure: Eat Right to Feel Right on Hemodialysis." While it mentions that a dietitian helps people with kidney failure learn about healthy food choices and that following a special diet can help keep wastes from building up in the blood, there is no specific content related to the topic "Eat Right to Feel Right on Hemodialysis." [medquad_ans_0000180-1__dup3]

**Citations detected**: `medquad_ans_0000180-1__dup3`
**Supported citations**: `medquad_ans_0000180-1__dup3`
**Unsupported citations**: _none_

## Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Top docs**:

1. `medquad_ans_0000677-2` (score=0.7000) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
2. `medquad_ans_0000675-2` (score=0.5006) - Mucolipidosis III alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
3. `medquad_ans_0000674-2` (score=0.4945) - Mucolipidosis II alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
4. `medquad_ans_0000676-2` (score=0.4270) - Mucolipidosis III gamma is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
5. `medquad_ans_0000677-1` (score=0.3519) - Mucolipidosis type IV is an inherited disorder characterized by delayed development and vision impairment that worsens over time. The severe form of the disorder is called typical mucolipidosis type IV, and the mild form is called atypic...

**Answer**:

Mucolipidosis type IV is estimated to occur in 1 in 40,000 people [medquad_ans_0000677-2].

**Citations detected**: `medquad_ans_0000677-2`
**Supported citations**: `medquad_ans_0000677-2`
**Unsupported citations**: _none_

## Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.7000) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin con...
2. `medquad_ans_0000193-1__dup4` (score=0.2483) - Cirrhosis\* is scarring of the liver. Scar tissue forms because of injury or long-term disease. Scar tissue replaces healthy liver tissue and blocks the normal flow of blood through the liver. A healthy liver - makes proteins - helps figh...
3. `medquad_ans_0005201-1` (score=0.1451) - Punctate palmoplantar keratoderma type I, also known as keratosis palmoplantaris papulosa (KPPP) or Brauer-Buschke-Fisher Syndrome is is a rare condition that affects the skin. It is a type of punctate palmoplantar keratoderma. Signs and...
4. `medquad_ans_0003715-1` (score=0.1287) - Lymphomatoid papulosis is a skin disorder that is characterized by crops of self healing skin lesions that look cancerous under the microscope but are actually benign (non-cancerous). Lesions contain unusual cells that are similar to tho...
5. `medquad_ans_0000063-12` (score=0.1250) - Skin cancer occurs when cancer cells form in the tissues of the skin. The skin is mainly made up of two layers: the inner layer, called the dermis, and the outer layer, called the epidermis. Within the epidermis, there are three types of...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams. [medquad_ans_0000795-1__dup1]

**Citations detected**: `medquad_ans_0000795-1__dup1`
**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

## Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Top docs**:

1. `medquad_ans_0001045-1` (score=0.7000) - X-linked adrenoleukodystrophy is a genetic disorder that occurs primarily in males. It mainly affects the nervous system and the adrenal glands, which are small glands located on top of each kidney. In this disorder, the fatty covering (...
2. `medquad_ans_0006488-4` (score=0.4850) - How is X-linked adrenal hypoplasia congenita inherited? X-linked adrenal hypoplasia congenita is inherited in an X-linked recessive pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on th...
3. `medquad_ans_0006488-1` (score=0.4622) - X-linked adrenal hypoplasia congenita is an inherited disorder that mainly affects males. It involves many hormone-producing (endocrine) tissues in the body, particularly a pair of small glands on top of each kidney called the adrenal gl...
4. `medquad_ans_0006502-4` (score=0.2626) - How is X-linked hypophosphatemia inherited? X-linked hypophosphatemia (XLH) is caused by mutations in the PHEX gene, and is inherited in an X-linked dominant manner. This means that the gene responsible for the condition is located on th...
5. `medquad_ans_0006553-1` (score=0.1909) - Congenital adrenal hyperplasia (CAH) due to 11-beta-hydroxylase deficiency is one of a group of disorders (collectively called congenital adrenal hyperplasia) that affect the adrenal glands. In this condition, the adrenal glands produce ...

**Answer**:

The provided context does not describe the specific genetic changes related to X-linked adrenoleukodystrophy. Therefore, based on the given information, the genetic changes cannot be determined. [medquad_ans_0001045-1]

**Citations detected**: `medquad_ans_0001045-1`
**Supported citations**: `medquad_ans_0001045-1`
**Unsupported citations**: _none_

## Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Top docs**:

1. `medquad_ans_0000988-2` (score=0.7000) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of European descent, where it...
2. `medquad_ans_0000988-1` (score=0.3163) - Transthyretin amyloidosis is a slowly progressive condition characterized by the buildup of abnormal deposits of a protein called amyloid (amyloidosis) in the body's organs and tissues. These protein deposits most frequently occur in the...
3. `medquad_ans_0000195-2` (score=0.1947) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0001101-2` (score=0.1723) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
5. `medquad_ans_0000303-1__dup1` (score=0.1579) - What are the signs and symptoms of Amyloidosis familial visceral? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyloidosis familial visceral. If the information is available, the table below includes...

**Answer**:

The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence is estimated to be one in 538 people. Among Americans of European descent, it affects one in 100,000 people. The cardiac form affects between 3 percent and 3.9 percent of African Americans and approximately 5 percent of people in some areas of West Africa [medquad_ans_0000988-2].

**Citations detected**: `medquad_ans_0000988-2`
**Supported citations**: `medquad_ans_0000988-2`
**Unsupported citations**: _none_

## Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.7000) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the information is available, the ...
2. `medquad_ans_0001787-1` (score=0.2220) - What are the signs and symptoms of Dentin dysplasia, type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Dentin dysplasia, type 1. If the information is available, the table below includes how often...
3. `medquad_ans_0005856-1` (score=0.1648) - What are the signs and symptoms of Spondylometaphyseal dysplasia with dentinogenesis imperfecta? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia with dentinogenesis imperfe...
4. `medquad_ans_0003176-1` (score=0.1510) - What are the signs and symptoms of Imperforate oropharynx-costo vetebral anomalies? The Human Phenotype Ontology provides the following list of signs and symptoms for Imperforate oropharynx-costo vetebral anomalies. If the information is...
5. `medquad_ans_0003065-1` (score=0.1483) - What are the signs and symptoms of Hypomandibular faciocranial dysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypomandibular faciocranial dysostosis. If the information is available, the ta...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include microdontia, taurodontia, dentinogenesis imperfecta limited to primary teeth, obliteration of the pulp chamber, and periapical radiolucency. These symptoms are reported in the Human Phenotype Ontology (HPO) database, which compiles information from sources such as Orphanet. The frequency of these symptoms is typically listed as a rough estimate of the percentage of patients who have the feature, or as a fraction indicating the number of patients with the symptom out of the total number examined in a study [medquad_ans_0001780-1]. However, specific symptom frequencies are not detailed in the provided context.

**Citations detected**: `medquad_ans_0001780-1`
**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

## Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Top docs**:

1. `medquad_ans_0000782-2` (score=0.7000) - Periventricular heterotopia is a rare condition. Its incidence is unknown.
2. `medquad_ans_0004818-2` (score=0.3001) - What are the signs and symptoms of periventricular nodular heterotopia? The condition is first noticed when seizures appear, often during the teenage years. The nodules around the ventricles are then typically discovered when magnetic re...
3. `medquad_ans_0000782-1` (score=0.1949) - Periventricular heterotopia is a condition in which nerve cells (neurons) do not migrate properly during the early development of the fetal brain, from about the 6th week to the 24th week of pregnancy. Heterotopia means "out of place." I...
4. `medquad_ans_0004818-1` (score=0.1900) - Periventricular heterotopia is a condition in which the nerve cells (neurons) do not migrate properly during the early development of the fetal brain from about the 6th week to the 24th week of pregnancy. Affected people typically develo...
5. `medquad_ans_0004818-3` (score=0.1844) - What are the recommended evaluations for patients diagnosed with periventricular nodular heterotopia? The following evaluations are recommended:[1823] Imaging exams of the brain to establish the diagnosis Evaluation by a neurologist Eval...

**Answer**:

The incidence of periventricular heterotopia is unknown [medquad_ans_0000782-2].

**Citations detected**: `medquad_ans_0000782-2`
**Supported citations**: `medquad_ans_0000782-2`
**Unsupported citations**: _none_

## Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Top docs**:

1. `medquad_ans_0003462-3` (score=0.7000) - How is Landau-Kleffner syndrome (LKS) diagnosed? LKS is diagnosed based on clinical features and the results of an electroencephalogram (EEG), a recording of the electric activity of the brain. All LKS children have abnormal electrical b...
2. `medquad_ans_0003462-1` (score=0.4934) - Landau-Kleffner syndrome (LKS) is a rare, childhood neurological disorder characterized by the sudden or gradual development of aphasia (the inability to understand or express language) and an abnormal electro-encephalogram (EEG). The di...
3. `medquad_ans_0003462-2` (score=0.4707) - What are the signs and symptoms of Landau-Kleffner syndrome? Landau-Kleffner syndrome is characterized by the sudden or gradual development of aphasia (the inability to understand or express language) in previously normal children along ...
4. `medquad_ans_0000004-1__dup2` (score=0.4087) - Landau-Kleffner syndrome (LKS) is a rare, childhood neurological disorder characterized by the sudden or gradual development of aphasia (the inability to understand or express language) and an abnormal electro-encephalogram (EEG). LKS af...
5. `medquad_ans_0006423-3` (score=0.1552) - How might Wernicke-Korsakoff syndrome be treated?

**Answer**:

The provided context does not mention any ongoing research or clinical trials for Landau-Kleffner Syndrome. Therefore, based on the given information, it is not possible to identify specific research or clinical trials. [doc_id]

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Top docs**:

1. `medquad_ans_0004514-4` (score=0.7000) - How is oculopharyngeal muscular dystrophy inherited?
2. `medquad_ans_0003601-1` (score=0.6347) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of years. Onset may occu...
3. `medquad_ans_0000631-1__dup1` (score=0.5995) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. The different types ...
4. `medquad_ans_0001537-1` (score=0.5333) - Congenital muscular dystrophy (CMD) refers to a group of inherited conditions that affect the muscles and are present at birth or in early infancy. The severity of the condition, the associated signs and symptoms and the disease progress...
5. `medquad_ans_0001935-1` (score=0.4670) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystrophin. Boys with Duchen...

**Answer**:

Muscular dystrophy (MD) is a group of more than 30 inherited diseases that cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood, while others may not appear until middle age or later. The different types can vary in whom they affect, which muscles they affect, and what the symptoms are. All forms of MD grow worse as the person's muscles get weaker, and most people with MD eventually lose the ability to walk. There is no cure for muscular dystrophy; treatments can help with symptoms and prevent complications, including physical and speech therapy, orthopedic devices, surgery, and medications [medquad_ans_0000631-1__dup1].

**Citations detected**: `medquad_ans_0000631-1__dup1`
**Supported citations**: `medquad_ans_0000631-1__dup1`
**Unsupported citations**: _none_

## Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.7000) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that the person has no oth...
2. `medquad_ans_0004956-1` (score=0.3562) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. Specifically, the surface of the brain develops too many folds which are unusually small. The signs and symptoms associated with the condition...
3. `medquad_ans_0005019-1` (score=0.2173) - What are the signs and symptoms of Preaxial polydactyly type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 1. If the information is available, the table below includes how...
4. `medquad_ans_0000754-5` (score=0.2140) - These resources address the diagnosis or management of oral-facial-digital syndrome: - Gene Review: Gene Review: Oral-Facial-Digital Syndrome Type I - Genetic Testing Registry: Mohr syndrome - Genetic Testing Registry: Oral-facial-digita...
5. `medquad_ans_0003881-2` (score=0.2031) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speech Drooling

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding with no other physical anomalies or intellectual impairment, or it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits, and surgical management depends greatly on the complexity of the deformity [1] [2].

**Citations detected**: `1`, `2`
**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

## Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Top docs**:

1. `medquad_ans_0002519-2` (score=0.7000) - What are the signs and symptoms of Geniospasm? The Human Phenotype Ontology provides the following list of signs and symptoms for Geniospasm. If the information is available, the table below includes how often the symptom is seen in peop...
2. `medquad_ans_0004534-2` (score=0.2905) - What are the signs and symptoms of Omenn syndrome? Infants with Omenn syndrome typically present shortly after birth, usually by 3 months of age. This is similar to other types of severe combined immunodeficiency (SCID). The characterist...
3. `medquad_ans_0002628-2` (score=0.2461) - What are the signs and symptoms of GM1 gangliosidosis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for GM1 gangliosidosis type 2. If the information is available, the table below includes how oft...
4. `medquad_ans_0003786-1` (score=0.1708) - What are the signs and symptoms of Mannosidosis, beta A, lysosomal? The Human Phenotype Ontology provides the following list of signs and symptoms for Mannosidosis, beta A, lysosomal. If the information is available, the table below incl...
5. `medquad_ans_0002114-2` (score=0.1062) - What are the signs and symptoms of Eosinophilic fasciitis? The Human Phenotype Ontology provides the following list of signs and symptoms for Eosinophilic fasciitis. If the information is available, the table below includes how often the...

**Answer**:

The signs and symptoms of Geniospasm include chin myoclonus. This information is derived from the Human Phenotype Ontology, which documents the occurrence of symptoms in individuals with Geniospasm [medquad_ans_0002519-2].

**Citations detected**: `medquad_ans_0002519-2`
**Supported citations**: `medquad_ans_0002519-2`
**Unsupported citations**: _none_

## Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Top docs**:

1. `medquad_ans_0000078-1__dup2` (score=0.7000) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens, your lungs can't p...
2. `medquad_ans_0000253-1__dup2` (score=0.4396) - Cystic fibrosis (CF) is an inherited disease of the mucus and sweat glands. It affects mostly your lungs, pancreas, liver, intestines, sinuses, and sex organs. CF causes your mucus to be thick and sticky. The mucus clogs the lungs, causi...
3. `medquad_ans_0000758-1__dup2` (score=0.4208) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough oxygen. Causes of pu...
4. `medquad_ans_0000049-3__dup2` (score=0.3435) - A defect in the CFTR gene causes cystic fibrosis (CF). This gene makes a protein that controls the movement of salt and water in and out of your body's cells. In people who have CF, the gene makes a protein that doesn't work well. This c...
5. `medquad_ans_0001551-1` (score=0.2759) - Congenital pulmonary alveolar proteinosis is a rare form of respiratory failure that is present from birth. In this condition, a type of protein builds up in the air sacs (alveoli) of the lungs, making breathing difficult. Congenital pul...

**Answer**:

The provided context does not specify the exact genetic changes related to idiopathic pulmonary fibrosis (IPF). It mentions that genetics may play a role in causing IPF and that familial IPF occurs when more than one member of a family has the disease [medquad_ans_0000078-1__dup2]. However, no details about specific genetic mutations or changes are given. Therefore, based on the available information, the genetic changes related to IPF cannot be determined.

**Citations detected**: `medquad_ans_0000078-1__dup2`
**Supported citations**: `medquad_ans_0000078-1__dup2`
**Unsupported citations**: _none_

## Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Top docs**:

1. `medquad_ans_0003593-4` (score=0.7000) - How is Liddle syndrome inherited? Liddle syndrome is inherited in an autosomal dominant manner. This means that only one mutated copy of the disease-causing gene in each cell is sufficient to cause the condition. The mutated copy of the ...
2. `medquad_ans_0004750-4` (score=0.2984) - Is Parsonage Turner syndrome inherited? Parsonage Turner syndrome, which is also known as idiopathic neuralgic amyotrophy, is not inherited. However, an inherited form of neuralgic amyotrophy does exist, which is passed down through fami...
3. `medquad_ans_0000673-4` (score=0.2862) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
4. `medquad_ans_0006446-4` (score=0.1918) - Is Williams syndrome inherited?
5. `medquad_ans_0004784-4` (score=0.1753) - How is Pelizaeus-Merzbacher disease inherited?

**Answer**:

The provided context does not contain information about the inheritance of Potocki-Shaffer syndrome. Therefore, it is not possible to determine based on the given text. [no_context]

**Citations detected**: `no_context`
**Supported citations**: _none_
**Unsupported citations**: `no_context`

## Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Top docs**:

1. `medquad_ans_0000177-1__dup2` (score=0.7000) - Summary : Children's healthy development depends on safe and positive experiences when they are very young. If you work or go to school, you want to know that your child is in good hands while you are away. You may choose in-home care, w...
2. `medquad_ans_0000189-1__dup2` (score=0.3484) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise and insuring their ...
3. `medquad_ans_0000744-1__dup2` (score=0.2309) - Summary : Prenatal care is the health care you get while you are pregnant. It includes your checkups and prenatal testing. Prenatal care can help keep you and your baby healthy. It lets your health care provider spot health problems earl...
4. `medquad_ans_0000175-1__dup2` (score=0.2139) - Child abuse is doing something or failing to do something that results in harm to a child or puts a child at risk of harm. Child abuse can be physical, sexual or emotional. Neglect, or not providing for a child's needs, is also a form of...
5. `medquad_ans_0000227-5` (score=0.1882) - These resources address the diagnosis or management of CHILD syndrome: - Gene Review: Gene Review: NSDHL-Related Disorders - Genetic Testing Registry: Child syndrome These resources from MedlinePlus offer information about the diagnosis ...

**Answer**:

Yes, there is information about child care. Child care options include in-home care, where the caregiver comes to your home, or the child going to the caregiver's home, and also child care centers. It is important to get to know your child's caregivers, as they will be a big part of your child's life. The caregiver's training should include knowledge of how young children learn and grow, positive and consistent discipline, knowledge of the signs that a child is sick, cleanliness and safety practices to help keep kids from getting sick or hurt, and basic first aid [medquad_ans_0000177-1__dup2].

**Citations detected**: `medquad_ans_0000177-1__dup2`
**Supported citations**: `medquad_ans_0000177-1__dup2`
**Unsupported citations**: _none_

## Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Top docs**:

1. `medquad_ans_0000046-2` (score=0.7000) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
2. `medquad_ans_0000046-5` (score=0.3738) - These resources address the diagnosis or management of alternating hemiplegia of childhood: - The Great Ormond Street Hospital - University of Utah School of Medicine These resources from MedlinePlus offer information about the diagnosis...
3. `medquad_ans_0000016-3__dup1` (score=0.3697) - Children with the benign form of alternating hemiplegia have a good prognosis. Those who experience the more severe form have a poor prognosis because intellectual and mental capacities do not respond to drug therapy, and balance and gai...
4. `medquad_ans_0000016-1__dup1` (score=0.2485) - Alternating hemiplegia is a rare neurological disorder that develops in childhood, most often before the child is 18 months old. The disorder is characterized by recurrent episodes of paralysis that involve one or both sides of the body,...
5. `medquad_ans_0000046-1` (score=0.2303) - Alternating hemiplegia of childhood is a neurological condition characterized by recurrent episodes of temporary paralysis, often affecting one side of the body (hemiplegia). During some episodes, the paralysis alternates from one side o...

**Answer**:

Approximately 1 in 1 million people are affected by alternating hemiplegia of childhood [medquad_ans_0000046-2].

**Citations detected**: `medquad_ans_0000046-2`
**Supported citations**: `medquad_ans_0000046-2`
**Unsupported citations**: _none_

## Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Top docs**:

1. `medquad_ans_0003687-2` (score=0.7000) - What are the signs and symptoms of Lupus? You can read about the signs and symptoms of lupus from MedlinePlus and the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS). The Human Phenotype Ontology provides th...
2. `medquad_ans_0000159-3__dup2` (score=0.6926) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time or shortly after lu...
3. `medquad_ans_0000186-1__dup1` (score=0.5022) - Lupus (also called systemic lupus erythematosus) is a disorder of the immune system. Normally, the immune system protects the body against invading infections and cancers. In lupus, the immune system is over-active and produces increased...
4. `medquad_ans_0000569-1__dup2` (score=0.3607) - If you have lupus, your immune system attacks healthy cells and tissues by mistake. This can damage your joints, skin, blood vessels and organs. There are many kinds of lupus. The most common type, systemic lupus erythematosus, affects m...
5. `medquad_ans_0003688-1` (score=0.2868) - Lupus nephritis is a kidney disorder that is a complication of systemic lupus erythematous (SLE), commonly known as lupus. The symptoms of lupus nephritis include blood in the urine, a foamy appearance to the urine, high blood pressure, ...

**Answer**:

The symptoms of Lupus include abdominal pain, abnormality of temperature regulation, abnormality of the heart valves, abnormality of the pericardium, alopecia, arthralgia, arthritis, autoimmunity, chest pain, cutaneous photosensitivity, skin rash, thrombocytopenia, thrombophlebitis, abnormal pyramidal signs, abnormal tendon morphology, abnormality of the autonomic nervous system, abnormality of the endocardium, abnormality of the pleura, anorexia, arterial thrombosis, aseptic leukocyturia, bone marrow hypocellularity, conjunctival telangiectasia, cranial nerve paralysis, cutis marmorata, dry skin, eczema, edema of the lower limbs, glomerulopathy, hallucinations, hematuria, hepatomegaly, hyperkeratosis, hypoproteinemia, increased antibody level in blood

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Top docs**:

1. `medquad_ans_0000194-2` (score=0.7000) - Cockayne syndrome occurs in about 2 per million newborns in the United States and Europe.
2. `medquad_ans_0000194-1` (score=0.6803) - Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features of this disorder include a failure to gain weight and grow at the expected rate (failure to thrive), abnormally small head...
3. `medquad_ans_0001395-1` (score=0.5914) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newborn, microcephaly, and...
4. `medquad_ans_0001398-1` (score=0.5914) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newborn, microcephaly, and...
5. `medquad_ans_0001397-1` (score=0.5914) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newborn, microcephaly, and...

**Answer**:

Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging (progeria). It is marked by failure to gain weight and grow at the expected rate (failure to thrive), abnormally small head size (microcephaly), and impaired development of the nervous system. Affected individuals have extreme sensitivity to sunlight (photosensitivity), where even small amounts of sun exposure can cause sunburn. Other possible signs and symptoms include hearing loss, eye abnormalities, severe tooth decay, bone abnormalities, and changes in the brain visible on brain scans. The syndrome is divided into subtypes based on severity and age of onset: classical (type I) presents in early childhood (usually after age 1 year), type II (severe or "early-onset") has symptoms apparent at birth and is associated with poor survival into childhood, and type III is the mildest form, appearing later in childhood. Cockayne syndrome is caused by mutations in

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Top docs**:

1. `medquad_ans_0000029-8__dup1` (score=0.7000) - Gingivitis is inflammation of the gums. In gingivitis, the gums become red, swollen and can bleed easily. Gingivitis is a mild form of gum disease. It can usually be reversed with daily brushing and flossing, and regular cleaning by a de...
2. `medquad_ans_0000029-6__dup1` (score=0.6891) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead to sore or bleeding...
3. `medquad_ans_0000413-1__dup1` (score=0.6061) - If you have gum disease, you're not alone. Many U.S. adults currently have some form of the disease. It ranges from simple gum inflammation, called gingivitis, to serious damage to the tissue and bone supporting the teeth. In the worst c...
4. `medquad_ans_0000029-1__dup5` (score=0.5875) - An Infection of the Gums and Surrounding Tissues Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. The two forms of gum disease are gingivitis, a mild form that is reversible with goo...
5. `medquad_ans_0000029-12` (score=0.5618) - In some studies, researchers have observed that people with periodontal disease (when compared to people without periodontal disease) were more likely to develop heart disease or have difficulty controlling their blood sugar. But so far,...

**Answer**:

Gum (periodontal) disease is an infection of the gums and surrounding tissues that hold teeth in place. It ranges from a mild form called gingivitis, which is reversible with good oral hygiene such as daily brushing and flossing and regular cleanings by a dentist or dental hygienist, to a more severe form called periodontitis, which can damage the soft tissues and bone supporting the teeth. If left untreated, periodontitis can lead to tooth loss [medquad_ans_0000029-1__dup5]. In its early stages, gum disease is usually painless, and many people are not aware they have it; in more advanced stages, it can cause sore gums and pain when chewing [medquad_ans_0000029-6__dup1]. Gingivitis is characterized by red, swollen gums that bleed easily, and it does not involve loss of bone or tissue

**Citations detected**: `medquad_ans_0000029-1__dup5`, `medquad_ans_0000029-6__dup1`
**Supported citations**: `medquad_ans_0000029-1__dup5`, `medquad_ans_0000029-6__dup1`
**Unsupported citations**: _none_

## Example 212 (`0000494-2`)

**Question**: How many people are affected by hyperkalemic periodic paralysis ?

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.7000) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0000494-5` (score=0.4060) - These resources address the diagnosis or management of hyperkalemic periodic paralysis: - Gene Review: Gene Review: Hyperkalemic Periodic Paralysis - Genetic Testing Registry: Familial hyperkalemic periodic paralysis - Genetic Testing Re...
3. `medquad_ans_0000505-2` (score=0.3735) - Although its exact prevalence is unknown, hypokalemic periodic paralysis is estimated to affect 1 in 100,000 people. Men tend to experience symptoms of this condition more often than women.
4. `medquad_ans_0003000-2` (score=0.3728) - What are the signs and symptoms of Hyperkalemic periodic paralysis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hyperkalemic periodic paralysis. If the information is available, the table below incl...
5. `medquad_ans_0000494-1` (score=0.3335) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the...

**Answer**:

Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people [medquad_ans_0000494-2].

**Citations detected**: `medquad_ans_0000494-2`
**Supported citations**: `medquad_ans_0000494-2`
**Unsupported citations**: _none_

## Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Top docs**:

1. `medquad_ans_0005212-1` (score=0.7000) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end of the bone), widenin...
2. `medquad_ans_0004813-1` (score=0.1769) - Periodic fever, aphthous stomatitis, pharyngitis, cervical adenitis (PFAPA) is a periodic disease, which is a heterogeneous group of disorders characterized by short episodes of illness that regularly recur for several years alternated w...
3. `medquad_ans_0005016-1` (score=0.1065) - Prader-Willi syndrome (PWS) is a genetic condition that affects many parts of the body. Infants with PWS have severe hypotonia (low muscle tone), feeding difficulties, and slow growth. In later infancy or early childhood, affected childr...
4. `medquad_ans_0000227-1__dup1` (score=0.0994) - Pelizaeus-Merzbacher disease (PMD) is a rare, progressive, degenerative central nervous system disorder in which coordination, motor abilities, and intellectual function deteriorate. The disease is one of a group of gene-linked disorders...
5. `medquad_ans_0000054-7` (score=0.0983) - Peripheral arterial disease (P.A.D.) is a disease in which plaque (plak) builds up in the arteries that carry blood to your head, organs, and limbs. Plaque is made up of fat, cholesterol, calcium, fibrous tissue, and other substances in ...

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (relative constriction of the diaphysis and flaring of the metaphysis), widening of the ribs and clavicles, platyspondyly (flattening of the spine bones), and cortical thinning. Cranial involvement is minimal, with some cases showing mild hyperostosis of the skull base and thickening of the frontal and occipital bones. Pyle disease is inherited in an autosomal recessive manner [medquad_ans_0005212-1].

**Citations detected**: `medquad_ans_0005212-1`
**Supported citations**: `medquad_ans_0005212-1`
**Unsupported citations**: _none_

## Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Top docs**:

1. `medquad_ans_0000051-1__dup2` (score=0.7000) - Espaol Deep vein thrombosis (throm-BO-sis), or DVT, is a blood clot that forms in a vein deep in the body. Blood clots occur when blood thickens and clumps together. Most deep vein blood clots occur in the lower leg or thigh. They also c...
2. `medquad_ans_0000256-1__dup3` (score=0.6310) - Deep vein thrombosis, or DVT, is a blood clot that forms in a vein deep in the body. Most deep vein clots occur in the lower leg or thigh. If the vein swells, the condition is called thrombophlebitis. A deep vein thrombosis can break loo...
3. `medquad_ans_0000104-1__dup4` (score=0.6296) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent blood clots from diss...
4. `medquad_ans_0000051-4__dup2` (score=0.5819) - The risk factors for deep vein thrombosis (DVT) include: A history of DVT. Conditions or factors that make your blood thicker or more likely to clot than normal. Some inherited blood disorders (such as factor V Leiden) will do this. Horm...
5. `medquad_ans_0000051-5__dup1` (score=0.5397) - The signs and symptoms of deep vein thrombosis (DVT) might be related to DVT itself or pulmonary embolism (PE). See your doctor right away if you have signs or symptoms of either condition. Both DVT and PE can cause serious, possibly lif...

**Answer**:

Deep vein thrombosis (DVT) is caused by blood thickening and clumping together, leading to the formation of a blood clot in a deep vein. Most DVTs occur in the lower leg or thigh, and risk factors include certain genetic disorders, atherosclerosis, diabetes, atrial fibrillation, overweight or obesity, metabolic syndrome, some medicines, and smoking [medquad_ans_0000104-1__dup4]. Sitting still for a long time can also increase the likelihood of developing DVT [medquad_ans_0000256-1__dup3].

**Citations detected**: `medquad_ans_0000104-1__dup4`, `medquad_ans_0000256-1__dup3`
**Supported citations**: `medquad_ans_0000104-1__dup4`, `medquad_ans_0000256-1__dup3`
**Unsupported citations**: _none_

## Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Top docs**:

1. `medquad_ans_0004079-1` (score=0.7000) - Complex II deficiency is a mitochondrial disease. Mitochondria are specialized compartments in cells that create more than 90% of the energy needed by the body. In mitochondrial diseases, the mitochondria don't work correctly resulting i...
2. `medquad_ans_0000661-1` (score=0.6420) - Mitochondrial complex III deficiency is a genetic condition that can affect several parts of the body, including the brain, kidneys, liver, heart, and the muscles used for movement (skeletal muscles). Signs and symptoms of mitochondrial ...
3. `medquad_ans_0004079-4` (score=0.6388) - How might mitochondrial complex II deficiency be treated? Treatment options for complex II deficiency may be similar to those for other mitochondrial disorders in general.[8677] The United Mitochondrial Disease Foundation (UMDF) provides...
4. `medquad_ans_0004081-2` (score=0.5876) - What are the signs and symptoms of Mitochondrial complex IV deficiency? There are currently 4 known forms of COX deficiency. The range and severity of signs and symptoms can vary widely from case to case. In one form, referred to as the ...
5. `medquad_ans_0004079-2` (score=0.4320) - What are the signs and symptoms of Mitochondrial complex II deficiency? The signs and symptoms of mitochondrial complex II deficiency can vary greatly from severe life-threatening symptoms in infancy to muscle disease beginning in adulth...

**Answer**:

The provided context does not contain any information about mitochondrial complex IV deficiency. Therefore, it is not possible to answer the question based on the given text. [no context]

**Citations detected**: `no context`
**Supported citations**: _none_
**Unsupported citations**: `no context`

## Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Top docs**:

1. `medquad_ans_0000078-2` (score=0.7000) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.
2. `medquad_ans_0000078-1` (score=0.6153) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most affected people also ...
3. `medquad_ans_0000078-5` (score=0.5428) - These resources address the diagnosis or management of ataxia with oculomotor apraxia: - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type 1 - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type 2 - Genetic Testing ...
4. `medquad_ans_0000496-1__dup1` (score=0.3971) - What are the signs and symptoms of Ataxia with Oculomotor Apraxia Type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Ataxia with Oculomotor Apraxia Type 2. If the information is available, the tabl...
5. `medquad_ans_0000495-2__dup1` (score=0.2708) - What are the signs and symptoms of Ataxia telangiectasia? Ataxia-telangiectasia affects the nervous system, immune system, and other body systems. This disorder is characterized by progressive difficulty with coordinating movements (atax...

**Answer**:

Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement, with the hallmark being difficulty coordinating movements (ataxia), often the first symptom. Most affected individuals also have oculomotor apraxia, which makes it difficult to move their eyes side-to-side, requiring them to turn their head to see peripheral vision. There are multiple types of this condition, which are very similar but caused by mutations in different genes. The two most common types (types 1 and 2) share features including involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, ataxia begins around age 4; in type 2, it begins around age 15. Chorea and myoclonus tend to disappear gradually in type 1 but persist throughout life in type 2. Type 1 individuals

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Top docs**:

1. `medquad_ans_0000221-1` (score=0.7000) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that form the inner ear are...
2. `medquad_ans_0001744-1` (score=0.5297) - What are the signs and symptoms of Deafness with labyrinthine aplasia microtia and microdontia (LAMM)? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness with labyrinthine aplasia microtia and mic...
3. `medquad_ans_0000221-5` (score=0.3440) - These resources address the diagnosis or management of LAMM syndrome: - Gene Review: Gene Review: Congenital Deafness with Labyrinthine Aplasia, Microtia, and Microdontia - Genetic Testing Registry: Deafness with labyrinthine aplasia mic...
4. `medquad_ans_0000623-1` (score=0.1611) - Mandibulofacial dysostosis with microcephaly (MFDM) is a disorder that causes abnormalities of the head and face. People with this disorder often have an unusually small head at birth, and the head does not grow at the same rate as the r...
5. `medquad_ans_0001496-1` (score=0.1441) - Congenital deafness with vitiligo and achalasia is a syndrome characterized by deafness present from birth (congenital), associated with short stature, vitiligo, muscle wasting and achalasia (swallowing difficulties). The condition was d...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects the development of the ears and teeth. It is characterized by the complete absence of inner ear structures (labyrinthine aplasia), leading to congenital sensorineural hearing loss. Affected individuals often have abnormally small outer ears (microtia) with narrow ear canals and unusually small, widely spaced teeth (microdontia). The condition may also cause delayed motor skill development due to balance issues from inner ear abnormalities. Additional signs include abnormalities of cranial nerves (seen in 90% of patients), microdontia (90%), and other features such as pointed chin, wide nasal bridge, and preauricular skin tags [medquad_ans_0000221-1][medquad_ans_0001744-1]. The syndrome

**Citations detected**: `medquad_ans_0000221-1`, `medquad_ans_0001744-1`
**Supported citations**: `medquad_ans_0000221-1`, `medquad_ans_0001744-1`
**Unsupported citations**: _none_

## Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000064-4` (score=0.7000) - Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases of Apert syndrome result from new mutations in the gene, and occur ...
2. `medquad_ans_0000064-2` (score=0.3707) - Apert syndrome affects an estimated 1 in 65,000 to 88,000 newborns.
3. `medquad_ans_0000420-1__dup1` (score=0.2655) - Apert syndrome is a disorder mainly characterized by craniosynostosis (premature fusion of skull bones, causing abnormalities in the shape of the head and face) and syndactyly (fusion or webbing or fingers and/or toes). Other signs and s...
4. `medquad_ans_0006597-4` (score=0.2491) - Can 48,XXYY syndrome be inherited?
5. `medquad_ans_0000064-1` (score=0.2156) - Apert syndrome is a genetic disorder characterized by the premature fusion of certain skull bones (craniosynostosis). This early fusion prevents the skull from growing normally and affects the shape of the head and face. In addition, a v...

**Answer**:

Yes, Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. However, almost all cases result from new mutations and occur in people with no family history of the disorder. Individuals with Apert syndrome can pass the condition to their offspring [medquad_ans_0000064-4]. The disorder is caused by mutations in the FGFR2 gene and is inherited in an autosomal dominant manner [medquad_ans_000420-1__dup1].

**Citations detected**: `medquad_ans_0000064-4`, `medquad_ans_000420-1__dup1`
**Supported citations**: `medquad_ans_0000064-4`
**Unsupported citations**: `medquad_ans_000420-1__dup1`

## Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.7000) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organisms are excreted in ...
2. `medquad_ans_0000341-1__dup3` (score=0.4540) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella burnetii usually devel...
3. `medquad_ans_0000341-4__dup2` (score=0.3915) - In the United States, Q fever outbreaks have resulted mainly from occupational exposure involving veterinarians, meat processing plant workers, sheep and dairy workers, livestock farmers, and researchers at facilities housing sheep. Prev...
4. `medquad_ans_0000341-3__dup2` (score=0.3875) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact the Rickettsial Zoon...
5. `medquad_ans_0000092-4__dup3` (score=0.2208) - CHHF virus has been successfully isolated from both blood and serum during the acute febrile phase of illness. Although not undertaken at the time of the initial cluster, virus can certainly be isolated from tissue obtained post-mortem i...

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air contaminated by dried placental material, birth fluids, and excreta of infected animals. Other rare modes of transmission include tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, individuals with acute Q fever develop a chronic form of the disease. Treatment of the acute form involves antibiotics, while treatment for the chronic form depends on the symptoms

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Top docs**:

1. `medquad_ans_0000038-1__dup2` (score=0.7000) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chronic cough, rapid breat...
2. `medquad_ans_0000038-5__dup1` (score=0.5331) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interstitial lung disease o...
3. `medquad_ans_0000038-9` (score=0.4711) - At this time, most types of childhood interstitial lung disease (chILD) can't be prevented. People who have a family history of inherited (genetic) interstitial lung disease may want to consider genetic counseling. A counselor can explai...
4. `medquad_ans_0000038-4__dup2` (score=0.4000) - Researchers don't yet know all of the causes of childhood interstitial lung disease (chILD). Many times, these diseases have no clear cause. Some conditions and factors that may cause or lead to chILD include: Inherited conditions, such ...
5. `medquad_ans_0000038-7` (score=0.3326) - Doctors diagnose childhood interstitial lung disease (chILD) based on a child's medical and family histories and the results from tests and procedures. To diagnose chILD, doctors may first need to rule out other diseases as the cause of ...

**Answer**:

Babies, children, and teens are at risk for Childhood Interstitial Lung Disease (chILD) [medquad_ans_0000038-1__dup2].

**Citations detected**: `medquad_ans_0000038-1__dup2`
**Supported citations**: `medquad_ans_0000038-1__dup2`
**Unsupported citations**: _none_

## Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.7000) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in late childhood or early...
2. `medquad_ans_0000547-1` (score=0.2485) - Juvenile idiopathic arthritis refers to a group of conditions involving joint inflammation (arthritis) that first appears before the age of 16. This condition is an autoimmune disorder, which means that the immune system malfunctions and...
3. `medquad_ans_0000087-1__dup1` (score=0.1471) - Vasculitis is an inflammation of blood vessels, which includes the veins, arteries, and capillaries. Inflammation occurs with infection or is thought to be due to a faulty immune system response. It also can be caused by other immune sys...
4. `medquad_ans_0002107-1` (score=0.1091) - Enthesitis-related juvenile idiopathic arthritis is a subtype of juvenile idiopathic arthritis that is characterized by both arthritis and inflammation of an enthesitis site (the point at which a ligament, tendon, or joint capsule attach...
5. `medquad_ans_0003322-1` (score=0.1076) - What are the signs and symptoms of Juvenile idiopathic arthritis? The Human Phenotype Ontology provides the following list of signs and symptoms for Juvenile idiopathic arthritis. If the information is available, the table below includes...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels. It is generally diagnosed in late childhood or early adulthood and specifically affects the temporal arteries, which are located at the lower sides of the skull, directly underneath the temple. Affected individuals often have no signs or symptoms aside from a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically with no family history. It is generally treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Citations detected**: `medquad_ans_0003334-1`
**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

## Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Top docs**:

1. `medquad_ans_0006577-1` (score=0.7000) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particular amino acid called...
2. `medquad_ans_0001082-1` (score=0.5142) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as organic acids. Abno...
3. `medquad_ans_0001082-2` (score=0.4452) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyryl-CoA dehydrogenase ...
4. `medquad_ans_0001090-2` (score=0.2828) - The exact incidence of 3-hydroxyacyl-CoA dehydrogenase deficiency is unknown; it has been reported in only a small number of people worldwide.
5. `medquad_ans_0001082-5` (score=0.2644) - These resources address the diagnosis or management of 2-methylbutyryl-CoA dehydrogenase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of 2-methylbutyryl-CoA dehydrogenase These resources from MedlinePlus offer i...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of the enzyme 2-methylbutyryl-CoA dehydrogenase, which is necessary for the proper processing of the amino acid isoleucine. This deficiency leads to the buildup of isoleucine and organic acids in the body, which can be toxic and cause a range of health problems. Symptoms vary widely, from severe and life-threatening to mild or absent, and may include poor feeding, lack of energy (lethargy), vomiting, irritability, difficulty breathing, seizures, coma, poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills such as standing and walking. Symptoms can be triggered by fasting, infections, or high-protein diets. The disorder is more common among individuals of Hmong ancestry, particularly in Hmong populations in southeast Asia and Hmong Americans,

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Top docs**:

1. `medquad_ans_0005840-2` (score=0.7000) - What are the signs and symptoms of Spondyloepiphyseal dysplasia congenita? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia congenita. If the information is available, the ta...
2. `medquad_ans_0005841-2` (score=0.5082) - What are the signs and symptoms of Spondyloepiphyseal dysplasia Maroteaux type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia Maroteaux type. If the information is availab...
3. `medquad_ans_0005827-1` (score=0.5020) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia joint laxity? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia joint laxity. If the information is ava...
4. `medquad_ans_0005835-1` (score=0.4818) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia with multiple dislocations? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia with multiple dislocation...
5. `medquad_ans_0005839-1` (score=0.4684) - What are the signs and symptoms of Spondyloepiphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia. If the information is available, the table below includes h...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include abnormal form of the vertebral bodies (90%), abnormality of epiphysis morphology (90%), micromelia (90%), narrow chest (90%), short neck (90%), short stature (90%), short thorax (90%), skeletal dysplasia (90%), broad forehead (50%), cleft palate (50%), hyperlordosis (50%), hypertelorism (50%), malar flattening (50%), osteoarthritis (50%), talipes (50%), cataract (7.5%), glaucoma (7.5%), hearing impairment (7.5%), kyphosis (7.5%), myopia (7.5%), nystagmus (7.5%), retinal detachment (7.5%), and scoliosis (7.5%)

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Top docs**:

1. `medquad_ans_0005219-1` (score=0.7000) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepticus). These seizures...
2. `medquad_ans_0000853-2` (score=0.6856) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
3. `medquad_ans_0000853-1` (score=0.6094) - Pyridoxine-dependent epilepsy is a condition that involves seizures beginning in infancy or, in some cases, before birth. Those affected typically experience prolonged seizures lasting several minutes (status epilepticus). These seizures...
4. `medquad_ans_0005219-2` (score=0.5104) - What are the signs and symptoms of Pyridoxine-dependent epilepsy? Those affected by pyridoxine-dependent epilepsy typically experience prolonged seizures lasting several minutes (status epilepticus). These seizures involve muscle rigidit...
5. `medquad_ans_0005219-3` (score=0.5047) - What causes pyridoxine-dependent epilepsy? Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations....

**Answer**:

Pyridoxine-dependent epilepsy is caused by mutations in the ALDH7A1 gene, which is inherited in an autosomal recessive fashion [medquad_ans_0005219-1].

**Citations detected**: `medquad_ans_0005219-1`
**Supported citations**: `medquad_ans_0005219-1`
**Unsupported citations**: _none_

## Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Top docs**:

1. `medquad_ans_0000004_6-2` (score=0.7000) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not having risk f...
2. `medquad_ans_0000004_1-2` (score=0.5017) - Age, gender, and Epstein-Barr infection can affect the risk of adult Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not havin...
3. `medquad_ans_0000004_7-2` (score=0.4524) - Signs of childhood non-Hodgkin lymphoma include breathing problems and swollen lymph nodes. These and other signs may be caused by childhood non-Hodgkin lymphoma or by other conditions. Check with a doctor if your child has any of the fo...
4. `medquad_ans_0000004_6-5` (score=0.3980) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whether there are B symp...
5. `medquad_ans_0000004_2-3` (score=0.3674) - Age, gender, and a weakened immune system can affect the risk of adult non-Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not...

**Answer**:

Children who are infected with the Epstein-Barr virus, infected with the human immunodeficiency virus (HIV), have certain diseases of the immune system, have a personal history of mononucleosis ("mono"), or have a parent or sibling with a personal history of Hodgkin lymphoma are at risk for childhood Hodgkin lymphoma [medquad_ans_0000004_6-2]. Being exposed to common infections in early childhood may decrease the risk of Hodgkin lymphoma in children due to its effect on the immune system [medquad_ans_0000004_6-2].

**Citations detected**: `medquad_ans_0000004_6-2`
**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

## Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Top docs**:

1. `medquad_ans_0001795-1` (score=0.7000) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow onset of symptoms u...
2. `medquad_ans_0001795-2` (score=0.3826) - What are the signs and symptoms of Dermatitis herpetiformis ? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatitis herpetiformis . If the information is available, the table below includes how oft...
3. `medquad_ans_0000095-2__dup3` (score=0.3813) - Dermatitis herpetiformis is caused by the deposit of immunoglobulin A (IgA) in the skin, which triggers further immunologic reactions resulting in lesion formation. DH is an external manifestation of an abnormal immune response to gluten...
4. `medquad_ans_0001795-3` (score=0.3401) - How might dermatitis herpetiformis be treated? The antibiotic dapsone is extremely effective in treating this condition. Symptomatic improvement may occur in as little as several hours after the first dose. However, dapsone may cause ser...
5. `medquad_ans_0000095-1__dup4` (score=0.2680) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation may precede lesion f...

**Answer**:

Dermatitis herpetiformis is a rare, chronic skin disorder characterized by groups of severely itchy blisters and raised skin lesions, commonly found on the knees, elbows, buttocks, and shoulder blades. It is associated with the inability to digest gluten and is caused by the deposit of immunoglobulin A (IgA) in the skin, triggering an immune response to gluten, specifically targeting the skin antigen epidermal transglutaminase. The condition is frequently linked to autoimmunity and malabsorption, with symptoms including abnormal blistering, pruritus, red lesions resembling hives, and fluid-filled sores. It typically presents in adulthood but can also affect children. The disease is often treated with dapsone [medquad_ans_0001795-1]. Family studies indicate that approximately 5% of first-degree relatives of an affected individual will also have the condition [medquad_ans_0

**Citations detected**: `medquad_ans_0001795-1`
**Supported citations**: `medquad_ans_0001795-1`
**Unsupported citations**: _none_

## Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Top docs**:

1. `medquad_ans_0000709-5` (score=0.7000) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Registry: Congenital myo...
2. `medquad_ans_0000210-2__dup2` (score=0.6965) - Treatment for myotonia may include mexiletine, quinine, phenytoin, and other anticonvulsant drugs. Physical therapy and other rehabilitative measures may help muscle function.
3. `medquad_ans_0000040-2__dup1` (score=0.6922) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors have had some succes...
4. `medquad_ans_0000210-3__dup1` (score=0.6864) - Myotonia is a chronic disorder. Symptoms may improve later in life.
5. `medquad_ans_0000040-4__dup1` (score=0.5287) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to myotonia congenita and also supports additional research through grants to major research institutions across the country. Current research ...

**Answer**:

Treatments for myotonia congenita may include mexiletine, quinine, phenytoin, and other anticonvulsant drugs. Physical therapy and other rehabilitative measures may help improve muscle function. Most people with myotonia congenita do not require special treatments, as stiff muscles usually resolve with exercise or light movement, especially after resting. For individuals with more limiting symptoms, medications such as quinine or anticonvulsant drugs like phenytoin have been used successfully. [medquad_ans_0000210-2__dup2] [medquad_ans_0000040-2__dup1]

**Citations detected**: `medquad_ans_0000210-2__dup2`, `medquad_ans_0000040-2__dup1`
**Supported citations**: `medquad_ans_0000210-2__dup2`, `medquad_ans_0000040-2__dup1`
**Unsupported citations**: _none_

## Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Top docs**:

1. `medquad_ans_0006449-5` (score=0.7000) - How might Wilson disease be treated? There is currently no cure for Wilson disease; however, therapies exist that aim to reduce or control the amount of copper that accumulates in the body. Affected people require lifelong treatment, whi...
2. `medquad_ans_0001037-2` (score=0.5165) - Wilson disease is a rare disorder that affects approximately 1 in 30,000 individuals.
3. `medquad_ans_0001037-5` (score=0.5144) - These resources address the diagnosis or management of Wilson disease: - Gene Review: Gene Review: Wilson Disease - Genetic Testing Registry: Wilson's disease - MedlinePlus Encyclopedia: Wilson's disease - National Human Genome Research ...
4. `medquad_ans_0000133-12` (score=0.5136) - A person cannot prevent Wilson disease; however, people with a family history of Wilson disease, especially those with an affected sibling or parent, should talk with a health care provider about testing. A health care provider may be ab...
5. `medquad_ans_0006449-1` (score=0.4525) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver, brain and eyes. Sig...

**Answer**:

Treatment for Wilson disease involves lifelong management aimed at reducing or controlling copper accumulation in the body. This may include specific medications and dietary modifications. In cases where treatment is ineffective or liver failure develops, a liver transplant may be necessary [medquad_ans_0006449-5]. Early diagnosis, especially in individuals with a family history of the disease, can lead to timely intervention that reduces or prevents organ damage [medquad_ans_0000133-12]. People with a family history should consult a healthcare provider about testing and may benefit from genetic testing to identify gene mutations, with referral to a geneticist if needed [medquad_ans_0000133-12]. Additional resources for diagnosis and management include GeneReviews, the Genetic Testing Registry, MedlinePlus Encyclopedia, and the National Institute of Diabetes and Digestive and Kidney Disease (NIDDK) [medquad_ans_00

**Citations detected**: `medquad_ans_0006449-5`, `medquad_ans_0000133-12`
**Supported citations**: `medquad_ans_0006449-5`, `medquad_ans_0000133-12`
**Unsupported citations**: _none_

## Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Top docs**:

1. `medquad_ans_0000592-3` (score=0.7000) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are related to an existing...
2. `medquad_ans_0000592-2` (score=0.5895) - Lennox-Gastaut syndrome affects an estimated 1 in 50,000 to 1 in 100,000 children. This condition accounts for about 4 percent of all cases of childhood epilepsy. For unknown reasons, it appears to be more common in males than in females.
3. `medquad_ans_0000592-4` (score=0.4674) - Most cases of Lennox-Gastaut syndrome are sporadic, which means they occur in people with no history of the disorder in their family. However, 3 to 30 percent of people with this condition have a family history of some type of epilepsy. ...
4. `medquad_ans_0000592-5` (score=0.4672) - These resources address the diagnosis or management of Lennox-Gastaut syndrome: - Cleveland Clinic - Genetic Testing Registry: Epileptic encephalopathy Lennox-Gastaut type - National Institute of Neurological Disorders and Stroke: Diagno...
5. `medquad_ans_0000179-3__dup1` (score=0.4558) - The prognosis for individuals with Lennox-Gastaut syndrome varies. There is no cure for the disorder. Complete recovery, including freedom from seizures and normal development, is very unusual.

**Answer**:

Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component [medquad_ans_0000592-3].

**Citations detected**: `medquad_ans_0000592-3`
**Supported citations**: `medquad_ans_0000592-3`
**Unsupported citations**: _none_

## Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Top docs**:

1. `medquad_ans_0000066-2` (score=0.7000) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
2. `medquad_ans_0000066-5` (score=0.4277) - These resources address the diagnosis or management of arginine:glycine amidinotransferase deficiency: - Gene Review: Gene Review: Creatine Deficiency Syndromes - Genetic Testing Registry: Arginine:glycine amidinotransferase deficiency T...
3. `medquad_ans_0003472-1` (score=0.3212) - L-arginine:glycine amidinotransferase (AGAT) deficiency is a rare condition that primarily affects the brain. People with AGAT deficiency generally have mild to moderate intellectual disability. Other signs and symptoms may include seizu...
4. `medquad_ans_0003472-2` (score=0.3137) - What are the signs and symptoms of L-arginine:glycine amidinotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for L-arginine:glycine amidinotransferase deficiency. If the information ...
5. `medquad_ans_0000445-1__dup1` (score=0.1950) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of arginine in the blo...

**Answer**:

The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families [medquad_ans_0000066-2]. Therefore, the exact number of people affected is not known.

**Citations detected**: `medquad_ans_0000066-2`
**Supported citations**: `medquad_ans_0000066-2`
**Unsupported citations**: _none_

## Example 231 (`0002399-2`)

**Question**: What are the symptoms of Focal dermal hypoplasia ?

**Top docs**:

1. `medquad_ans_0002399-2` (score=0.7000) - What are the signs and symptoms of Focal dermal hypoplasia? Focal dermal hypoplasia is usually evident from birth and primarily affects the skin, skeleton, eyes, and face. The signs and symptoms of vary widely, although almost all affect...
2. `medquad_ans_0002399-1` (score=0.4575) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. The skin abnormalities are present from birth and can include streaks of very thin skin (dermal hypoplasia), cutis aplasia, and tela...
3. `medquad_ans_0000379-2` (score=0.3584) - Focal dermal hypoplasia appears to be a rare condition, although its exact prevalence is unknown.
4. `medquad_ans_0000379-1` (score=0.2566) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. About 90 percent of affected individuals are female. Males usually have milder signs and symptoms than females. Although intelligenc...
5. `medquad_ans_0004054-1` (score=0.2428) - What are the signs and symptoms of Microsomia hemifacial radial defects? The Human Phenotype Ontology provides the following list of signs and symptoms for Microsomia hemifacial radial defects. If the information is available, the table ...

**Answer**:

The symptoms of Focal dermal hypoplasia include skin abnormalities such as streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), telangiectases, and streaks of slightly darker or lighter skin. These can cause pain, itching, irritation, or lead to skin infections. With age, most develop wart-like growths (papillomas) around the nostrils, lips, anus, and female genitalia, and may also be present in the throat, causing problems with swallowing, breathing, or sleeping. Other features include small, ridged fingernails and toenails, sparse or brittle scalp hair, hand and foot abnormalities such as missing fingers or toes (oligodactyly), webbed or fused fingers or toes (syndactyly), and a deep split in

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=0.7000) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or cooking puts you at...
2. `medquad_ans_0000418-1__dup2` (score=0.5970) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (meat-eating) animals...
3. `medquad_ans_0000418-6` (score=0.4206) - Safe and effective prescription drugs are available to treat both Trichinella infection and the symptoms that occur as a result of infection. Treatment should begin as soon as possible; a doctor will make the decision to treat based upon...
4. `medquad_ans_0000418-5__dup1` (score=0.3037) - A diagnosis of trichinellosis is made in patients whose signs and symptoms are compatible with trichinellosis, have a positive laboratory test for Trichinella, and who can recall eating raw or undercooked pork or wild game meat. Laborato...
5. `medquad_ans_0000091-2__dup3` (score=0.2071) - Chagas disease, or American trypanosomiasis, is caused by the parasite Trypanosoma cruzi. Infection is most commonly acquired through contact with the feces of an infected triatomine bug (or "kissing bug"), a blood-sucking insect that fe...

**Answer**:

People at risk for trichinellosis (also known as trichinosis) include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating undercooked wild game, especially bear meat, puts individuals at risk. Additionally, individuals who consume raw or undercooked meat from animals such as domestic pigs or wild boar, which can harbor Trichinella larvae, are also at risk [medquad_ans_0000418-1\_\_dup2][medquad_ans_0000418-2__dup1].

**Citations detected**: `medquad_ans_0000418-1__dup2`, `medquad_ans_0000418-2__dup1`
**Supported citations**: `medquad_ans_0000418-1__dup2`, `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

## Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Top docs**:

1. `medquad_ans_0001924-2` (score=0.7000) - What are the signs and symptoms of Dravet syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Dravet syndrome. If the information is available, the table below includes how often the symptom is se...
2. `medquad_ans_0005016-2` (score=0.2719) - What are the signs and symptoms of Prader-Willi syndrome? In infancy, Prader-Willi syndrome (PWS) is characterized by weak muscle tone (hypotonia), feeding difficulties, poor growth, and delayed development. In later infancy or early chi...
3. `medquad_ans_0002063-2` (score=0.2300) - What are the signs and symptoms of Ehlers-Danlos syndrome, vascular type? The signs and symptoms of Ehlers-Danlos syndrome (EDS), vascular type vary but may include: Fragile tissues (including arteries, muscles and internal organs) that ...
4. `medquad_ans_0003700-2` (score=0.2150) - What are the signs and symptoms of Lymphedema-distichiasis syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lymphedema-distichiasis syndrome. If the information is available, the table below in...
5. `medquad_ans_0002055-2` (score=0.1133) - What are the signs and symptoms of Ehlers-Danlos syndrome, dermatosparaxis type? The signs and symptoms of Ehlers-Danlos syndrome (EDS), dermatosparaxis type vary but may include: Soft, doughy skin that is extremely fragile Severe bruisi...

**Answer**:

The symptoms of Dravet syndrome include absence seizures, ataxia, autosomal dominant inheritance, cerebral atrophy, cortical visual impairment, epileptic encephalopathy, focal seizures with impairment of consciousness or awareness, generalized myoclonic seizures, hemiclonic seizures, infantile onset, mental deterioration, motor delay, postnatal microcephaly, and status epilepticus [medquad_ans_0001924-2].

**Citations detected**: `medquad_ans_0001924-2`
**Supported citations**: `medquad_ans_0001924-2`
**Unsupported citations**: _none_

## Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Top docs**:

1. `medquad_ans_0002533-3` (score=0.7000) - What causes Gerstmann-Straussler-Scheinker disease? Gerstmann-Straussler-Scheinker disease (GSS) is usually caused by certain changes (mutations) in the PRNP gene. PRNP encodes a protein called prion protein. Although the exact function ...
2. `medquad_ans_0004784-4` (score=0.5916) - How is Pelizaeus-Merzbacher disease inherited?
3. `medquad_ans_0000011-13` (score=0.5311) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten CJD from medical proc...
4. `medquad_ans_0000007-2__dup1` (score=0.3035) - Health Alert: Adrenal Crisis Causes Death in Some People Who Were Treated with Human Growth Hormone National Hormone and Pituitary Program (NHPP): Information for People Treated with Pituitary Human Growth Hormone (Summary) National Horm...
5. `medquad_ans_0006446-4` (score=0.2534) - Is Williams syndrome inherited?

**Answer**:

Genetic changes related to prion disease are mutations in the PRNP gene. These mutations result in the production of an abnormally shaped prion protein, which accumulates in the brain and forms clumps that damage or destroy neurons. This process leads to the signs and symptoms of prion diseases such as Gerstmann-Straussler-Scheinker disease (GSS) [medquad_ans_0002533-3]. In inherited cases of Creutzfeldt-Jakob disease (CJD), an abnormal gene in PRNP leads to changes in the prion protein that make it more likely to adopt an abnormal shape [medquad_ans_0000011-13].

**Citations detected**: `medquad_ans_0002533-3`, `medquad_ans_0000011-13`
**Supported citations**: `medquad_ans_0002533-3`, `medquad_ans_0000011-13`
**Unsupported citations**: _none_

## Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Top docs**:

1. `medquad_ans_0006430-1` (score=0.7000) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information is available, the ta...
2. `medquad_ans_0003459-2` (score=0.2856) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficulty lifting objects ...
3. `medquad_ans_0003474-2` (score=0.1429) - What are the signs and symptoms of Larsen syndrome? The signs and symptoms of Larsen syndrome vary from person to person, but may include the following: Joint dislocation (especially of the hips, knees, and elbows) Hypermobile joints Fla...
4. `medquad_ans_0002388-1` (score=0.1254) - What are the signs and symptoms of Fitzsimmons-Guilbert syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fitzsimmons-Guilbert syndrome. If the information is available, the table below includes...
5. `medquad_ans_0006442-2` (score=0.1239) - What are the signs and symptoms of Wiedemann-Steiner syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Wiedemann-Steiner syndrome. If the information is available, the table below includes how o...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, hypoplasia of the radius, oligodactyly (hands), proximal placement of thumb, proximal radial head dislocation, cleft palate, cleft upper lip, high palate, hydronephrosis, hypotelorism, long face, mesomelia, narrow face, and high palate [medquad_ans_0006430-1].

**Citations detected**: `medquad_ans_0006430-1`
**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

## Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.7000) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the table below includes h...
2. `medquad_ans_0004524-1` (score=0.3721) - What are the signs and symptoms of Okamoto syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Okamoto syndrome. If the information is available, the table below includes how often the symptom is ...
3. `medquad_ans_0002061-2` (score=0.1938) - What are the signs and symptoms of Ehlers-Danlos syndrome, progeroid type? Ehlers-Danlos syndrome refers to a group of connective tissue disorders characterized by stretchy or kneadable skin, double jointedness, and delayed healing of sk...
4. `medquad_ans_0003546-2` (score=0.1740) - What are the signs and symptoms of Leprechaunism? The Human Phenotype Ontology provides the following list of signs and symptoms for Leprechaunism. If the information is available, the table below includes how often the symptom is seen i...
5. `medquad_ans_0004961-1` (score=0.1314) - What are the signs and symptoms of Polyneuropathy mental retardation acromicria premature menopause? The Human Phenotype Ontology provides the following list of signs and symptoms for Polyneuropathy mental retardation acromicria prematur...

**Answer**:

The symptoms of Premature aging Okamoto type include abnormality of the blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face [medquad_ans_0005029-1]. The frequency of these symptoms is based on data from the Human Phenotype Ontology, which collects information from sources such as Orphanet, with frequencies often estimated as percentages or fractions of patients exhibiting each feature [medquad_ans_0005029-1].

**Citations detected**: `medquad_ans_0005029-1`
**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

## Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Top docs**:

1. `medquad_ans_0000329-2` (score=0.7000) - Essential thrombocythemia affects an estimated 1 to 24 per 1 million people worldwide.
2. `medquad_ans_0001065-2` (score=0.5995) - The estimated incidence of X-linked thrombocytopenia is between 1 and 10 per million males worldwide; this condition is rarer among females.
3. `medquad_ans_0000979-1` (score=0.5157) - Thrombocytopenia-absent radius (TAR) syndrome is characterized by the absence of a bone called the radius in each forearm. Affected individuals also have a shortage (deficiency) of blood cells involved in clotting (platelets). This plate...
4. `medquad_ans_0000484-2` (score=0.2189) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
5. `medquad_ans_0000770-2` (score=0.2027) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.

**Answer**:

The provided context does not specify the exact number of people affected by thrombocytopenia-absent radius (TAR) syndrome. While it mentions that TAR syndrome is characterized by a deficiency of platelets and associated malformations, and that it is unusual among similar conditions, no incidence or estimated prevalence is given for the condition in the context. Therefore, based on the available information, the number of people affected by TAR syndrome cannot be determined.

[medquad_ans_0000979-1]

**Citations detected**: `medquad_ans_0000979-1`
**Supported citations**: `medquad_ans_0000979-1`
**Unsupported citations**: _none_

## Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Top docs**:

1. `medquad_ans_0000972-5` (score=0.7000) - These resources address the diagnosis or management of Tay-Sachs disease: - Gene Review: Gene Review: Hexosaminidase A Deficiency - Genetic Testing Registry: Tay-Sachs disease - MedlinePlus Encyclopedia: Tay-Sachs Disease These resources...
2. `medquad_ans_0006017-5` (score=0.6489) - How might children with Tay-Sachs disease be treated? Although several attempts have been made at purified enzyme replacement therapy for children with Tay-Sachs disease, none has been successful. Cellular infusions and even bone marrow ...
3. `medquad_ans_0000871-1__dup2` (score=0.6381) - Tay-Sachs disease is a rare, inherited disorder. It causes too much of a fatty substance to build up in the brain. This buildup destroys nerve cells, causing mental and physical problems. Infants with Tay-Sachs disease appear to develop ...
4. `medquad_ans_0000265-2__dup1` (score=0.5858) - Presently there is no specific treatment for Tay-Sachs disease. Anticonvulsant medicine may initially control seizures. Other supportive treatment includes proper nutrition and hydration and techniques to keep the airway open. Children m...
5. `medquad_ans_0000265-1__dup1` (score=0.4677) - Tay-Sachs disease is a inherited metabolic disease caused by the harmful buildup of lipids (fatty materials such as oils and acids) in various cells and tissues in the body. It is part of a group of genetic disorders called the GM2 gangl...

**Answer**:

The provided context does not mention any ongoing research or clinical trials for Tay-Sachs disease. [medquad_ans_0000972-5], [medquad_ans_0006017-5], [medquad_ans_0000871-1__dup2], and [medquad_ans_0000265-2__dup1] discuss diagnosis, management, and supportive care but do not reference specific research or clinical trials. Therefore, based on the given information, no such details are available.

**Citations detected**: `medquad_ans_0000972-5`, `medquad_ans_0006017-5`, `medquad_ans_0000871-1__dup2`, `medquad_ans_0000265-2__dup1`
**Supported citations**: `medquad_ans_0000972-5`, `medquad_ans_0006017-5`, `medquad_ans_0000871-1__dup2`, `medquad_ans_0000265-2__dup1`
**Unsupported citations**: _none_

## Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Top docs**:

1. `medquad_ans_0000008-5` (score=0.7000) - These resources address the diagnosis or management of achromatopsia: - Gene Review: Gene Review: Achromatopsia - Genetic Testing Registry: Achromatopsia - MedlinePlus Encyclopedia: Color Vision Test These resources from MedlinePlus offe...
2. `medquad_ans_0000008-2` (score=0.4718) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on one of the Eastern Ca...
3. `medquad_ans_0000008-1` (score=0.4297) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder...
4. `medquad_ans_0000025-1__dup1` (score=0.2724) - An acanthoma is a small, reddish bump that usually develops on the skin of an older adult. There are several types of acanthoma, including "acantholytic", "epidermolytic", "clear cell", and "melanoacanthoma". Though most individuals have...
5. `medquad_ans_0000008-3` (score=0.2548) - Achromatopsia results from changes in one of several genes: CNGA3, CNGB3, GNAT2, PDE6C, or PDE6H. A particular CNGB3 gene mutation underlies the condition in Pingelapese islanders. Achromatopsia is a disorder of the retina, which is the ...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. The condition also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals may also have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia results from changes in one of several genes: CNGA3, CNGB3, GNAT2, PDE6C, or PDE6H. It is a disorder of the retina,

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Top docs**:

1. `medquad_ans_0000039-11` (score=0.7000) - Treatments your doctor will likely recommend before knee replacement include - exercises to strengthen the muscles around the knee and improve flexibility - weight loss, if needed, to reduce the load the knee must bear - walking aids suc...
2. `medquad_ans_0000039-26` (score=0.5018) - You can learn more about knee replacement from the following resources. National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) Information Clearinghouse National Institutes of Health 1 AMS Circle Bethesda, MD 20892...
3. `medquad_ans_0000544-1__dup1` (score=0.4570) - Knee replacement is surgery for people with severe knee damage. Knee replacement can relieve pain and allow you to be more active. Your doctor may recommend it if you have knee pain and medicine and other treatments are not helping you a...
4. `medquad_ans_0000061-13` (score=0.4382) - Medication, exercise, and, in some cases, surgery are common treatments for this disease. Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation. People with rheuma...
5. `medquad_ans_0000029-18` (score=0.4119) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.

**Answer**:

The treatments recommended before knee replacement include exercises to strengthen the muscles around the knee and improve flexibility, weight loss if needed to reduce the load on the knee, using walking aids such as canes to reduce stress on the joint, wearing shoe inserts to improve knee alignment, and taking medicines to relieve pain [medquad_ans_0000039-11].

Knee replacement surgery itself involves removing damaged cartilage and bone from the knee joint and replacing them with a man-made surface of metal and plastic in a total knee replacement, or replacing only one part of the joint in a partial knee replacement [medquad_ans_0000544-1__dup1]. After surgery, patients may experience scarring, blood clots, and rarely infections, and will generally no longer be able to perform certain activities like jogging or high-impact sports [medquad_ans_0000544-1\_\_dup

**Citations detected**: `medquad_ans_0000039-11`, `medquad_ans_0000544-1__dup1`
**Supported citations**: `medquad_ans_0000039-11`, `medquad_ans_0000544-1__dup1`
**Unsupported citations**: _none_

## Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Top docs**:

1. `medquad_ans_0000090-5__dup1` (score=0.7000) - Metabolic syndrome is a group of risk factors that raises your risk for heart disease and other health problems, such as diabetes and stroke. These risk factors can increase your risk for health problems even if they're only moderately r...
2. `medquad_ans_0000044-11` (score=0.5535) - Metabolic syndrome, also called insulin resistance syndrome, is a group of traits and medical conditions linked to overweight and obesity that puts people at risk for both CVD and type 2 diabetes. Metabolic syndrome is defined\* as the pr...
3. `medquad_ans_0000608-1__dup1` (score=0.5533) - Metabolic syndrome is a group of conditions that put you at risk for heart disease and diabetes. These conditions are - High blood pressure - High blood glucose, or blood sugar, levels - High levels of triglycerides, a type of fat, in yo...
4. `medquad_ans_0000032-6__dup1` (score=0.4459) - Metabolic syndrome is a grouping of traits and medical conditions that puts people at risk for both heart disease and type 2 diabetes. It is defined by the National Cholesterol Education Program as having any three of the following five ...
5. `medquad_ans_0000090-3__dup2` (score=0.3551) - Metabolic syndrome has several causes that act together. You can control some of the causes, such as overweight and obesity, an inactive lifestyle, and insulin resistance. You can't control other factors that may play a role in causing m...

**Answer**:

The symptoms of metabolic syndrome include increased thirst, increased urination (especially at night), fatigue (tiredness), and blurred vision, which are symptoms of high blood sugar, particularly when type 2 diabetes is present. High blood pressure usually has no signs or symptoms, although some people in the early stages may experience dull headaches, dizziness, or more frequent nosebleeds. A large waistline is a visible sign of metabolic syndrome. However, most of the metabolic risk factors have no signs or symptoms [medquad_ans_0000090-5__dup1].

Note: Metabolic syndrome itself does not have specific symptoms; rather, symptoms arise from individual components such as high blood sugar or high blood pressure [medquad_ans_0000044-11].

**Citations detected**: `medquad_ans_0000090-5__dup1`, `medquad_ans_0000044-11`
**Supported citations**: `medquad_ans_0000090-5__dup1`, `medquad_ans_0000044-11`
**Unsupported citations**: _none_

## Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Top docs**:

1. `medquad_ans_0005211-1` (score=0.7000) - What are the signs and symptoms of Pyknoachondrogenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyknoachondrogenesis. If the information is available, the table below includes how often the sym...
2. `medquad_ans_0002152-2` (score=0.1754) - What are the signs and symptoms of Erdheim-Chester disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Erdheim-Chester disease. If the information is available, the table below includes how often t...
3. `medquad_ans_0004625-2` (score=0.1591) - What are the signs and symptoms of Osteomesopyknosis? Osteomesopyknosis may cause chronic, low-grade back pain in the thoracic (middle) and lumbar (lower) regions. It is considered a mild form of osteosclerosis and is usually found in yo...
4. `medquad_ans_0005210-2` (score=0.1376) - What are the signs and symptoms of Pycnodysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pycnodysostosis. If the information is available, the table below includes how often the symptom is se...
5. `medquad_ans_0000309-2__dup1` (score=0.0797) - What are the signs and symptoms of Amyotrophic lateral sclerosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyotrophic lateral sclerosis. If the information is available, the table below includes...

**Answer**:

The symptoms of Pyknoachondrogenesis include abnormal vertebral ossification, abnormality of pelvic girdle bone morphology, abnormality of the mouth, abnormality of the sacrum, depressed nasal ridge, enlarged thorax, increased bone mineral density, low-set, posteriorly rotated ears, macrocephaly, micromelia, palpebral edema, premature birth, short stature, short thorax, thickened nuchal skin fold, and stillbirth. These symptoms are reported in approximately 90% of patients with this condition [medquad_ans_0005211-1].

**Citations detected**: `medquad_ans_0005211-1`
**Supported citations**: `medquad_ans_0005211-1`
**Unsupported citations**: _none_

## Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Top docs**:

1. `medquad_ans_0003965-2` (score=0.7000) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the table below includes h...
2. `medquad_ans_0003966-1` (score=0.3082) - What are the signs and symptoms of Metachromatic leukodystrophy due to saposin B deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy due to saposin B deficiency. If ...
3. `medquad_ans_0003566-1` (score=0.2816) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to defects of the myelin, ...
4. `medquad_ans_0000647-2` (score=0.2377) - Metachromatic leukodystrophy is reported to occur in 1 in 40,000 to 160,000 individuals worldwide. The condition is more common in certain genetically isolated populations: 1 in 75 in a small group of Jews who immigrated to Israel from s...
5. `medquad_ans_0000555-1__dup1` (score=0.2203) - The leukodystrophies are rare diseases that affect the cells of the brain. Specifically, the diseases affect the myelin sheath, the material that surrounds and protects nerve cells. Damage to this sheath slows down or blocks messages bet...

**Answer**:

The symptoms of Metachromatic leukodystrophy include behavioral abnormality, cognitive impairment, decreased nerve conduction velocity, developmental regression, gait disturbance, genu recurvatum, incoordination, muscle weakness, neurological speech impairment, peripheral neuropathy, reduced consciousness/confusion, seizures, amaurosis fugax, hyperreflexia, hypertonia, limitation of joint mobility, muscular hypotonia, nystagmus, optic atrophy, aganglionic megacolon, ataxia, Babinski sign, bulbar palsy, cholecystitis, chorea, delusions, dysarthria, dystonia, EMG: neuropathic changes, emotional lability, gallbladder dysfunction, hallucinations, hyporeflexia, increased CSF protein, intellectual disability, loss of speech, mental deterioration, peripheral demyelination, progressive peripheral neuropathy, spastic tetraplegia, tetraplegia

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Top docs**:

1. `medquad_ans_0000673-4` (score=0.7000) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
2. `medquad_ans_0000095-4` (score=0.5060) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condition.
3. `medquad_ans_0000585-4` (score=0.5060) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condition.
4. `medquad_ans_0000362-4` (score=0.5060) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condition.
5. `medquad_ans_0004097-4` (score=0.3075) - Is mixed connective tissue disease inherited? The role of genetics in the onset of mixed connective tissue disease (MCTD) is still unclear. Some people with MCTD have family members who are also affected by the condition. This suggests t...

**Answer**:

The provided context does not contain information about whether chronic granulomatous disease is inherited. Therefore, based on the given context, it is not possible to determine if chronic granulomatous disease is inherited. [no relevant context]

**Citations detected**: `no relevant context`
**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

## Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Top docs**:

1. `medquad_ans_0000061-4__dup1` (score=0.7000) - Our understanding of central cord syndrome has increased greatly in recent decades as a result of research funded conducted by the National Institute of Neurological Disorders and Stroke (NINDS). Much of this research focuses on finding ...
2. `medquad_ans_0000061-3__dup1` (score=0.6294) - The prognosis for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict he likelihood that neurological...
3. `medquad_ans_0000061-2__dup1` (score=0.4657) - There is no cure for central cord syndrome although some people recover near-normal function. There is no standard course of treatment, although drug therapy, surgery, and rest are often part of the program. Magnetic resonance imaging (M...
4. `medquad_ans_0000061-1__dup1` (score=0.4316) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to and from parts of the...
5. `medquad_ans_0000085-2__dup1` (score=0.2319) - Currently, only central core disease has an effective treatment (see above). There are no known cures for any of these disorders. Supportive treatment may involve orthopedic treatments, as well as physical, occupational or speech therapy.

**Answer**:

The outlook for central cord syndrome is generally favorable, with most people experiencing some recovery of neurological function, particularly after trauma. Many individuals recover substantial function following their initial injury, and the ability to walk is regained in most cases, although some impairment may remain. Recovery typically begins in the legs, followed by the bladder, and hand function recovers last, if at all. Improvement is generally better in younger patients compared to those over the age of 50. Early medical intervention is associated with good outcomes, and evaluation of abnormal signals on MRI images can help predict the likelihood of natural neurological recovery. While there is no cure, early surgery may improve recovery chances, and surgery is beneficial in cases of persistent spinal cord compression and ongoing neurological deterioration [medquad_ans_0000061-3__dup1, medquad_ans_0000061-2__dup1].

**Citations detected**: `medquad_ans_0000061-3__dup1, medquad_ans_0000061-2__dup1`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000061-3__dup1, medquad_ans_0000061-2__dup1`

## Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Top docs**:

1. `medquad_ans_0000094-13` (score=0.7000) - - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no apparent cause. - Th...
2. `medquad_ans_0000094-1__dup6` (score=0.6929) - Cyclic vomiting syndrome, sometimes referred to as CVS, is an increasingly recognized disorder with sudden, repeated attacksalso called episodesof severe nausea, vomiting, and physical exhaustion that occur with no apparent cause. The ep...
3. `medquad_ans_0000094-3__dup3` (score=0.6199) - The cause of cyclic vomiting syndrome is unknown. However, some experts believe that some possible problems with bodily functions may contribute to the cause, such as the following: - gastrointestinal motilitythe way food moves through t...
4. `medquad_ans_0000094-4__dup3` (score=0.6019) - Cyclic vomiting syndrome is more common in children than adults, although reports of the syndrome in adults have increased in recent years.1 Usually, children are about 5 years old when diagnosed with cyclic vomiting syndrome, which occu...
5. `medquad_ans_0000094-6__dup1` (score=0.5283) - The main symptoms of cyclic vomiting syndrome are severe nausea and sudden vomiting lasting hours to days. A person may also experience one or more of the following symptoms: - retching, or making an attempt to vomit - heaving or gagging...

**Answer**:

The cause of cyclic vomiting syndrome is unknown. However, some experts believe that possible problems with bodily functions may contribute to the cause, such as gastrointestinal motility, central nervous system function, autonomic nervous system function, and hormone imbalances [medquad_ans_0000094-3__dup3].

**Citations detected**: `medquad_ans_0000094-3__dup3`
**Supported citations**: `medquad_ans_0000094-3__dup3`
**Unsupported citations**: _none_

## Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Top docs**:

1. `medquad_ans_0000180-1__dup2` (score=0.7000) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not always easy to know when...
2. `medquad_ans_0000603-1__dup1` (score=0.4389) - Summary : Mental health includes our emotional, psychological, and social well-being. It affects how we think, feel and act as we cope with life. It also helps determine how we handle stress, relate to others, and make choices. Mental he...
3. `medquad_ans_0000189-1__dup2` (score=0.2749) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise and insuring their ...
4. `medquad_ans_0000776-5` (score=0.2412) - These resources address the diagnosis or management of Partington syndrome: - American Academy of Child and Adolescent Psychiatry: Services in School for Children with Special Needs - American Academy of Pediatrics: What is a Development...
5. `medquad_ans_0000176-1__dup2` (score=0.2383) - All kids misbehave some times. And some may have temporary behavior problems due to stress. For example, the birth of a sibling, a divorce, or a death in the family may cause a child to act out. Behavior disorders are more serious. They ...

**Answer**:

Yes, there is information about child mental health. Early recognition and treatment of mental illnesses in children are important because once mental illness develops, it becomes a regular part of the child's behavior and is more difficult to treat [medquad_ans_0000180-1__dup2]. Warning signs that may indicate a more serious problem include problems in more than one setting (at school, at home, with peers), changes in appetite or sleep, social withdrawal or fear of things previously not feared, returning to younger behaviors like bedwetting, signs of being upset such as sadness or tearfulness, self-destructive behaviors like head-banging or frequent injuries, and repeated thoughts of death [medquad_ans_0000180-1__dup2]. Diagnosis involves evaluating the child's signs and symptoms, medical history, and family history [medquad_ans_0000180-1__dup2].

**Citations detected**: `medquad_ans_0000180-1__dup2`
**Supported citations**: `medquad_ans_0000180-1__dup2`
**Unsupported citations**: _none_

## Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Top docs**:

1. `medquad_ans_0000673-2__dup1` (score=0.7000) - What are the signs and symptoms of Behcet's disease? Symptoms of Behcet's disease include recurrent ulcers in the mouth (resembling canker sores) and on the genitals, and eye inflammation (uveitis). The disorder may also cause various ty...
2. `medquad_ans_0000003-2__dup3` (score=0.5279) - Signs and symptoms of Cushing's syndrome vary, but most people with the disorder have upper body obesity, a rounded face, increased fat around the neck, and relatively slender arms and legs. Children tend to be obese with slowed growth r...
3. `medquad_ans_0002066-2` (score=0.4981) - What are the signs and symptoms of Eisenmenger syndrome? Symptoms of Eisenmenger include shortness of breath, chest pain, feeling tired or dizzy, fainting, abnormal heart rhythm (arrhythmia), stroke, coughing up blood, swelling of joints...
4. `medquad_ans_0004451-2` (score=0.4108) - What are the signs and symptoms of Noonan syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome 1. If the information is available, the table below includes how often the symptom i...
5. `medquad_ans_0000757-2__dup1` (score=0.3042) - What are the signs and symptoms of Blepharophimosis, ptosis, and epicanthus inversus syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Blepharophimosis, ptosis, and epicanthus inversus sy...

**Answer**:

The provided context does not contain any information about the symptoms of Cowden syndrome. Therefore, a response cannot be generated based on the given context. [no context]

**Citations detected**: `no context`
**Supported citations**: _none_
**Unsupported citations**: `no context`

## Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Top docs**:

1. `medquad_ans_0000395-5` (score=0.7000) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscular Dystrophy - Geneti...
2. `medquad_ans_0000395-2` (score=0.5127) - Fukuyama congenital muscular dystrophy is seen almost exclusively in Japan, where it is the second most common form of childhood muscular dystrophy (after Duchenne muscular dystrophy). Fukuyama congenital muscular dystrophy has an estima...
3. `medquad_ans_0000395-1` (score=0.2942) - Fukuyama congenital muscular dystrophy is an inherited condition that predominantly affects the muscles, brain, and eyes. Congenital muscular dystrophies are a group of genetic conditions that cause muscle weakness and wasting (atrophy) ...
4. `medquad_ans_0004219-1` (score=0.2813) - Muscular dystrophy (MD) refers to a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or childhood, while oth...
5. `medquad_ans_0000631-1__dup1` (score=0.2411) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. The different types ...

**Answer**:

The provided context does not specify treatments for Fukuyama congenital muscular dystrophy. While it mentions resources related to diagnosis and management, such as genetic counseling, drug therapy, surgery and rehabilitation, palliative care, and diagnostic tests, it does not detail specific treatments for this condition [medquad_ans_0000395-5]. Therefore, based on the available information, the exact treatments are not outlined.

**Citations detected**: `medquad_ans_0000395-5`
**Supported citations**: `medquad_ans_0000395-5`
**Unsupported citations**: _none_

## Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Top docs**:

1. `medquad_ans_0000004_7-2` (score=0.7000) - Signs of childhood non-Hodgkin lymphoma include breathing problems and swollen lymph nodes. These and other signs may be caused by childhood non-Hodgkin lymphoma or by other conditions. Check with a doctor if your child has any of the fo...
2. `medquad_ans_0000004_6-1` (score=0.5007) - Key Points - Childhood Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two types of childhood Hodgkin lymphoma. - Epstein-Barr virus infection increases the risk of childhood Hodgkin ...
3. `medquad_ans_0000483-1__dup1` (score=0.4341) - Hodgkin disease is a type of lymphoma. Lymphoma is a cancer of a part of the immune system called the lymph system. The first sign of Hodgkin disease is often an enlarged lymph node. The disease can spread to nearby lymph nodes. Later it...
4. `medquad_ans_0000004_6-3` (score=0.4324) - Signs of childhood Hodgkin lymphoma include swollen lymph nodes, fever, night sweats, and weight loss. These and other signs and symptoms may be caused by childhood Hodgkin lymphoma or by other conditions. Check with your child's doctor ...
5. `medquad_ans_0000004_7-1` (score=0.3486) - Key Points - Childhood non-Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - The main types of lymphoma are Hodgkin lymphoma and non-Hodgkin lymphoma. - There are three major types of childhood n...

**Answer**:

Tests that examine the lymph system are used to detect (find) and diagnose childhood Hodgkin lymphoma [medquad_ans_0000004_6-1].

**Citations detected**: `medquad_ans_0000004_6-1`
**Supported citations**: `medquad_ans_0000004_6-1`
**Unsupported citations**: _none_

## Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Top docs**:

1. `medquad_ans_0000117-4` (score=0.7000) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one affected parent. A ...
2. `medquad_ans_0006446-4` (score=0.5752) - Is Williams syndrome inherited?
3. `medquad_ans_0000673-4` (score=0.2540) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
4. `medquad_ans_0000786-4` (score=0.1741) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one affected parent. Ho...
5. `medquad_ans_0000149-4` (score=0.1404) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one affected parent. Ot...

**Answer**:

The provided context does not contain information about KBG syndrome or its inheritance pattern. Therefore, it is not possible to answer the question based on the given text. [no relevant context]

**Citations detected**: `no relevant context`
**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

## Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Top docs**:

1. `medquad_ans_0000194-1__dup1` (score=0.7000) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and phy...
2. `medquad_ans_0000193-1__dup1` (score=0.7000) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and phy...
3. `medquad_ans_0000195-1__dup1` (score=0.7000) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and phy...
4. `medquad_ans_0000946-1__dup1` (score=0.1585) - Cardiofaciocutaneous (CFC) syndrome is a disorder that affects many parts of the body, particularly the heart (cardio-), face (facio-), and the skin and hair (cutaneous). People with this condition also have developmental delay and intel...
5. `medquad_ans_0000674-1__dup1` (score=0.0976) - Behr syndrome is a disorder mainly characterized by early-onset optic atrophy, ataxia, and spasticity. Other signs and symptoms may be present and vary from person to person. Although the exact cause is unknown, the syndrome is believed ...

**Answer**:

Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and physical disability. Additional symptoms may include epilepsy, painful, itchy skin lesion (chilblains), vision problems, and joint stiffness. Symptoms usually progress over several months before the disease course stabilizes. There are six different types of Aicardi-Goutieres syndrome, which are distinguished by the gene that causes the condition: TREX1, RNASEH2A, RNASEH2B, RNASEH2C, SAMHD1, and ADAR genes. Most cases are inherited in an autosomal recessive pattern, although rare autosomal dominant cases have been reported. Treatment is symptomatic and supportive [medquad_ans_0000194-1__dup1].

**Citations detected**: `medquad_ans_0000194-1__dup1`
**Supported citations**: `medquad_ans_0000194-1__dup1`
**Unsupported citations**: _none_

## Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Top docs**:

1. `medquad_ans_0000026_1-7` (score=0.7000) - Key Points - There are different types of treatment for patients with renal cell cancer. - Five types of standard treatment are used: - Surgery - Radiation therapy - Chemotherapy - Biologic therapy - Targeted therapy - New types of treat...
2. `medquad_ans_0000026_2-7` (score=0.5645) - Key Points - There are different types of treatment for patients with transitional cell cancer of the renal pelvis and ureter. - One type of standard treatment is used: - Surgery - New types of treatment are being tested in clinical tria...
3. `medquad_ans_0001354-2` (score=0.5215) - What treatments for metastatic clear cell renal cell carcinoma are available in North America? There are several treatments for metastatic clear cell renal cell carcinoma available in North America. IL-2 and sunitinib - as well as the me...
4. `medquad_ans_0000026_1-1` (score=0.3880) - Key Points - Renal cell cancer is a disease in which malignant (cancer) cells form in tubules of the kidney. - Smoking and misuse of certain pain medicines can affect the risk of renal cell cancer. - Signs of renal cell cancer include bl...
5. `medquad_ans_0000026_3-7` (score=0.3827) - Key Points - There are different types of treatment for patients with Wilms tumor and other childhood kidney tumors. - Children with Wilms tumor or other childhood kidney tumors should have their treatment planned by a team of health car...

**Answer**:

Research involving clinical trials is being conducted for renal cell cancer to improve current treatments or obtain information on new treatments. These clinical trials are designed to test new therapies and may be open to patients before, during, or after starting their cancer treatment. When clinical trials demonstrate that a new treatment is more effective than the standard treatment, it may become the new standard of care [medquad_ans_0000026_1-7].

**Citations detected**: `medquad_ans_0000026_1-7`
**Supported citations**: `medquad_ans_0000026_1-7`
**Unsupported citations**: _none_

## Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Top docs**:

1. `medquad_ans_0000190-6` (score=0.7000) - Treatment for primary biliary cirrhosis depends on how early a health care provider diagnoses the disease and whether complications are present. In the early stages of primary biliary cirrhosis, treatment can slow the progression of live...
2. `medquad_ans_0000190-9` (score=0.5012) - - Primary biliary cirrhosis is a chronic disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. - When chronic inflammation damages the bile ducts, bile and toxic wastes build up in...
3. `medquad_ans_0000183-10` (score=0.4712) - - Biliary atresia is a life-threatening condition in infants in which the bile ducts inside or outside the liver do not have normal openings. - The first symptom of biliary atresia is jaundicewhen the skin and whites of the eyes turn yel...
4. `medquad_ans_0000193-7` (score=0.3760) - To prevent cirrhosis, - see your doctor for treatment of your liver disease. Many of the causes of cirrhosis are treatable. Early treatment may prevent cirrhosis. - try to keep your weight in the normal range. Being overweight can make s...
5. `medquad_ans_0000190-2__dup2` (score=0.3263) - The causes of primary biliary cirrhosis are unknown. Most research suggests it is an autoimmune disease. The immune system protects people from infection by identifying and destroying bacteria, viruses, and other potentially harmful fore...

**Answer**:

Treatment for primary biliary cirrhosis depends on the stage of the disease and the presence of complications. In the early stages, treatment can slow the progression of liver damage to cirrhosis. The primary medication prescribed is ursodiol (Actigall, Urso), a nontoxic bile acid that replaces more toxic bile acids produced by the liver, reducing levels of bilirubin and liver enzymes in the blood. Early treatment with ursodiol reduces the likelihood of needing a liver transplant and improves survival, although it does not cure the disease. Treatment benefits are greatest when started early, but ursodiol can still slow disease progression even when used later.

Patients should avoid alcohol and illegal substances, as these can worsen liver damage. They should also avoid complementary and alternative medications, such as herbs, and should consult a healthcare provider before taking any new prescription, over-the-counter medications, or vitamins, as many can affect liver

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=0.7000) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People who have seizure dis...
2. `medquad_ans_0000967-1__dup1` (score=0.4609) - Summary : Playing in the water - whether swimming, boating or diving - can be fun. It can also be dangerous, especially for children. Being safe can help prevent injuries and drowning. To stay safe in the water - Avoid alcohol when swimm...
3. `medquad_ans_0000378-2` (score=0.2727) - Floating-Harbor syndrome is a rare disorder; only about 50 cases have been reported in the medical literature.
4. `medquad_ans_0000378-4` (score=0.2694) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of Floating-Harbor syndrome result from new mutations in the gene and occ...
5. `medquad_ans_0000916-1__dup2` (score=0.1526) - Summary : A tsunami is a series of huge ocean waves created by an underwater disturbance. Causes include earthquakes, landslides, volcanic eruptions, or meteorites--chunks of rock from space that strike the surface of Earth. A tsunami ca...

**Answer**:

Drowning occurs when people get too much water in their lungs. It can happen quickly and silently, and even in as little as an inch or two of water. Babies can drown in a sink or bathtub, and preschoolers are most likely to drown in a swimming pool. People with seizure disorders are also at risk in the water [medquad_ans_0000299-1__dup2]. Drowning is the most common cause of death related to a tsunami [medquad_ans_0000916-1__dup2].

**Citations detected**: `medquad_ans_0000299-1__dup2`, `medquad_ans_0000916-1__dup2`
**Supported citations**: `medquad_ans_0000299-1__dup2`, `medquad_ans_0000916-1__dup2`
**Unsupported citations**: _none_

## Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Top docs**:

1. `medquad_ans_0002058-1` (score=0.7000) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragile and bruises easil...
2. `medquad_ans_0002058-5` (score=0.5290) - How is Ehlers-Danlos syndrome, kyphoscoliosis type diagnosed? A diagnosis of Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is typically based on the presence of characteristic signs and symptoms. The following tests may then be recom...
3. `medquad_ans_0002058-3` (score=0.4286) - What causes Ehlers-Danlos syndrome, kyphoscoliosis type? Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is caused by changes (mutations) in the PLOD1 gene. This gene encodes an enzyme that helps process molecules which allow collagen ...
4. `medquad_ans_0000313-1` (score=0.4101) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and symptoms of Ehlers-Danlos...
5. `medquad_ans_0000313-4` (score=0.3904) - The inheritance pattern of Ehlers-Danlos syndrome varies by type. The arthrochalasia, classical, hypermobility, and vascular forms of the disorder have an autosomal dominant pattern of inheritance. Autosomal dominant inheritance means th...

**Answer**:

Ehlers-Danlos syndrome, kyphoscoliosis type is an inherited connective tissue disorder caused by mutations in the PLOD1 gene, which encodes an enzyme essential for collagen formation and stability. This disorder leads to defective collagen networks, resulting in weakened connective tissues and characteristic signs and symptoms such as hyperextensible skin that bruises easily, joint hypermobility, severe hypotonia at birth, progressive kyphoscoliosis (kyphosis and scoliosis), and scleral fragility. It is inherited in an autosomal recessive manner and is diagnosed based on clinical signs and symptoms, with confirmation through urine tests, skin biopsy to detect enzyme deficiencies, or genetic testing for PLOD1 gene mutations [medquad_ans_0002058-1, medquad_ans_0002058-5, medquad_ans_0002058-3].

**Citations detected**: `medquad_ans_0002058-1, medquad_ans_0002058-5, medquad_ans_0002058-3`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0002058-1, medquad_ans_0002058-5, medquad_ans_0002058-3`

## Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Top docs**:

1. `medquad_ans_0001860-2` (score=0.7000) - What are the signs and symptoms of Dihydropteridine reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropteridine reductase deficiency. If the information is available, the tabl...
2. `medquad_ans_0001862-1` (score=0.4056) - Dihydropyrimidine dehydrogenase (DPD) deficiency is a condition in which the body cannot break down the nucleotides thymine and uracil. DPD deficiency can have a wide range of severity; some individuals may have various neurological prob...
3. `medquad_ans_0006567-2` (score=0.2838) - What are the signs and symptoms of 2,4-Dienoyl-CoA reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for 2,4-Dienoyl-CoA reductase deficiency. If the information is available, the table ...
4. `medquad_ans_0000288-2` (score=0.2535) - Severe dihydropyrimidine dehydrogenase deficiency, with its early-onset neurological symptoms, is a rare disorder. Its prevalence is unknown. However, between 2 and 8 percent of the general population may be vulnerable to toxic reactions...
5. `medquad_ans_0001860-1` (score=0.2259) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH4). Tetrahydrobiopte...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include cognitive impairment, microcephaly, cerebral calcification, choreoathetosis, dysphagia, dystonia, episodic fever, excessive salivation, hyperphenylalaninemia, hypertonia, infantile onset, intellectual disability, irritability, muscular hypotonia, myoclonus, progressive neurologic deterioration, seizures, tremor, and variable expressivity. These symptoms occur in approximately 90% of patients for cognitive impairment and microcephaly [medquad_ans_0001860-2].

**Citations detected**: `medquad_ans_0001860-2`
**Supported citations**: `medquad_ans_0001860-2`
**Unsupported citations**: _none_

## Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Top docs**:

1. `medquad_ans_0000669-2` (score=0.7000) - The prevalence of monilethrix is unknown.
2. `medquad_ans_0000484-2` (score=0.2579) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
3. `medquad_ans_0000770-2` (score=0.2399) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.
4. `medquad_ans_0000989-2` (score=0.2066) - This condition affects an estimated 1 in 50,000 people.
5. `medquad_ans_0000195-2` (score=0.1988) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.

**Answer**:

The prevalence of monilethrix is unknown [medquad_ans_0000669-2]. However, estimates suggest it affects 1 in 87,000 people [medquad_ans_0000484-2], 1 to 3 per million people worldwide [medquad_ans_0000770-2], 1 in 50,000 people [medquad_ans_0000989-2], and 1 in 40,000 to 50,000 people [medquad_ans_0000195-2]. Due to the variability in estimates and the lack of a definitive prevalence figure, the exact number of people affected remains uncertain.

**Citations detected**: `medquad_ans_0000669-2`, `medquad_ans_0000484-2`, `medquad_ans_0000770-2`, `medquad_ans_0000989-2`, `medquad_ans_0000195-2`
**Supported citations**: `medquad_ans_0000669-2`, `medquad_ans_0000484-2`, `medquad_ans_0000770-2`, `medquad_ans_0000989-2`, `medquad_ans_0000195-2`
**Unsupported citations**: _none_

## Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.7000) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the results of newborn s...
2. `medquad_ans_0002733-2` (score=0.3509) - What are the signs and symptoms of Hartnup disease? The signs and symptoms of Hartnup disease may vary and include the following: Skin findings: sensitivity to sunlight Neurologic symptoms: ataxia, spasticity, headaches,and hypotonia Psy...
3. `medquad_ans_0002152-1` (score=0.1049) - Erdheim-Chester disease is a rare condition that can affect many different organs of the body. This condition, which usually affects adults, is characterized by excessive production and accumulation of histiocytes (specific cells that no...
4. `medquad_ans_0000372-2` (score=0.0964) - Feingold syndrome appears to be a rare condition, although its exact prevalence is unknown.
5. `medquad_ans_0000708-2` (score=0.0692) - The prevalence of this condition is unknown.

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system, making it a type of aminoaciduria. It is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. The condition may be diagnosed through newborn screening tests. Most individuals are asymptomatic, but those who do show symptoms typically experience onset between ages 3 and 9, although it can occasionally present in adulthood. Signs and symptoms include skin photosensitivity, neurologic findings (such as ataxia, spasticity, headaches, and hypotonia), psychiatric symptoms (like anxiety, emotional instability, and mood changes), and ocular findings (such as double vision, nystagmus, strabismus, and photophobia). Symptoms may be triggered by sunlight exposure, fever, drugs, or emotional or physical stress and often last 1–4 weeks before spontaneous

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Top docs**:

1. `medquad_ans_0000427-5` (score=0.7000) - These resources address the diagnosis or management of glycogen storage disease type VI: - Gene Review: Gene Review: Glycogen Storage Disease Type VI - Genetic Testing Registry: Glycogen storage disease, type VI These resources from Medl...
2. `medquad_ans_0000423-5` (score=0.6648) - These resources address the diagnosis or management of glycogen storage disease type III: - Gene Review: Gene Review: Glycogen Storage Disease Type III - Genetic Testing Registry: Glycogen storage disease type III These resources from Me...
3. `medquad_ans_0002622-1` (score=0.6378) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepatomegaly), or an inc...
4. `medquad_ans_0000428-5` (score=0.5767) - These resources address the diagnosis or management of glycogen storage disease type VII: - Genetic Testing Registry: Glycogen storage disease, type VII - The Swedish Information Centre for Rare Diseases These resources from MedlinePlus ...
5. `medquad_ans_0002615-3` (score=0.3976) - What causes glycogen storage disease type 13? Glycogen storage disease type 13 (GSD13) is caused by changes (mutations) in the ENO3 gene. Glycogen is a substance that is stored in muscle tissue and is used as an important source of energ...

**Answer**:

The provided context does not specify the genetic changes related to glycogen storage disease type VII. [medquad_ans_0000428-5] mentions resources for diagnosis and management of glycogen storage disease type VII, including the Genetic Testing Registry, but does not detail the underlying genetic mutations or changes associated with this condition. Therefore, based on the given information, the genetic changes related to glycogen storage disease type VII cannot be determined.

**Citations detected**: `medquad_ans_0000428-5`
**Supported citations**: `medquad_ans_0000428-5`
**Unsupported citations**: _none_

## Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Top docs**:

1. `medquad_ans_0000977-2` (score=0.7000) - Thiamine-responsive megaloblastic anemia syndrome has been reported in approximately 30 families worldwide. Its prevalence is unknown.
2. `medquad_ans_0006061-1` (score=0.6054) - Thiamine-responsive megaloblastic anemia syndrome is a very rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Affected individuals begin to show symptoms of this condition between i...
3. `medquad_ans_0000977-5` (score=0.5385) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megaloblastic anemia, thi...
4. `medquad_ans_0000977-1` (score=0.3959) - Thiamine-responsive megaloblastic anemia syndrome is a rare condition characterized by hearing loss, diabetes, and a blood disorder called megaloblastic anemia. Megaloblastic anemia occurs when a person has a low number of red blood cell...
5. `medquad_ans_0006061-2` (score=0.3610) - What are the signs and symptoms of Thiamine responsive megaloblastic anemia syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Thiamine responsive megaloblastic anemia syndrome. If the informatio...

**Answer**:

The treatment for thiamine-responsive megaloblastic anemia syndrome is high doses of vitamin B1 (thiamine), which can treat the anemia associated with the condition [medquad_ans_0006061-1]. However, it remains unclear whether thiamine treatment can improve hearing loss or prevent diabetes, although individuals with the syndrome usually require insulin to manage their diabetes [medquad_ans_0000977-1]. Additional management may include genetic counseling, drug therapy, and palliative care as part of a comprehensive approach [medquad_ans_0000977-5].

**Citations detected**: `medquad_ans_0006061-1`, `medquad_ans_0000977-1`, `medquad_ans_0000977-5`
**Supported citations**: `medquad_ans_0006061-1`, `medquad_ans_0000977-1`, `medquad_ans_0000977-5`
**Unsupported citations**: _none_

## Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Top docs**:

1. `medquad_ans_0000587-4__dup1` (score=0.7000) - Is there a cure or treatment for autosomal recessive polycystic kidney disease? Although a cure or treatment for the underlying genetic cause of autosomal recessive polycystic kidney disease does not exist, advancements have been made in...
2. `medquad_ans_0000804-5` (score=0.5244) - These resources address the diagnosis or management of polycystic kidney disease: - Gene Review: Gene Review: Polycystic Kidney Disease, Autosomal Dominant - Gene Review: Gene Review: Polycystic Kidney Disease, Autosomal Recessive - Gene...
3. `medquad_ans_0004939-2` (score=0.3974) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their ability to filter waste pr...
4. `medquad_ans_0000804-1` (score=0.3698) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the blood. The growth o...
5. `medquad_ans_0004939-1` (score=0.3560) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the their ability to fil...

**Answer**:

Medical management of polycystic kidney disease is currently symptomatic and involves supportive care. This includes treatment of high blood pressure with medication, management of urinary tract infections with antibiotics, and treatment of low sodium levels (hyponatremia) with diuresis and/or sodium supplementation. When kidneys are severely enlarged, nephrectomy (removal of one or both kidneys) may be performed. Dialysis may be required during the first days of life if an infant produces little or no urine (oliguria or anuria). Kidney failure may require dialysis or kidney transplantation. Growth failure may be managed with gastrostomy tubes or growth hormone therapy. Mechanical ventilation may be used to treat breathing issues due to enlarged kidneys. For liver involvement, a shunt may be required to manage progressive high blood pressure, and liver transplantation may be considered [medquad_ans_0000587-4__dup1]. Resources for diagnosis and

**Citations detected**: `medquad_ans_0000587-4__dup1`
**Supported citations**: `medquad_ans_0000587-4__dup1`
**Unsupported citations**: _none_

## Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Top docs**:

1. `medquad_ans_0000061-5__dup2` (score=0.7000) - Scientists are making rapid progress in understanding the complexities of rheumatoid arthritis. They are learning more about how and why it develops and why some people have more severe symptoms than others. Research efforts are focused ...
2. `medquad_ans_0000061-14` (score=0.5906) - Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation. Still others, called disease-modifying anti-rheumatic drugs or DMARDs, can often slow the course of the dise...
3. `medquad_ans_0000061-4__dup3` (score=0.5852) - Most Symptoms Are Treatable Doctors use a variety of approaches to treat rheumatoid arthritis. The goals of treatment are to help relieve pain, reduce swelling, slow down or help prevent joint damage, increase the ability to function, an...
4. `medquad_ans_0000061-13` (score=0.5707) - Medication, exercise, and, in some cases, surgery are common treatments for this disease. Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation. People with rheuma...
5. `medquad_ans_0000061-15` (score=0.4964) - Rest. People with rheumatoid arthritis need a good balance between rest and exercise; they should rest more when the disease is active and exercise more when it is not. Rest helps to reduce active joint inflammation and pain and to fight...

**Answer**:

Research on rheumatoid arthritis is focused on understanding how and why the disease develops, particularly identifying possible triggers such as infectious agents like viruses and bacteria that may contribute to the disease in individuals with a genetic predisposition [medquad_ans_0000061-5__dup2]. Scientists are also investigating the complex relationships between the hormonal, nervous, and immune systems to explain why more women than men develop the disease, including how changes in steroid hormones like estrogen and testosterone may influence disease development or flares [medquad_ans_0000061-5__dup2]. Additionally, researchers are exploring why rheumatoid arthritis often improves during pregnancy.

Efforts are underway to develop new drugs that can reduce inflammation and slow or stop the disease with fewer side effects. This includes research into disease-modifying anti-rheumatic drugs (DMARDs) such as methotrexate, leflunomide, sulfas

**Citations detected**: `medquad_ans_0000061-5__dup2`
**Supported citations**: `medquad_ans_0000061-5__dup2`
**Unsupported citations**: _none_

## Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Top docs**:

1. `medquad_ans_0002561-4` (score=0.7000) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (anterior) part of the ...
2. `medquad_ans_0002561-2` (score=0.4983) - What are the signs and symptoms of Glass-Chapman-Hockley syndrome? Glass-Chapman-Hockley syndrome has only been described in one family with five affected family members in three generations. The signs and symptoms seen in the five affec...
3. `medquad_ans_0002561-1` (score=0.4235) - The Glass-Chapman-Hockley syndrome is a very rare disease. To date, the syndrome has only been reported in one family with five members affected in three generations. The first patients were two brothers that had an abnormally-shaped hea...
4. `medquad_ans_0000860-2__dup1` (score=0.2325) - How might Brown-Sequard syndrome be treated?
5. `medquad_ans_0000883-5__dup1` (score=0.1748) - How might Buschke Ollendorff syndrome be treated? There is currently no cure for BOS. Surgical removal of lesions on or under the skin may be done for cosmetic purposes. In some patients, surgical treatment of deafness may be possible. S...

**Answer**:

The treatment for Glass-Chapman-Hockley syndrome is typically surgery, which is based on the person's specific signs and symptoms. The goal of the surgery is to increase the space in the front (anterior) part of the skull. The operation is usually performed when the person is between 9 to 12 months of age. If other sutures, other than the coronal suture, are involved, other surgeries may be performed [medquad_ans_0002561-4].

**Citations detected**: `medquad_ans_0002561-4`
**Supported citations**: `medquad_ans_0002561-4`
**Unsupported citations**: _none_

## Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Top docs**:

1. `medquad_ans_0000868-2` (score=0.7000) - Renpenning syndrome is a rare disorder; its prevalence is unknown. More than 60 affected individuals in at least 15 families have been identified.
2. `medquad_ans_0000247-2__dup2` (score=0.1777) - There is no cure for Rett syndrome. Treatment for the disorder is symptomatic, focusing on the management of symptoms, and supportive. Medication may be needed for breathing irregularities and motor difficulties, and antiepileptic drugs ...
3. `medquad_ans_0003615-3` (score=0.1703) - How might CREST syndrome be treated? Unfortunately, CREST syndrome has no known cure. The condition carries both physical and psychological consequences, so a holistic approach to management should be taken. Treatment generally focuses o...
4. `medquad_ans_0000146-2__dup1` (score=0.1588) - There are no standard treatments for hereditary neuropathies. Treatment is mainly symptomatic and supportive. Medical treatment includes physical therapy and if needed, pain medication. Orthopedic surgery may be needed to correct severe ...
5. `medquad_ans_0005119-5` (score=0.1311) - How might Proud syndrome be treated? The treatment of Proud syndrome is based on the signs and symptoms present in each person. For example, spasticity may be treated with a variety of therapies including medications and/or physical ther...

**Answer**:

The provided context does not contain information about the treatments for Renpenning syndrome. [medquad_ans_0000868-2] mentions that Renpenning syndrome is a rare disorder with an unknown prevalence and that over 60 affected individuals in at least 15 families have been identified, but it does not specify any treatments. Therefore, based on the available context, the treatments for Renpenning syndrome cannot be determined.

**Citations detected**: `medquad_ans_0000868-2`
**Supported citations**: `medquad_ans_0000868-2`
**Unsupported citations**: _none_

## Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Top docs**:

1. `medquad_ans_0001105-1` (score=0.7000) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is available, the table be...
2. `medquad_ans_0001100-1__dup1` (score=0.6742) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is available, the table ...
3. `medquad_ans_0001111-1` (score=0.6363) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2N? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2N. If the information is available, the table be...
4. `medquad_ans_0001112-1` (score=0.6089) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2O? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2O. If the information is available, the table be...
5. `medquad_ans_0001110-1` (score=0.5706) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is available, the table be...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 2G include onion bulb formation (7.5%), areflexia, autosomal recessive inheritance, axonal degeneration/regeneration, decreased motor nerve conduction velocity, decreased number of peripheral myelinated nerve fibers, distal amyotrophy, distal muscle weakness, distal sensory impairment, flexion contracture, neonatal onset, pes cavus, spinal deformities, split hand, and vocal cord paresis [medquad_ans_0001105-1].

**Citations detected**: `medquad_ans_0001105-1`
**Supported citations**: `medquad_ans_0001105-1`
**Unsupported citations**: _none_

## Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Top docs**:

1. `medquad_ans_0001393-2` (score=0.7000) - What signs and symptoms are associated with coccygodynia? The classic symptom is pain when pressure is applied to the tailbone, such as when sitting on a hard chair. Symptoms usually improve with relief of pressure when standing or walki...
2. `medquad_ans_0000676-2__dup1` (score=0.2511) - What are the symptoms of Bell's palsy?
3. `medquad_ans_0003881-2` (score=0.1876) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speech Drooling
4. `medquad_ans_0001393-1` (score=0.1661) - Coccygodynia is a rare condition in that causes pain in and around the coccyx (tailbone). Although various causes have been described for the condition, the more common causes are direct falls and injury.
5. `medquad_ans_0000037-2__dup1` (score=0.1030) - What are the signs and symptoms of achalasia? Most people with achalasia experience difficulty swallowing, also known as dysphagia and heartburn. Other symptoms might include: regurgitation or vomiting, noncardiac chest pain, odynophagia...

**Answer**:

The symptoms of coccygodynia include pain when pressure is applied to the tailbone, such as when sitting on a hard chair, which usually improves with relief of pressure when standing or walking. Other symptoms may include immediate and severe pain when moving from sitting to standing, pain during bowel movements, pain during sex, and a deep ache in the region of the tailbone [medquad_ans_0001393-2].

**Citations detected**: `medquad_ans_0001393-2`
**Supported citations**: `medquad_ans_0001393-2`
**Unsupported citations**: _none_

## Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Top docs**:

1. `medquad_ans_0004025-1` (score=0.7000) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgenesis and cleft lip-p...
2. `medquad_ans_0003783-2` (score=0.3545) - What are the signs and symptoms of Mandibulofacial dysostosis with microcephaly? Mandibulofacial dysostosis with microcephaly (MFDM) may affect multiple parts of the body but primarily affects the head and face. People with MFDM are usua...
3. `medquad_ans_0003270-2` (score=0.2882) - What are the signs and symptoms of Isolated corpus callosum agenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Isolated corpus callosum agenesis. If the information is available, the table below ...
4. `medquad_ans_0002446-2` (score=0.1948) - What are the signs and symptoms of Frontonasal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontonasal dysplasia. If the information is available, the table below includes how often the s...
5. `medquad_ans_0002040-1` (score=0.1420) - What are the signs and symptoms of Ectrodactyly and ectodermal dysplasia without cleft lip/palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Ectrodactyly and ectodermal dysplasia without cleft lip...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, and microcephaly. These symptoms are listed in the Human Phenotype Ontology as part of the signs and symptoms associated with this condition [medquad_ans_0004025-1]. Additionally, other symptoms such as preaxial hand polydactyly, proptosis, and seizures may also be present, though their frequency varies and is not always specified [medquad_ans_0004025-1].

**Citations detected**: `medquad_ans_0004025-1`
**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

## Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Top docs**:

1. `medquad_ans_0000562-5` (score=0.7000) - These resources address the diagnosis or management of Klippel-Feil syndrome: - Genetic Testing Registry: Klippel Feil syndrome - Genetic Testing Registry: Klippel-Feil syndrome 1, autosomal dominant - Genetic Testing Registry: Klippel-F...
2. `medquad_ans_0000170-3__dup1` (score=0.6347) - The prognosis for most individuals with Klippel-Feil Syndrome is good if the disorder is treated early and appropriately. Activities that can injure the neck should be avoided.
3. `medquad_ans_0000170-2__dup1` (score=0.5948) - Treatment for Klippel-Feil Syndrome is symptomatic and may include surgery to relieve cervical or craniocervical instability and constriction of the spinal cord, and to correct scoliosis. Physical therapy may also be useful.
4. `medquad_ans_0003402-2` (score=0.5913) - What are the signs and symptoms of Klippel Feil syndrome? Klippel Feil syndrome is characterized by the fusion of 2 or more spinal bones in the neck (cervical vertebrae). The condition is present from birth (congenital). The 3 most commo...
5. `medquad_ans_0003402-5` (score=0.5615) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal a...

**Answer**:

The provided context does not contain specific information on how to diagnose Klippel-Feil syndrome. While it mentions resources such as the Genetic Testing Registry and MedlinePlus that offer information on diagnosis and management, no detailed diagnostic methods or criteria are described in the given text. [medquad_ans_0000562-5]

**Citations detected**: `medquad_ans_0000562-5`
**Supported citations**: `medquad_ans_0000562-5`
**Unsupported citations**: _none_

## Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Top docs**:

1. `medquad_ans_0000222-2` (score=0.7000) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
2. `medquad_ans_0000222-1` (score=0.6722) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of the diaphragm before ...
3. `medquad_ans_0001497-1` (score=0.6459) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severity from a thinned are...
4. `medquad_ans_0000222-5` (score=0.5524) - These resources address the diagnosis or management of congenital diaphragmatic hernia: - Boston Children's Hospital - Children's Hospital of Philadelphia - Columbia University Medical Center: DHREAMS - Columbia University Medical Center...
5. `medquad_ans_0000222-4` (score=0.5243) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or chromosomal abnormali...

**Answer**:

Congenital diaphragmatic hernia (CDH) is a defect in the diaphragm that results from abnormal development before birth. The diaphragm, composed of muscle and other fibrous tissue, normally separates the organs in the abdomen from those in the chest. CDH ranges in severity from a thinned area in the diaphragm to its complete absence. This defect allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs, which can lead to underdevelopment of the lungs (pulmonary hypoplasia) and potentially life-threatening breathing difficulties at birth [medquad_ans_0000222-1]. CDH may also present later in life with symptoms such as breathing problems or abdominal pain due to intestinal protrusion into the chest cavity, or it may be asymptomatic and detected incidentally during imaging for other reasons [medquad_ans_0000222

**Citations detected**: `medquad_ans_0000222-1`
**Supported citations**: `medquad_ans_0000222-1`
**Unsupported citations**: _none_

## Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Top docs**:

1. `medquad_ans_0000173-2` (score=0.7000) - Charcot-Marie-Tooth disease is the most common inherited disorder that involves the peripheral nerves, affecting an estimated 150,000 people in the United States. It occurs in populations worldwide with a prevalence of about 1 in 2,500 i...
2. `medquad_ans_0001088-1__dup1` (score=0.3345) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common symptoms may includ...
3. `medquad_ans_0001093-1__dup1` (score=0.3067) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is available, the table be...
4. `medquad_ans_0001090-1__dup1` (score=0.2876) - Charcot-Marie-Tooth disease type 1A (CMT1A) is a type of inherited neurological disorder that affects the peripheral nerves. Affected individuals experience weakness and wasting (atrophy) of the muscles of the lower legs beginning in ado...
5. `medquad_ans_0000173-1` (score=0.2721) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as touch, pain, heat, and ...

**Answer**:

Charcot-Marie-Tooth disease is the most common inherited disorder that involves the peripheral nerves, affecting an estimated 150,000 people in the United States and occurring in populations worldwide with a prevalence of about 1 in 2,500 individuals. It is a group of disorders that affect the peripheral nerves, which run from the brain and spine to the rest of the body. Defects in many different genes cause different forms of this disease. Common symptoms include foot drop, foot deformity, loss of lower leg muscle, numbness in the foot or leg, slapping gait, and weakness of the hips, legs, or feet. There is currently no cure, but management can include physical therapy, occupational therapy, braces, orthopedic devices, pain medication, and orthopedic surgery. There are over 40 types of Charcot-Marie-Tooth disease [medquad_ans_0000

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Top docs**:

1. `medquad_ans_0000300-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
2. `medquad_ans_0000676-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
3. `medquad_ans_0000990-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
4. `medquad_ans_0000685-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...
5. `medquad_ans_0000770-4` (score=0.7000) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of the mutated gene, b...

**Answer**:

The provided context does not specify whether monilethrix is inherited. While several documents describe autosomal recessive inheritance patterns, none explicitly mention monilethrix or its mode of inheritance. Therefore, based on the given information, it is not possible to determine if monilethrix is inherited.

[Note: No specific document cites monilethrix, so no doc_id can be assigned.]

**Citations detected**: `Note: No specific document cites monilethrix, so no doc_id can be assigned.`
**Supported citations**: _none_
**Unsupported citations**: `Note: No specific document cites monilethrix, so no doc_id can be assigned.`

## Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Top docs**:

1. `medquad_ans_0000223-12` (score=0.7000) - - A urinary tract infection (UTI) is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscope. - Most UTIs are caused by bacteria that live in the bowel, the part of the dige...
2. `medquad_ans_0000215-15` (score=0.6880) - - Urinary tract infections (UTIs) usually occur when the body fails to remove bacteria rapidly from the urinary tract. - UTIs affect about 3 percent of children in the United States every year. - Most UTIs are not serious, but chronic ki...
3. `medquad_ans_0000214-1__dup4` (score=0.6240) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Normally, bacteria that e...
4. `medquad_ans_0000215-1__dup3` (score=0.6240) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Normally, bacteria that e...
5. `medquad_ans_0000071-7__dup1` (score=0.4925) - Urinary tract infections (UTI) are a common bladder problem, especially as people age. UTIs are the second most common type of infection in the body. Each year, UTIs cause more than 8 million visits to health care providers. UTIs can hap...

**Answer**:

A urinary tract infection (UTI) in children is an infection in the urinary tract caused by microorganisms too small to be seen without a microscope, most commonly bacteria [medquad_ans_0000214-1__dup4]. These infections occur when bacteria enter the urinary tract and overcome the body's natural defenses, which normally remove bacteria rapidly before they cause symptoms [medquad_ans_0000214-1__dup4]. Bacteria that live in the bowel are a common source of UTIs [medquad_ans_0000223-12]. UTIs can affect any child, though girls are more likely to get them than boys [medquad_ans_0000223-12]. Symptoms may range from mild, such as a burning sensation during urination or unusual-smelling urine, to severe, including high fever and pain, or the child may have

**Citations detected**: `medquad_ans_0000214-1__dup4`, `medquad_ans_0000223-12`
**Supported citations**: `medquad_ans_0000214-1__dup4`, `medquad_ans_0000223-12`
**Unsupported citations**: _none_

## Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Top docs**:

1. `medquad_ans_0004939-2` (score=0.7000) - What are the signs and symptoms of Polycystic kidney disease? Signs and symptoms vary greatly from person to person. But affected individuals typically develop multiple cysts in both kidneys, which impair their ability to filter waste pr...
2. `medquad_ans_0000804-1` (score=0.6182) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the blood. The growth o...
3. `medquad_ans_0004939-1` (score=0.4842) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the their ability to fil...
4. `medquad_ans_0000804-2` (score=0.4645) - Polycystic kidney disease is a fairly common genetic disorder. It affects about 500,000 people in the United States. The autosomal dominant form of the disease is much more common than the autosomal recessive form. Autosomal dominant pol...
5. `medquad_ans_0000167-1__dup4` (score=0.4139) - Polycystic kidney disease is a genetic disorder that causes numerous cysts to grow in the kidneys. A kidney cyst is an abnormal sac filled with fluid. PKD cysts can greatly enlarge the kidneys while replacing much of their normal structu...

**Answer**:

Complications of Polycystic Kidney Disease include dangerously high blood pressure (hypertension), severe pain in the back or sides, blood in the urine (hematuria), recurrent urinary tract infections, kidney stones (nephrolithiasis), heart valve abnormalities, and an increased risk of aortic aneurysm in the brain (cerebral aneurysm), which can be life-threatening if it tears or ruptures. Other complications include cystic liver disease, proteinuria, renal insufficiency, abnormality of prenatal development or birth, abnormality of the pancreas, abnormality of the respiratory system, dilatation of the ascending aorta, hydrocephalus, recurrent fractures, reduced bone mineral density, and sarcoma [medquad_ans_0004939-2].

**Citations detected**: `medquad_ans_0004939-2`
**Supported citations**: `medquad_ans_0004939-2`
**Unsupported citations**: _none_

## Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Top docs**:

1. `medquad_ans_0000649-1__dup1` (score=0.7000) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contaminated food or water....
2. `medquad_ans_0000949-1__dup1` (score=0.2874) - Viruses are capsules with genetic material inside. They are very tiny, much smaller than bacteria. Viruses cause familiar infectious diseases such as the common cold, flu and warts. They also cause severe illnesses such as HIV/AIDS, smal...
3. `medquad_ans_0000123-4__dup2` (score=0.2147) - Four types of viruses cause most cases of viral gastroenteritis. Rotavirus Rotavirus is the leading cause of gastroenteritis among infants and young children. Rotavirus infections are most common in infants 3 to 15 months old. Symptoms u...
4. `medquad_ans_0000092-1__dup1` (score=0.1657) - Cytomegalovirus (CMV) is a virus found throughout the world that infects between 50 to 80 percent of all adults in the United States by the age of 40. CMV is in the same family of viruses that causes cold sores (herpes simplex virus), in...
5. `medquad_ans_0000228-4__dup2` (score=0.1370) - Laboratory tests that are used to diagnose Hendra virus (HV) and Nipah virus (NV) include detection of antibody by ELISA (IgG and IgM), real time polymerase chain reaction (RT-PCR), and virus isolation attempts. In most countries, handli...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by noroviruses, which are a group of related viruses. Infection leads to an inflammation of the stomach and intestines, with common symptoms including diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms are fever, headache, and body aches. The illness can spread from person to person, through contaminated food or water, or by touching a contaminated surface. Norovirus is the most common calicivirus and the most common cause of viral gastroenteritis in adults, often responsible for epidemics, especially from October to April. Symptoms typically appear 1 to 2 days after exposure and last for 1 to 3 days. There is no specific medicine to treat norovirus infections; treatment includes bed rest and plenty of fluids to prevent dehydration. Prevention can be achieved through proper hand washing and safe food preparation [medquad_ans_000064

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Top docs**:

1. `medquad_ans_0000222-4` (score=0.7000) - Isolated congenital diaphragmatic hernia is rarely inherited. In almost all cases, there is only one affected individual in a family. When congenital diaphragmatic hernia occurs as a feature of a genetic syndrome or chromosomal abnormali...
2. `medquad_ans_0000222-2` (score=0.6520) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
3. `medquad_ans_0001497-3` (score=0.6512) - What causes congenital diaphragmatic hernia? Congenital diaphragmatic hernia (CDH) can occur as an isolated finding, as part of a genetic syndrome or chromosome abnormality, or as part of a complex but nonsyndromic set of findings. Curre...
4. `medquad_ans_0000222-5` (score=0.5880) - These resources address the diagnosis or management of congenital diaphragmatic hernia: - Boston Children's Hospital - Children's Hospital of Philadelphia - Columbia University Medical Center: DHREAMS - Columbia University Medical Center...
5. `medquad_ans_0000222-3` (score=0.5611) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow syndrome, Fryns syn...

**Answer**:

Genetic changes related to congenital diaphragmatic hernia (CDH) include:

- Recognized chromosome abnormalities [medquad_ans_0001497-3].
- Single gene disorders [medquad_ans_0001497-3].
- Currently undetectable chromosomal microdeletions (tiny loss of genetic material) or microduplications (an extra copy of genetic material) [medquad_ans_0001497-3].
- Mutations in major genes important for diaphragm development [medquad_ans_0001497-3].
- Combined effects of multiple minor genetic mutations or variants (polygenic inheritance) [medquad_ans_0001497-3].
- Gene-environment interactions (multifactorial inheritance) [medquad_ans_0001497-3].

**Citations detected**: `medquad_ans_0001497-3`
**Supported citations**: `medquad_ans_0001497-3`
**Unsupported citations**: _none_

## Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Top docs**:

1. `medquad_ans_0002533-3` (score=0.7000) - What causes Gerstmann-Straussler-Scheinker disease? Gerstmann-Straussler-Scheinker disease (GSS) is usually caused by certain changes (mutations) in the PRNP gene. PRNP encodes a protein called prion protein. Although the exact function ...
2. `medquad_ans_0004784-4` (score=0.1389) - How is Pelizaeus-Merzbacher disease inherited?
3. `medquad_ans_0004750-3` (score=0.1256) - What causes Parsonage Turner syndrome? The exact cause of Parsonage Turner syndrome (PTS) is unknown. Researchers suspect that most cases are due to an autoimmune response following exposure to an illness or environmental factor. In many...
4. `medquad_ans_0002840-1` (score=0.0900) - Hereditary neuralgic amyotrophy is a type of nervous system disease that affects the brachial plexus. Common signs and symptoms include episodes of severe pain and muscle wasting in one or both shoulders and arms. Attacks may be spontane...
5. `medquad_ans_0000011-13` (score=0.0443) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten CJD from medical proc...

**Answer**:

The provided context does not contain information about the causes of Omenn syndrome. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Citations detected**: `no_context`
**Supported citations**: _none_
**Unsupported citations**: `no_context`

## Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Top docs**:

1. `medquad_ans_0000097-1` (score=0.7000) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the cerebellum). Signs and...
2. `medquad_ans_0005792-1` (score=0.4307) - Spinocerebellar ataxia autosomal recessive 7, also called SCAR7, is a slowly progressive hereditary form of spinocerebellar ataxia. Symptoms of SCAR7 can include difficulty walking and writing, speech difficulties (dysarthria), limb atax...
3. `medquad_ans_0005761-1` (score=0.2948) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal features (increased...
4. `medquad_ans_0000097-5` (score=0.2702) - These resources address the diagnosis or management of ARCA1: - Gene Review: Gene Review: SYNE1-Related Autosomal Recessive Cerebellar Ataxia - Genetic Testing Registry: Spinocerebellar ataxia, autosomal recessive 8 - Johns Hopkins Medic...
5. `medquad_ans_0005794-1` (score=0.2029) - What are the signs and symptoms of Spinocerebellar ataxia autosomal recessive with axonal neuropathy? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia autosomal recessive with axon...

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the cerebellum, the part of the brain that coordinates movement. Signs and symptoms first appear in early to mid-adulthood and include impaired speech (dysarthria), problems with coordination and balance (ataxia), or both. Individuals may also have difficulty with movements that involve judging distance or scale (dysmetria). Other features include abnormal eye movements (nystagmus) and problems following the movements of objects with the eyes. The movement problems are slowly progressive and may eventually result in the need for a cane, walker, or wheelchair [medquad_ans_0000097-1].

**Citations detected**: `medquad_ans_0000097-1`
**Supported citations**: `medquad_ans_0000097-1`
**Unsupported citations**: _none_

## Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Top docs**:

1. `medquad_ans_0000049-2__dup1` (score=0.7000) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in some cases, surgery.
2. `medquad_ans_0000050-2__dup1` (score=0.7000) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in some cases, surgery.
3. `medquad_ans_0000050-4__dup1` (score=0.6481) - The NINDS conducts and supports research on injuries to the nervous system such as brachial plexus injuries. Much of this research is aimed at finding ways to prevent and treat these disorders.
4. `medquad_ans_0000049-4__dup1` (score=0.6481) - The NINDS conducts and supports research on injuries to the nervous system such as brachial plexus injuries. Much of this research is aimed at finding ways to prevent and treat these disorders.
5. `medquad_ans_0000123-1__dup2` (score=0.6304) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Symptoms may include - A limp or paralyzed arm - Lack of muscl...

**Answer**:

The treatments for brachial plexus injuries include physical therapy and, in some cases, surgery [medquad_ans_0000049-2__dup1].

**Citations detected**: `medquad_ans_0000049-2__dup1`
**Supported citations**: `medquad_ans_0000049-2__dup1`
**Unsupported citations**: _none_

## Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Top docs**:

1. `medquad_ans_0000216-7` (score=0.7000) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
2. `medquad_ans_0000216-8` (score=0.6419) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PUV) - ureterocele - S...
3. `medquad_ans_0000216-4__dup1` (score=0.5467) - When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis. Hydronephrosis is the most common problem found during prenatal ultrasound of a baby...
4. `medquad_ans_0000216-3__dup1` (score=0.5177) - Before leaving the hospital, a baby with urine blockage may urinate only small amounts or may not urinate at all. As part of the routine newborn exam, the health care provider may feel an enlarged kidney or find a closed urethra, which m...
5. `medquad_ans_0000154-7` (score=0.3968) - The complications of kidney dysplasia can include - hydronephrosis of the working kidney. A baby with kidney dysplasia in only one kidney might have other urinary tract defects. When other defects in the urinary tract block the flow of u...

**Answer**:

Urine blockage in newborns can be caused by various defects in the urinary tract, including vesicoureteral reflux (VUR), ureteropelvic junction (UPJ) obstruction, bladder outlet obstruction (BOO) such as posterior urethral valves (PUV), and ureterocele. Additionally, genetic conditions like prune belly syndrome (PBS), esophageal atresia (EA), and congenital heart defects can affect the urinary tract. Spina bifida and other birth defects affecting the spinal cord may also cause urine blockage [medquad_ans_0000216-8]. When a defect blocks urine flow, it causes urine to back up, leading to hydroureter and hydronephrosis [medquad_ans_0000216-8]. Hydronephrosis is commonly detected during prenatal ultrasound and may indicate the presence of a blockage [medquad

**Citations detected**: `medquad_ans_0000216-8`
**Supported citations**: `medquad_ans_0000216-8`
**Unsupported citations**: _none_

## Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Top docs**:

1. `medquad_ans_0003179-1` (score=0.7000) - Inclusion body myopathy 2, also known as hereditary inclusion body myopathy (HIBM), GNE-related myopathy, distal myopathy with rimmed vacuoles, and Nonaka myopathy, is an inherited condition that primarily affects the skeletal muscles (t...
2. `medquad_ans_0000707-1__dup1` (score=0.5648) - Bethlem myopathy is an inherited movement disorder characterized by progressive muscle weakness and joint stiffness (contractures) in the fingers, wrists, elbows, and ankles. Due to a progressive course, up to two-thirds of people with t...
3. `medquad_ans_0001035-1__dup1` (score=0.5049) - Centronuclear myopathy refers to a group of rare, inherited conditions that affect the muscles. There are three main forms of the condition that are differentiated by their pattern of inheritance: X-linked Myotubular Myopathy Autosomal D...
4. `medquad_ans_0005086-3` (score=0.4501) - How might progressive hemifacial atrophy be treated?
5. `medquad_ans_0006236-4` (score=0.4346) - Is tubular aggregate myopathy genetic? It is evident from family history studies that the condition can be passed through families in either an autosomal dominant or autosomal recessive fashion. Some cases appear to be due to dominant mu...

**Answer**:

The provided context does not contain information about the number of people affected by intranuclear rod myopathy. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Citations detected**: `no_context`
**Supported citations**: _none_
**Unsupported citations**: `no_context`

## Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Top docs**:

1. `medquad_ans_0000058-3__dup4` (score=0.7000) - Blood pressure is the force of blood flow inside your blood vessels. Blood pressure is written with two numbers separated by a slash. For example, a blood pressure result of 130/80 is said as 130 over 80. The first number is the pressure...
2. `medquad_ans_0000036-11` (score=0.5862) - High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. There are two main types of high blood pressure: primary and secondary high blood pressure. Primary, or essent...
3. `medquad_ans_0000036-9` (score=0.5811) - Blood pressure is the force of blood pushing against the walls of the blood vessels as the heart pumps blood. If your blood pressure rises and stays high over time, its called high blood pressure. High blood pressure is dangerous because...
4. `medquad_ans_0000471-1__dup1` (score=0.5550) - Blood pressure is the force of your blood pushing against the walls of your arteries. Each time your heart beats, it pumps blood into the arteries. Your blood pressure is highest when your heart beats, pumping the blood. This is called s...
5. `medquad_ans_0000150-1__dup3` (score=0.5489) - Blood pressure is the force of blood pushing against blood vessel walls as the heart pumps out blood, and high blood pressure, also called hypertension, is an increase in the amount of force that blood places on blood vessels as it moves...

**Answer**:

High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. There are two main types: primary (or essential) high blood pressure, which is the most common type and tends to develop over years as a person ages, and secondary high blood pressure, which is caused by another medical condition or the use of certain medicines and usually resolves after the cause is treated or removed. High blood pressure is dangerous because it makes the heart work too hard and the high force of blood flow can harm arteries and organs such as the heart, kidneys, brain, and eyes. It is typically defined as a blood pressure reading of 140/90 or higher, with prehypertension occurring between 120–139 for the top number or 80–89 for the bottom number. High blood pressure usually has no symptoms but can lead to serious problems such as stroke, heart

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Top docs**:

1. `medquad_ans_0000139-2__dup2` (score=0.7000) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the gene on to you. You...
2. `medquad_ans_0000692-1__dup2` (score=0.6208) - Parkinson's disease (PD) is a type of movement disorder. It happens when nerve cells in the brain don't produce enough of a brain chemical called dopamine. Sometimes it is genetic, but most cases do not seem to run in families. Exposure ...
3. `medquad_ans_0001065-4` (score=0.6037) - This condition is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In females (who have two X chromosomes), a mutation in one of the two co...
4. `medquad_ans_0004758-4` (score=0.5251) - Is Partington syndrome inherited? Partington syndrome is inherited in an X-linked recessive manner. A condition is considered X-linked if the mutated gene that causes the condition is located on the X chromosome, one of the two sex chrom...
5. `medquad_ans_0000307-1__dup1` (score=0.4896) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a group of muscles or ...

**Answer**:

The provided context does not contain information about X-linked dystonia-parkinsonism or its inheritance pattern. Therefore, based on the given context, it is not possible to determine whether X-linked dystonia-parkinsonism is inherited.

[No supporting source available]

**Citations detected**: `No supporting source available`
**Supported citations**: _none_
**Unsupported citations**: `No supporting source available`

## Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Top docs**:

1. `medquad_ans_0000120-1` (score=0.7000) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process ketones, which are molec...
2. `medquad_ans_0000702-1__dup1` (score=0.5682) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ketones, which are mole...
3. `medquad_ans_0000120-2` (score=0.5083) - Beta-ketothiolase deficiency appears to be very rare. It is estimated to affect fewer than 1 in 1 million newborns.
4. `medquad_ans_0000702-2__dup1` (score=0.4390) - What are the signs and symptoms of Beta ketothiolase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Beta ketothiolase deficiency. If the information is available, the table below includes h...
5. `medquad_ans_0000120-5` (score=0.3669) - These resources address the diagnosis or management of beta-ketothiolase deficiency: - Baby's First Test - Genetic Testing Registry: Deficiency of acetyl-CoA acetyltransferase These resources from MedlinePlus offer information about the ...

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. Signs and symptoms typically appear between the ages of 6 months and 24 months and include episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, known as ketoacidotic attacks, may lead to coma and are frequently triggered by infections, periods without food (fasting), or increased intake of protein-rich foods. The condition is inherited in an autosomal recessive fashion and is caused by mutations in the ACAT1 gene. It is very rare, affecting fewer than 1 in 1 million newborns [medquad*ans_0000120-1, medquad_ans*

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 285 (`0000006_5-7`)

**Question**: What are the stages of Childhood Brain Stem Glioma ?

**Top docs**:

1. `medquad_ans_0000006_5-7` (score=0.7000) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread...
2. `medquad_ans_0000006_5-1` (score=0.6396) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - There are two types o...
3. `medquad_ans_0000006_5-4` (score=0.4475) - The signs and symptoms of brain stem glioma are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain. - The size of the tumor and whether it has spread all through the brain stem. ...
4. `medquad_ans_0000006_5-6` (score=0.3473) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis and treatment options depend on: - The type of brain stem glioma. - Where the tumor is found in the brain and if it has spread within the brain st...
5. `medquad_ans_0002574-1` (score=0.3188) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of glial cell is involved...

**Answer**:

There is no standard staging system for childhood brain stem glioma [medquad_ans_0000006_5-7].

**Citations detected**: `medquad_ans_0000006_5-7`
**Supported citations**: `medquad_ans_0000006_5-7`
**Unsupported citations**: _none_

## Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Top docs**:

1. `medquad_ans_0000266-1__dup2` (score=0.7000) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal cord (conus medullari...
2. `medquad_ans_0000266-3__dup1` (score=0.5417) - With treatment, individuals with tethered spinal cord syndrome have a normal life expectancy. However, some neurological and motor impairments may not be fully correctable. Surgery soon after symptoms emerge appears to improve chances fo...
3. `medquad_ans_0000061-1__dup1` (score=0.1706) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to and from parts of the...
4. `medquad_ans_0004257-2` (score=0.1429) - What are the signs and symptoms of myelomeningocele? A baby born with a myelomeningocele may have a sac sticking out of the mid to lower back that the doctor cannot see through when shining a light behind it. Symptoms of this condition i...
5. `medquad_ans_0000257-1__dup2` (score=0.1405) - Spinal cord infarction is a stroke either within the spinal cord or the arteries that supply it. It is caused by arteriosclerosis or a thickening or closing of the major arteries to the spinal cord. Frequently spinal cord infarction is c...

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course. In children, symptoms may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is believed to result from improper growth of the neural tube during fetal development and is closely linked to spina bifida. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. Delayed symptom presentation is related to the degree of strain on the spinal cord over

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Top docs**:

1. `medquad_ans_0000058-1__dup1` (score=0.7000) - Carpal tunnel syndrome (CTS) occurs when the median nerve, which runs from the forearm into the palm of the hand, becomes pressed or squeezed at the wrist. The carpal tunnel is a narrow, rigid passageway of ligament and bones at the base...
2. `medquad_ans_0000058-3__dup1` (score=0.6972) - In general, carpal tunnel syndrome responds well to treatment, but less than half of individuals report their hand(s) feeling completely normal following surgery. Some residual numbness or weakness is common. At work, people can perform ...
3. `medquad_ans_0000155-1__dup3` (score=0.4063) - You're working at your desk, trying to ignore the tingling or numbness you've had for some time in your hand and wrist. Suddenly, a sharp, piercing pain shoots through the wrist and up your arm. Just a passing cramp? It could be carpal t...
4. `medquad_ans_0000469-1` (score=0.3773) - Hereditary neuropathy with liability to pressure palsies is a disorder that affects peripheral nerves. These nerves connect the brain and spinal cord to muscles as well as sensory cells that detect touch, pain, and temperature. In people...
5. `medquad_ans_0000970-1` (score=0.2811) - Tarsal-carpal coalition syndrome is a rare, inherited bone disorder that affects primarily the hands and feet. Several individual bones make up each wrist (carpal bones) and ankle (tarsal bones). In tarsal-carpal coalition syndrome, the ...

**Answer**:

The outlook for carpal tunnel syndrome is generally positive, as the condition responds well to treatment. However, less than half of individuals report their hand(s) feeling completely normal following surgery, and some residual numbness or weakness is common [medquad_ans_0000058-3__dup1]. Early diagnosis and treatment are important to prevent permanent nerve damage [medquad_ans_00000155-1__dup3]. With appropriate management, such as splints, rest, medications, or surgery, symptoms can be effectively managed, although full recovery is not guaranteed in all cases.

**Citations detected**: `medquad_ans_0000058-3__dup1`, `medquad_ans_00000155-1__dup3`
**Supported citations**: `medquad_ans_0000058-3__dup1`
**Unsupported citations**: `medquad_ans_00000155-1__dup3`

## Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Top docs**:

1. `medquad_ans_0005077-3` (score=0.7000) - How is progressive bulbar palsy treated? Treatments aim to help people cope with the symptoms of progressive bulbar palsy, such as feeding tubes, devices to help with talking, and medicines to treat muscle spasms, weakness, drooling, sle...
2. `medquad_ans_0000188-2__dup1` (score=0.6016) - MJD is incurable, but some symptoms of the disease can be treated. For those individuals who show parkinsonian features, levodopa therapy can help for many years. Treatment with antispasmodic drugs, such as baclofen, can help reduce spas...
3. `medquad_ans_0006133-6` (score=0.5655) - Is there treatment for Townes-Brocks syndrome? Treatment is directed towards the specific symptoms, including immediate surgical intervention for imperforate anus; surgery for severe malformations of the hands; routine management of cong...
4. `medquad_ans_0000860-2__dup1` (score=0.5411) - How might Brown-Sequard syndrome be treated?
5. `medquad_ans_0003799-5` (score=0.4833) - How might Marden-Walker syndrome be treated? Very little information is available regarding the treatment of Marden-Walker syndrome. In general, treatment is symptomatic, with a multidisciplinary approach. The team of providers may inclu...

**Answer**:

The provided context does not contain any information about the treatments for Donnai-Barrow syndrome. [doc_id]

**Citations detected**: `doc_id`
**Supported citations**: _none_
**Unsupported citations**: `doc_id`

## Example 289 (`0000811-1`)

**Question**: What is (are) porphyria ?

**Top docs**:

1. `medquad_ans_0000811-1` (score=0.7000) - Porphyria is a group of disorders caused by abnormalities in the chemical steps that lead to heme production. Heme is a vital molecule for all of the body's organs, although it is most abundant in the blood, bone marrow, and liver. Heme ...
2. `medquad_ans_0000729-1__dup1` (score=0.6960) - Porphyrias are a group of genetic disorders caused by problems with how your body makes a substance called heme. Heme is found throughout the body, especially in your blood and bone marrow, where it carries oxygen. There are two main typ...
3. `medquad_ans_0001134-1` (score=0.6745) - Chester porphyria is a unique type of porphyria with the signs and symptoms of acute intermittent porphyria (AIP) and the biochemical defects of both AIP and variegate porphyria (VP). Chester porphyria does not conform to any of the reco...
4. `medquad_ans_0000189-4__dup2` (score=0.6321) - The exact rates of porphyria are unknown and vary around the world. For example, porphyria cutanea tarda is most common in the United States, and variegate porphyria is most common in South America.1
5. `medquad_ans_0000811-3` (score=0.5986) - Each form of porphyria results from mutations in one of these genes: ALAD, ALAS2, CPOX, FECH, HMBS, PPOX, UROD, or UROS. The genes related to porphyria provide instructions for making the enzymes needed to produce heme. Mutations in most...

**Answer**:

Porphyria is a group of disorders caused by abnormalities in the chemical steps that lead to heme production. Heme is a vital molecule for all of the body's organs, although it is most abundant in the blood, bone marrow, and liver. Heme is a component of several iron-containing proteins called hemoproteins, including hemoglobin (the protein that carries oxygen in the blood). Researchers have identified several types of porphyria, which are distinguished by their genetic cause and their signs and symptoms. Some types of porphyria, called cutaneous porphyrias, primarily affect the skin, causing areas of skin exposed to the sun to become fragile and blistered, which can lead to infection, scarring, changes in skin coloring (pigmentation), and increased hair growth. Cutaneous porphyrias include congenital erythropoietic porphyria, erythropoietic protoporphyria, hepatoerythropoiet

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 290 (`0000585-1__dup1`)

**Question**: What is (are) Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy ?

**Top docs**:

1. `medquad_ans_0000585-1__dup1` (score=0.7000) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable degrees of hearing lo...
2. `medquad_ans_0004557-1` (score=0.3748) - Optic atrophy 1 is a condition that mainly affects vision, but may include other features. Vision loss typically begins within the first decade of life; severity varies widely among affected people (from nearly normal vision to complete ...
3. `medquad_ans_0005818-1` (score=0.1487) - Spondylocarpotarsal synostosis (SCT) syndrome is an inherited syndrome characterized by disproportionate short stature, abnormalities of the vertebrae in the spine, scoliosis and lordosis, carpal and tarsal fusion (fusion of the bones in...
4. `medquad_ans_0004797-1` (score=0.1380) - Pendred syndrome is a condition usually characterized by sensorineural hearing loss in both ears (bilateral) and euthyroid goiter (enlargement of the thyroid gland with normal thyroid gland function). The amount of hearing loss varies am...
5. `medquad_ans_0001196-1` (score=0.1018) - Choroideremia is a genetic condition that causes vision loss. This disorder typically affects males. The first symptom is usually impairment of night vision (night blindness), which can occur in childhood. People with this disorder also ...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. Autosomal recessive inheritance has been suggested [medquad_ans_0000585-1].

**Citations detected**: `medquad_ans_0000585-1`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000585-1`

## Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Top docs**:

1. `medquad_ans_0000167-2__dup3` (score=0.7000) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up a gene. In most cas...
2. `medquad_ans_0004167-3` (score=0.4745) - What causes Muir-Torre syndrome? Muir-Torre syndrome is a subtype of Lynch syndrome and may be caused by changes (mutations) in either the MLH1, MSH2, or MSH6 gene. These genes give the body instructions to make proteins needed for repai...
3. `medquad_ans_0002833-3` (score=0.4053) - What causes hereditary leiomyomatosis and renal cell cancer? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is caused by changes (mutations) in the FH gene. This gene gives the body instructions for making an enzyme called fumar...
4. `medquad_ans_0003635-1` (score=0.3521) - Lipoic acid synthetase deficiency is a rare condition that affects the mitochondria. Mitochondria are tiny structures found in almost every cell of the body. They are responsible for creating most of the energy necessary to sustain life ...
5. `medquad_ans_0001047-3` (score=0.2858) - X-linked chondrodysplasia punctata 1 is caused by genetic changes involving the ARSE gene. This gene provides instructions for making an enzyme called arylsulfatase E. The function of this enzyme is unknown, although it appears to be imp...

**Answer**:

The provided context does not contain information about the genetic changes related to phosphoglycerate kinase deficiency. Therefore, a definitive answer cannot be provided based on the given text. [no_context]

**Citations detected**: `no_context`
**Supported citations**: _none_
**Unsupported citations**: `no_context`

## Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Top docs**:

1. `medquad_ans_0000673-4` (score=0.7000) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
2. `medquad_ans_0004784-4` (score=0.6175) - How is Pelizaeus-Merzbacher disease inherited?
3. `medquad_ans_0000947-4` (score=0.4945) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000493-4` (score=0.4945) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000465-4` (score=0.4945) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The provided context does not contain any information about the genetic changes related to pulmonary veno-occlusive disease. Therefore, the question cannot be answered based on the given context. [no relevant context]

**Citations detected**: `no relevant context`
**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

## Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Top docs**:

1. `medquad_ans_0001006-3` (score=0.7000) - Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. In the liver, enzymes break down tyrosine in a five step process, resulting in molecules that are either excreted by the kidneys or used t...
2. `medquad_ans_0000189-5__dup1` (score=0.5341) - Most porphyrias are inherited disorders. Scientists have identified genes for all eight enzymes in the heme biosynthetic pathway. Most porphyrias result from inheriting an abnormal gene, also called a gene mutation, from one parent. Some...
3. `medquad_ans_0004086-1` (score=0.2113) - Mitochondrial genetic disorders refer to a group of conditions that affect the mitochondria (the structures in each cell of the body that are responsible for making energy). People with these conditions can present at any age with almost...
4. `medquad_ans_0000396-1__dup2` (score=0.1960) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of the brain. Some geneti...
5. `medquad_ans_0000133-3__dup2` (score=0.1730) - Wilson disease is caused by an inherited autosomal recessive mutation, or change, in the ATP7B gene. In an autosomal recessive disease, the child has to inherit the gene mutation from both parents to have an increased likelihood for the ...

**Answer**:

Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. These mutations result in a decrease in the activity of one of the enzymes involved in the breakdown of tyrosine, leading to the accumulation of toxic levels of tyrosine and its byproducts, which can damage cells in the liver, kidneys, nervous system, and other organs [medquad_ans_0001006-3].

**Citations detected**: `medquad_ans_0001006-3`
**Supported citations**: `medquad_ans_0001006-3`
**Unsupported citations**: _none_

## Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Top docs**:

1. `medquad_ans_0000436-5` (score=0.7000) - These resources address the diagnosis or management of Graves disease: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Graves disease 2 - Genetic Testing Registry: Graves disease 3 - Genetic Testing Reg...
2. `medquad_ans_0000004-6__dup2` (score=0.6946) - People with Graves disease have three treatment options: radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves disease in the United States. Graves disease is often diagnosed a...
3. `medquad_ans_0000004-2__dup3` (score=0.4614) - People with Graves disease may have common symptoms of hyperthyroidism such as - nervousness or irritability - fatigue or muscle weakness - heat intolerance - trouble sleeping - hand tremors - rapid and irregular heartbeat - frequent bow...
4. `medquad_ans_0000004-3__dup4` (score=0.4587) - Graves ophthalmopathy is a condition associated with Graves disease that occurs when cells from the immune system attack the muscles and other tissues around the eyes. The result is inflammation and a buildup of tissue and fat behind the...
5. `medquad_ans_0000004-9` (score=0.3906) - - Graves disease is the most common cause of hyperthyroidism in the United States. - In Graves disease, the immune system stimulates the thyroid gland to make too much thyroid hormone. - Common symptoms of hyperthyroidism include nervous...

**Answer**:

The treatments for Graves' disease include radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves' disease in the United States. In this therapy, patients take radioactive iodine-131 by mouth, which is collected by the thyroid gland and gradually destroys thyroid cells, reducing hormone production. The dose may be adjusted to either shut down the thyroid completely or bring hormone production into the normal range. Multiple rounds may be needed, and improvements in symptoms may take several weeks or months. Radioiodine therapy is not used in pregnant women or women who are breastfeeding due to potential harm to the fetus or passage into breast milk. Experts recommend waiting at least one year after treatment before becoming pregnant. Almost everyone who receives radioiodine therapy eventually develops hypothyroidism. Other treatments, such as prescription steroids, may help prevent worsening of Graves' ophthalmopathy (GO) associated with radioiodine therapy [med

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Top docs**:

1. `medquad_ans_0000039-1__dup2` (score=0.7000) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, or stumbling. Sympto...
2. `medquad_ans_0000039-2__dup1` (score=0.5107) - As yet, no specific treatment is known that can halt or reverse the symptoms of Batten disease. However, seizures can sometimes be reduced or controlled with anticonvulsant drugs, and other medical problems can be treated appropriately a...
3. `medquad_ans_0000039-3__dup1` (score=0.4749) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten disease is often fatal b...
4. `medquad_ans_0000545-2` (score=0.4319) - Juvenile Batten disease is the most common type of NCL, but its exact prevalence is unknown. Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, where approximately 1 in...
5. `medquad_ans_0000545-1` (score=0.4205) - Juvenile Batten disease is an inherited disorder that primarily affects the nervous system. After a few years of normal development, children with this condition develop progressive vision loss, intellectual and motor disability, speech ...

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. It is characterized by a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. Early signs may include personality and behavior changes, slow learning, clumsiness, or stumbling. Vision loss is often an early sign and may lead to suspicion during an eye exam. The disease progresses over time, leading to cognitive impairment, worsening seizures, progressive loss of sight and motor skills, and eventually blindness, bedridden state, and dementia. Juvenile Batten disease, the most common type of neuronal ceroid lipofuscinosis (NCL), typically presents with progressive vision loss starting between ages 4 and 8, followed by developmental regression, intellectual decline, speech difficulties, and loss of motor skills such as walking or sitting. It is often fatal by the late teens or twenties [medquad_ans_000

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Top docs**:

1. `medquad_ans_0000318-1__dup1` (score=0.7000) - Ehlers-Danlos syndrome (EDS) is a group of inherited disorders that weaken connective tissues. Connective tissues are proteins that support skin, bones, blood vessels, and other organs. EDS usually affects your skin, joints and blood ves...
2. `medquad_ans_0005116-1` (score=0.2455) - Proteus-like syndrome describes people who do not meet the diagnostic criteria for Proteus syndrome but who share many of the characteristic signs and symptoms associated with the condition. Affected people may experience some of the fol...
3. `medquad_ans_0004784-4` (score=0.2270) - How is Pelizaeus-Merzbacher disease inherited?
4. `medquad_ans_0000673-4` (score=0.0586) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
5. `medquad_ans_0000886-1__dup1` (score=0.0173) - C syndrome, also known as Opitz trigonocephaly syndrome, is characterized by trigonocephaly, severe intellectual disability, hypotonia, variable cardiac defects, redundant (extra folds of) skin, joint and limb abnormalities, and unusual ...

**Answer**:

The provided context does not contain information about Ehlers-Danlos syndrome, progeroid type. Therefore, a specific answer cannot be generated based on the given context. [medquad_ans_0000318-1__dup1] discusses general Ehlers-Danlos syndrome but does not mention the progeroid type.

**Citations detected**: `medquad_ans_0000318-1__dup1`
**Supported citations**: `medquad_ans_0000318-1__dup1`
**Unsupported citations**: _none_

## Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.7000) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the severe form has been o...
2. `medquad_ans_0000416-3` (score=0.4536) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps in the breakdown (me...
3. `medquad_ans_0002601-1` (score=0.4308) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form have minor delays i...
4. `medquad_ans_0000416-5` (score=0.4039) - These resources address the diagnosis or management of glutamate formiminotransferase deficiency: - Baby's First Test - Genetic Testing Registry: Glutamate formiminotransferase deficiency These resources from MedlinePlus offer informatio...
5. `medquad_ans_0002601-2` (score=0.3748) - What are the signs and symptoms of Glutamate formiminotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Glutamate formiminotransferase deficiency. If the information is available, ...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the last two steps of the breakdown of the amino acid histidine and in the production of one of several forms of the vitamin folate [medquad_ans_0000416-3].

**Citations detected**: `medquad_ans_0000416-3`
**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

## Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Top docs**:

1. `medquad_ans_0001047-2` (score=0.7000) - The prevalence of X-linked chondrodysplasia punctata 1 is unknown. Several dozen affected males have been reported in the scientific literature.
2. `medquad_ans_0001169-1` (score=0.5407) - Chondrodysplasia punctata 1, X-linked recessive (CDPX1) is a genetic disorder present from birth that affects bone and cartilage development. On x-ray, affected infants have characteristic spots at the ends of their bones. These spots ar...
3. `medquad_ans_0001048-2` (score=0.5168) - X-linked chondrodysplasia punctata 2 has been estimated to affect fewer than 1 in 400,000 newborns. However, the disorder may actually be more common than this estimate because it is likely underdiagnosed, particularly in females with mi...
4. `medquad_ans_0001047-1` (score=0.4847) - X-linked chondrodysplasia punctata 1 is a disorder of cartilage and bone development that occurs almost exclusively in males. Chondrodysplasia punctata is an abnormality that appears on x-rays as spots (stippling) near the ends of bones ...
5. `medquad_ans_0001169-2` (score=0.4797) - What are the signs and symptoms of Chondrodysplasia punctata 1, X-linked recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 1, X-linked recessive. If the information is...

**Answer**:

Symptoms of Chondrodysplasia punctata 1, X-linked recessive (CDPX1) include characteristic spots (stippled epiphyses) on x-rays at the ends of bones, which typically disappear between ages 2 and 3 [medquad_ans_0001169-1]. Other common features are shortened fingers, a flat nose, breathing abnormalities, hearing loss, abnormalities of the spinal bones in the neck, and delayed intellectual development [medquad_ans_0001169-1]. Additionally, affected individuals may have short stature, unusually short fingertips and ends of toes, and distinctive facial features such as a flattened nose with crescent-shaped nostrils and a flat nasal bridge [medquad_ans_0001047-1]. Some individuals may experience serious or life-threatening complications, including airway stenosis that restricts breathing and spinal abnormalities leading to nerve compression

**Citations detected**: `medquad_ans_0001169-1`, `medquad_ans_0001047-1`
**Supported citations**: `medquad_ans_0001169-1`, `medquad_ans_0001047-1`
**Unsupported citations**: _none_

## Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Top docs**:

1. `medquad_ans_0000165-2__dup1` (score=0.7000) - The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfadizine or solfametho...
2. `medquad_ans_0000132-9` (score=0.5895) - Experts have not yet found a way to prevent Whipple disease.
3. `medquad_ans_0000132-10` (score=0.3989) - A person with Whipple disease and malabsorption may need - a diet high in calories and protein - vitamins - nutritional supplements
4. `medquad_ans_0000132-6` (score=0.3238) - People with Whipple disease may have complications caused by malnutrition, which is due to damaged villi in the small intestine. As a result of delayed diagnosis or treatment, people may experience the following complications in other ar...
5. `medquad_ans_0000132-1__dup3` (score=0.2856) - Whipple disease is a rare bacterial infection that primarily affects the small intestine. The infection may spread to any organ in the body; however, it more commonly affects the - joints - central nervous system, which includes the brai...

**Answer**:

The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfadizine or solfamethoxazole can treat neurological symptoms. Relapsing neurologic Whipple's disease, marked by bouts of worsening of symptoms, is sometimes treated with a combination of antibiotics and weekly injections of interferon gamma, a substance made by the body that activates the immune system [medquad_ans_0000165-2__dup1].

**Citations detected**: `medquad_ans_0000165-2__dup1`
**Supported citations**: `medquad_ans_0000165-2__dup1`
**Unsupported citations**: _none_

## Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Top docs**:

1. `medquad_ans_0000676-4__dup1` (score=0.7000) - How might Bell's palsy be treated?
2. `medquad_ans_0000676-3__dup1` (score=0.5220) - What causes Bell's palsy?
3. `medquad_ans_0000676-2__dup1` (score=0.5106) - What are the symptoms of Bell's palsy?
4. `medquad_ans_0000042-2__dup2` (score=0.2864) - Steroids such as prednisone -- used to reduce inflammation and swelling -- are an effective treatment for Bell's palsy. Antiviral drugs may have some benefit in shortening the course of the disease. Analgesics such as aspirin, acetaminop...
5. `medquad_ans_0000042-3__dup2` (score=0.2818) - The prognosis for individuals with Bell's palsy is generally very good. The extent of nerve damage determines the extent of recovery. With or without treatment, most individuals begin to get better within 2 weeks after the initial onset ...

**Answer**:

Steroids such as prednisone, which reduce inflammation and swelling, are an effective treatment for Bell's palsy. Antiviral drugs may have some benefit in shortening the course of the disease. Analgesics such as aspirin, acetaminophen, or ibuprofen may relieve pain. Individuals should always consult their doctors before taking over-the-counter medicines due to possible drug interactions. Keeping the eye moist and protecting it from debris and injury, especially at night, is important; lubricating eye drops can help. Other therapies such as physical therapy, facial massage, or acupuncture may provide a potential small improvement in facial nerve function and pain [medquad_ans_0000042-2__dup2].

**Citations detected**: `medquad_ans_0000042-2__dup2`
**Supported citations**: `medquad_ans_0000042-2__dup2`
**Unsupported citations**: _none_
